---
tags: [torchtitan, moe, tpu, expert-parallel, token-dispatcher, learning-notes]
repo: google-pytorch/torchtitan (local: ~/sft/torchtitan)
updated: 2026-10-23
---

# MoE Token Dispatcher on TPU - TorchTitan Walkthrough

Running Q&A notes on how a TorchTitan MoE layer works end to end, centered on
`TpuAllToAllTokenDispatcher` / `TpuTokenDispatcherConverter`.
Code paths are relative to the torchtitan repo root.

## Contents
1. [[#1. Big picture - where MoE sits in the transformer]]
2. [[#2. From input_ids (B, S) to x (T, D)]]
3. [[#3. Router - how each token gets K experts and K scores]]
4. [[#4. What an expert is - Expert_e(x)]]
5. [[#5. Shape glossary - T, K, N, E, EP, e, R, D]]
6. [[#6. Dispatch and combine - full numerical example (EP=2)]]
7. [[#7. Why combine reverses R -> N -> T]]
8. [[#8. Parallelism - where EP ranks come from]]
9. [[#9. Why TP is paired with SP, and why EP needs token-sharded input]]
10. [[#10. TPU dispatcher - TpuAllToAllTokenDispatcher and converter]]
11. [[#11. Pallas grouped matmul]]
12. [[#12. Code map - router example step by step]]

---

## 1. Big picture - where MoE sits in the transformer

```
input_ids (B, S)
  -> flatten to tokens (T,) + positions (T,)
  -> embedding                      -> x (T, D)
  -> for each of L layers (Qwen3 block, torchtitan/models/qwen3/model.py L61-L68):
        x = x + Attention(attention_norm(x))   # tokens exchange information
        x = x + MoE(ffn_norm(x))               # per-token transform
  -> final norm -> lm_head -> logits (T, vocab) -> cross-entropy
```

- `x` is the **residual stream**: one D-dim vector per token, carried through all layers.
- Every sublayer returns a `(T, D)` update that is **added** to `x`. MoE does not
  "update T with weights"; it computes a per-token update. Weights change only in
  the optimizer step.
- Attention is the only place tokens interact. FFN/MoE is row-independent, which is
  why tokens can be sorted, replicated and shipped across chips.
- MoE is a drop-in replacement for the dense FFN: same `(T, D) -> (T, D)` contract,
  `E` FFNs ("experts"), but each token only uses `K` of them.

MoE forward (`torchtitan/models/common/moe.py` `MoE.forward` L916-L964):
```
router(x_TD)  -> topk_scores_TK, topk_expert_ids_TK, routing_map_TE
counts_E = routing_map_TE.sum(0)
routed_experts(x_TD, scores, ids, counts):          # RoutedExperts.forward L487-L519
    dispatch  -> T -> N -> R
    experts   -> grouped matmul (one FFN per expert)
    combine   -> R -> N -> T  (score-weight, sum K)
return out_TD
```

---

## 2. From input_ids (B, S) to x (T, D)

Example: `B = 2`, `S = 4`, vocab `V = 10`, `D = 3`.

```
input_ids (2, 4) = [[5, 2, 7, 1],     # "the cat sat ."
                    [3, 3, 9, 0]]     # "a a dog <eos>"

flatten:
tokens    (8,) = [5, 2, 7, 1, 3, 3, 9, 0]      T = B * S = 8
positions (8,) = [0, 1, 2, 3, 0, 1, 2, 3]      reset to 0 = new sequence
segment_ids    = [0, 0, 0, 0, 1, 1, 1, 1]      (decoder.py L252-L254)
```

Embedding = table lookup in `tok_embeddings.weight (V, D)`: `x_TD = W[tokens]`
(`torchtitan/models/common/decoder.py` L246).

| row t | id | word | x_TD[t] |
|---|---|---|---|
| 0 | 5 | the | `[0.1, 0.0, 0.9]` |
| 1 | 2 | cat | `[0.5, 0.2, 0.1]` |
| 2 | 7 | sat | `[-0.2, 0.3, 0.4]` |
| 3 | 1 | . | `[0.3, -0.1, 0.0]` |
| 4 | 3 | a | `[-0.4, 0.6, 0.2]` |
| 5 | 3 | a | `[-0.4, 0.6, 0.2]` |
| 6 | 9 | dog | `[0.7, -0.5, 0.3]` |
| 7 | 0 | eos | `[0.0, 0.1, -0.2]` |

- Rows 4 and 5 are identical until attention + RoPE give them different context.
- The model receives a flat `[T]` stream (e.g. `base_distributed_device_test.py` L363:
  `model(inputs.reshape(-1), positions)`).
- TPU kernels needing `(B, S, ...)` use `torchtitan/experiments/tpu/unflatten_shim.py`
  (requires `T % S == 0`, no varlen packing).

Real Qwen3-30B-A3B recipe per rank: `(B, S) = (6, 2048)`, `T = 12288`, `D = 2048`,
embedding `(151936, 2048)`, `x_TD` in BF16 is about 48 MiB.

---

## 3. Router - how each token gets K experts and K scores

`TokenChoiceTopKRouter.forward` (`moe.py` L647-L726). Qwen3: `score_func="softmax"`,
`route_norm=True`, `E = 128`, `K = 8` (`models/qwen3/__init__.py` L178-L185, L554-L555).

```
1. logits_TE = x_TD @ W_gate.T            # RouterGateLinear, FP32 output
2. scores_TE = softmax(logits_TE)          # over all E
3. ids_TK    = topk(scores_TE + expert_bias_E, K).indices   # bias only for choice
4. topk_scores_TK = scores_TE.gather(-1, ids_TK)            # original scores
5. topk_scores_TK /= topk_scores_TK.sum(-1)                 # route_norm
6. topk_scores_TK *= route_scale                            # 1.0 for Qwen3
7. routing_map_TE = one_hot(ids_TK)  -> counts_E = routing_map_TE.sum(0)
```

### Worked example: one token, `E = 4`, `K = 2`, `D = 3`, `x = [0.1, 0.0, 0.9]`

| expert | W_gate[e] | logit |
|---|---|---|
| e0 | `[1, 0, 1]` | `1.0` |
| e1 | `[0, 1, 2]` | `1.8` |
| e2 | `[-1, 0, 0]` | `-0.1` |
| e3 | `[2, 1, 0]` | `0.2` |

```
softmax: exp = [2.718, 6.050, 0.905, 1.221], sum = 10.894
scores   = [0.250, 0.555, 0.083, 0.112]
top-2    = ids [1, 0], scores [0.555, 0.250]
route_norm: [0.555, 0.250] / 0.805 = [0.689, 0.311]
=> y = 0.689 * Expert_1(x) + 0.311 * Expert_0(x)
```

With load-balancing bias `expert_bias_E = [0, -0.4, 0, +0.2]`:
```
scores + bias = [0.250, 0.155, 0.083, 0.312] -> top-2 = [3, 0]
gather ORIGINAL scores -> [0.112, 0.250] -> norm -> [0.309, 0.691]
```
Bias changes *which* experts, not *how much* each contributes.

Bias update (non-learned, optimizer pre-step hook,
`components/optimizer/optimizer.py` L549-L554):
`delta = coeff * sign(mean(tokens_per_expert) - tokens_per_expert)`, centered, then added.
The `30B-A3B-global-aux` recipe uses an aux loss instead (`load_balance_coeff=None`).

### How the router learns
- `topk` indices are not differentiable.
- Gradient reaches `W_gate` only through `topk_scores_TK`:
  `dScore = sum_D(dout * expert_output)` in combine -> route_norm -> gather -> softmax -> gate.
- `_select_experts` runs under `remat.region(..., recompute=False)` so activation
  checkpoint replay reuses the same routing decision.

---

## 4. What an expert is - Expert_e(x)

Each expert is a SwiGLU FFN with private weights:
```
Expert_e(x) = W2_e @ ( silu(W1_e @ x) * (W3_e @ x) )
  W1_e, W3_e : (F, D)    W2_e : (D, F)    silu(z) = z * sigmoid(z)
```

Example `D = 3`, `F = 2`, `x = [0.1, 0.0, 0.9]`:
```
W1_1 = [[1,0,1],[0,1,0]]   W3_1 = [[1,1,0],[0,0,2]]   W2_1 = [[1,0],[0,1],[1,-1]]
gate = W1_1 @ x = [1.0, 0.0]  -> silu -> [0.731, 0.0]
up   = W3_1 @ x = [0.1, 1.8]
h    = [0.0731, 0.0]
out  = W2_1 @ h = [0.0731, 0.0, 0.0731]
```

Code: `GroupedExperts` (`moe.py` L387-L470). Weights stacked as
`w1_EFD (E, F, D)`, `w2_EDF (E, D, F)`, `w3_EFD (E, F, D)`.
```python
offsets_E = cumsum(num_tokens_per_expert_E)       # row boundaries per expert
h = silu(grouped_mm(x_RD, w1)) * grouped_mm(x_RD, w3)
return grouped_mm(h, w2)
```

Qwen3-30B-A3B: `D = 2048`, `F = 768`, `E = 128`, `K = 8`, 48 layers.
- per expert: `3 * 2048 * 768 = 4.7M` params
- per layer: `128 * 4.7M = 604M`; all layers about 29B (the "30B")
- active per token per layer: `8 * 4.7M = 38M` (part of "A3B")

---

## 5. Shape glossary - T, K, N, E, EP, e, R, D

Suffix convention: `x_TD`, `routed_input_ND`, `routed_input_RD`, `counts_E`, `counts_e`.

| Symbol | Meaning (per rank) |
|---|---|
| `T` | tokens on this rank |
| `K` | top_k experts per token |
| `N = T * K` | routes this rank **sends** (fixed) |
| `E` | total experts (global) |
| `EP` | expert-parallel ranks |
| `e = E / EP` | experts **owned** by this rank |
| `R` | rows this rank **receives** to compute (data dependent) |
| `D` | hidden size |

With EP = 1: `R = N = T * K` and `e = E`.

---

## 6. Dispatch and combine - full numerical example (EP=2)

Setup: `EP = 2`, `E = 4`, `e = 2`, `T = 3`, `K = 2`, `N = 6`.
Rank 0 owns experts {0, 1}; rank 1 owns {2, 3}.
Rank 0 tokens `A0, A1, A2`; rank 1 tokens `B0, B1, B2`.

### Step 0 - router output
```
rank 0: expert_ids_TK = [[1,0],[2,1],[0,3]]   counts_E = [2,2,1,1]
rank 1: expert_ids_TK = [[0,1],[0,3],[1,2]]   counts_E = [2,2,1,1]
```

### Step 1 - `_local_reorder` (T, D) -> (N, D)
Rank 0: flat ids `[1,0,2,1,0,3]` -> stable argsort `[1,4,0,3,2,5]` ->
`token_idx = perm // K = [0,2,0,1,1,2]`.
```
rank 0 routed_input_ND: A0e0, A2e0, A0e1, A1e1, A1e2, A2e3
rank 1 routed_input_ND: B0e0, B1e0, B0e1, B2e1, B2e2, B1e3
```

### Step 2 - count exchange (`_token_count_exchange`, `_sync_token_count_exchange`)
`counts_E.view(EP, e)`; row r goes to rank r via all-to-all.
```
rank 0 num_global_tokens_per_local_expert_E = [2,2, 2,2]   # (from r0: e0,e1),(from r1: e0,e1)
rank 1 num_global_tokens_per_local_expert_E = [1,1, 1,1]

        input_splits (send to r0,r1)   output_splits (recv from r0,r1)
rank 0  [4, 2]                         [4, 4]  -> R = 8
rank 1  [4, 2]                         [2, 2]  -> R = 4
```
Both send `N = 6`, but receive different `R`. `output_splits` requires a D2H sync.

### Step 3 - data all-to-all (N, D) -> (R, D), rank-major
```
rank 0 (R=8): [A0e0, A2e0, A0e1, A1e1 | B0e0, B1e0, B0e1, B2e1]
rank 1 (R=4): [A1e2, A2e3 | B2e2, B1e3]
```

### Step 4 - `_permute` rank-major -> expert-major
```
t_mat = [[2,2],[2,2]], input_starts = [[0,2],[4,6]]
segments (e0,r0),(e0,r1),(e1,r0),(e1,r1): lens [2,2,2,2], starts [0,4,2,6]
permuted_indices = [0,1,4,5,2,3,6,7]
routed_input_RD = [A0e0, A2e0, B0e0, B1e0 | A0e1, A1e1, B0e1, B2e1]
num_global_tokens_per_local_expert_e = [4, 4]   # group_sizes for grouped matmul
```
Rank 1: `[A1e2, B2e2 | A2e3, B1e3]`, group_sizes `[2, 2]`.

### Step 5 - experts
```
rank 0 routed_output_RD = [E0(A0), E0(A2), E0(B0), E0(B1) | E1(A0), E1(A1), E1(B0), E1(B2)]
rank 1 routed_output_RD = [E2(A1), E2(B2) | E3(A2), E3(B1)]
```

### Step 6 - combine (R, D) -> (N, D) -> (T, D)
1. `_unpermute`: expert-major -> rank-major
   `[E0(A0), E0(A2), E1(A0), E1(A1) | E0(B0), E0(B1), E1(B0), E1(B2)]`
2. reverse all-to-all (splits swapped): rank 0 gets back its own `N = 6` rows,
   in Step-1 order: `[E0(A0), E0(A2), E1(A0), E1(A1), E2(A1), E3(A2)]`
3. multiply by `topk_scores_experts_sorted_N`, scatter_add with
   `token_idx = [0,2,0,1,1,2]`:
```
out[A0] = s(A0,e0)*E0(A0) + s(A0,e1)*E1(A0)
out[A1] = s(A1,e1)*E1(A1) + s(A1,e2)*E2(A1)
out[A2] = s(A2,e0)*E0(A2) + s(A2,e3)*E3(A2)
```

---

## 7. Why combine reverses R -> N -> T

After experts, each token's K partial results are:
1. on the **wrong rank** (e.g. `E0(B0)` sits on rank 0, B0 lives on rank 1),
2. in the **wrong order** (expert-major, not token order),
3. **not yet weighted or summed**.

| Shape | Order | Purpose |
|---|---|---|
| `(T, D)` | one row per token, original order | what the transformer consumes |
| `(N, D)` | K copies per token, expert-sorted, home rank | one row per route, ready to ship |
| `(R, D)` | all routes for my experts, expert-major | contiguous segments for grouped GEMM |

Dispatch: `T -> N -> R` (replicate K, ship to owner, group by expert).
Combine: `R -> N -> T` (ungroup, ship home, weight + sum K).
Backward walks the same path in reverse (another pair of all-to-alls).

---

## 8. Parallelism - where EP ranks come from

EP has no devices of its own. `ParallelDims` (`torchtitan/distributed/parallel_dims.py`):
- world = `dp_replicate * dp_shard * cp * tp * pp`
- constraint: `(dp_shard * cp * tp) % ep == 0` (L161-L166)
- also `num_experts % ep == 0` (`models/common/decoder.py` L183-L186)
- `efsdp = dp_shard * cp * tp / ep` (L256)

Two views of the same devices (`build_mesh` L254-L307):
```
dense  mesh: (pp, dp_replicate, fsdp,  tp)    fsdp = dp_shard * cp
sparse mesh: (pp, dp_replicate, efsdp, ep)
```
`ep` is innermost, so an EP group takes consecutive ranks: fills `tp` first, then
`cp`, then `dp_shard`. EP never spans `dp_replicate` or `pp`.

Why tokens differ across EP ranks: those ranks are DP ranks (different batches),
CP ranks (different sequence chunks), or TP+SP ranks (different sequence shards).

`dp_shard` alone need **not** be a multiple of `ep`:

| dp_shard | cp | tp | ep | valid? |
|---|---|---|---|---|
| 8 | 1 | 1 | 2 | yes |
| 2 | 1 | 4 | 8 | yes (dp_shard not multiple of ep) |
| 3 | 1 | 2 | 2 | yes |
| 4 | 1 | 1 | 8 | no |

### Example: 8 chips, dp_shard = 4, tp = 2, ep = 2 (efsdp = 4)
```
dense (fsdp=4, tp=2)          sparse (efsdp=4, ep=2)
        tp0  tp1                      ep0  ep1
fsdp0 [ r0 , r1 ]             efsdp0 [ r0 , r1 ]
fsdp1 [ r2 , r3 ]             efsdp1 [ r2 , r3 ]
fsdp2 [ r4 , r5 ]             efsdp2 [ r4 , r5 ]
fsdp3 [ r6 , r7 ]             efsdp3 [ r6 , r7 ]
```
- TP groups = EP groups = {r0,r1}, {r2,r3}, ... (only because ep == tp)
- r0: seq0[0:4], r1: seq0[4:8] (SP split, `T = 4`, `N = 8` each)
- experts 0,1 on ep0 ranks (r0,r2,r4,r6); experts 2,3 on ep1 ranks; each FSDP-sharded
  4 ways over efsdp and all-gathered before compute
- experts are NOT tensor-parallel here; TP chips act as EP
- expert grads reduce-scatter over efsdp; router aux counts all-reduce over tp (+cp), not dp

Recipes:
- `qwen3_moe_30b`: `dp_shard=-1, ep=128` (EP over FSDP ranks, all-to-all)
- `_qwen3_moe_30b_fsdp_only_full_ac`: `ep=1` (no all-to-all, local dispatch)

---

## 9. Why TP is paired with SP, and why EP needs token-sharded input

**TP alone**: after the row-parallel matmul, an all-reduce leaves activations replicated
on every TP rank; norms/residual are computed `tp` times and stored `tp` times.

**TP + SP**: split the all-reduce into reduce-scatter + all-gather; between them each
rank holds `S / tp` tokens.

| | TP only | TP + SP |
|---|---|---|
| comm per sublayer | 1 all-reduce | 1 all-gather + 1 reduce-scatter (same volume) |
| norm/residual activation memory | `S * D` | `S * D / tp` |
| norm/residual compute | done tp times | done once |

So SP is a free optimization for dense layers (default in TorchTitan, not mandatory).

**For MoE**:
- EP off, expert-TP: every rank must see all tokens (each holds an F-slice of every
  expert). Replicated tokens are correct here.
- EP borrows the tp axis: each rank owns whole experts. Replicated tokens would send
  every route twice (double compute and traffic, needs dedup). With SP each token has
  exactly one owner, so each route is sent once. Hence under EP the gate emits
  `dense_sequence_parallel_placement` (`moe.py` L824-L829).

---

## 10. TPU dispatcher - TpuAllToAllTokenDispatcher and converter

File: `torchtitan/experiments/tpu/token_dispatcher.py`. Same forward math as upstream;
replaces autograd backwards that are expensive on TPU.

Base classes (`torchtitan/models/common/token_dispatcher.py`):
- `LocalTokenDispatcher` (EP=1): `_local_reorder` (argsort + gather), `combine`
  (score + `deterministic_scatter_add` with an index expanded to `[N, D]`)
- `AllToAllTokenDispatcher` (EP>1): reorder + count exchange + all-to-all + `_permute`;
  falls back to local when `ep_mesh is None`

Problems with native autograd:
- `x_TD[token_idx]` repeats rows K times -> backward is a collision scatter
- scatter_add backward gathers with an `[N, D]` index (about 768 MiB per layer at prod shape)

### 10a. `local_reorder` / `_LocalReorder` (EP=1 and EP>1)
- forward: same gather; saves only `inverse_N = argsort(permutation_N)`
- backward: `index_select(grad, inverse_N)` -> reshape `(T, K, D)` -> FP32 sum over K;
  scores gathered with the same inverse
- example: perm `[1,4,0,3,2,5]` -> inverse `[2,0,4,3,1,5]`
- forward sort stays stable (changing equal-expert order shifted training trajectory)

### 10b. `stable_grouped_reduce` / `_StableGroupedReduce` (EP=1 combine only)
- forward: stable argsort of `token_idx` groups each token's K rows -> reshape
  `(T, K, D)` -> FP32 sum
- backward: `grad_out[index_N]`, saves only 1-D index
- example: `argsort([0,2,0,1,1,2]) = [0,2,3,4,1,5]`
- relies on exactly K rows per token (dropless EP=1). EP>1 delegates to `super().combine`

### 10c. `scale_route_weights` / `_ScaleRouteWeights` (opt-in `fuse_route_score_backward`)
- saves BF16 expert output instead of an FP32 `[N, D]`
- `dScore` via `route_score_grad` (Pallas `jax_op`,
  `experiments/tpu/kernels/route_score.py`): `sum(grad.f32 * routed.f32, axis=1)`
- enabled in `qwen3_moe_30b_fsdp_only_attention_parity_benchmark` and
  `_fuse_route_score_backward` (about 1.12 s/step on Q235 @ 512 ranks)

### 10d. `TpuTokenDispatcherConverter`
`ModelConfigConverter` that rewrites configs before build:
- traverses `AllToAllTokenDispatcher.Config`
- replaces only exact-type matches (skips subclasses)
- `derive(config, TpuAllToAllTokenDispatcher.Config)` copies `num_experts`, `top_k`
- used in recipes with `TpuGroupedExpertsConverter`

| Piece | Path | Forward | Backward win |
|---|---|---|---|
| `local_reorder` | EP=1, EP>1 | same | inverse gather + (T,K,D) sum, no collision scatter |
| `stable_grouped_reduce` | EP=1 | grouped reshape-sum | 1-D row gather, no [N,D] index |
| `scale_route_weights` | EP=1, opt-in | same | no saved FP32 [N,D], fused dScore |
| converter | config time | swaps class | - |

---

## 11. Pallas grouped matmul

**Grouped mm**: one kernel computing `E` GEMMs over contiguous row segments with
data-dependent sizes (`group_sizes`), no padding, no Python loop.
```
rows [0, g0)       @ W[0]
rows [g0, g0+g1)   @ W[1]
...
```

**Pallas**: JAX's TPU kernel language (like Triton for GPU). Kernel = MegaBlox:
- `gmm`: `Y_e = X_e @ W_e`; also `dX_e = dY_e @ W_e.T` (`transpose_rhs=True`)
- `tgmm`: `dW_e = X_e.T @ dY_e`, FP32 accumulate, store in weight dtype

Wrapper: `experiments/tpu/kernels/expert_segment_matmul.py` (custom ops
`tt_experts::grouped_mm`, `group_sizes` stays an on-device int32 tensor -> no recompile
when routing changes).

Why: native `torch._grouped_mm` mode-2 dWeight is wrong at M=4096
(torch_tpu issue #2668), plus tunable tiles.

`TpuGroupedExperts.Config` (`experiments/tpu/grouped_experts.py` L109-L122):

| Flag | Meaning |
|---|---|
| `use_pallas_grouped_mm` | MegaBlox instead of `torch._grouped_mm` |
| `use_pallas_paired_grouped_mm` | W1 and W3 in one kernel (EP=1 only) |
| `pallas_grouped_mm_tile_m/k/n` | forward / dInput tiles (0 = full dim) |
| `pallas_grouped_mm_tgmm_tiling` | dWeight tiles (paired path needs n=0) |
| `pallas_grouped_mm_w2_tgmm_tiling` | separate dWeight tiling for W2 |
| `use_kernel_native_weight_layout` | store `(E, K, N)`, checkpoint stays canonical |

Q30 FSDP recipe: all on, `tile_m=256`, `tgmm_tiling=(256, 0, 0)`.

---

## 12. Code map - router example step by step

| Step | Code |
|---|---|
| input `ffn_norm(x)` into MoE | `models/qwen3/model.py` L66 |
| MoE calls router | `models/common/moe.py` L935-L939 |
| 1. gate logits | `moe.py` L663; `models/common/linear.py` L70-L73, L112-L116 |
| 2. softmax | `moe.py` L669-L670 |
| 3. choose top-K (+bias) | `moe.py` L682-L687, L635-L645 |
| 4. gather original scores | `moe.py` L689-L691 |
| 5. route_norm / route_scale | `moe.py` L693-L696 |
| 6. routing map, aux loss | `moe.py` L702-L720 |
| counts_E | `moe.py` L940, L947-L949 |
| 7. dispatch -> experts -> combine | `moe.py` L951-L956, L500-L518 |
| scores sorted with tokens | base `token_dispatcher.py` L93-L100; TPU L190-L194 |
| weight + sum K | base `token_dispatcher.py` L156-L169; TPU L269-L283 |
| residual add | `models/qwen3/model.py` L66 |
| bias update | `components/optimizer/optimizer.py` L517-L560 |
| dScore backward | base autograd L158-L161; TPU `_ScaleRouteWeights.backward`, `route_score_grad` |
| dW_gate | `models/common/linear.py` L82-L98 |
