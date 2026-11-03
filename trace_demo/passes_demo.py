"""Step-1 "graph passes + compilation", on a toy model, with numbers.

Mirrors TpuGraphTrainer._make_fx_forward_backward_step:
  1. TRACE    make_fx(fwd_bwd)                 -> FX graph (fwd + loss + bwd)
  2. PASS     simplified activation-recompute  -> FX graph rewritten
              (real recipe: memory_policy="full" -> tag_with_memory_policy_pass
               + selective_activation_remat_pass + apply_tpu_remat_schedule)
  3. COMPILE  torch_tpu StaticCompiler         -> TPU executable
  4. RUN      both executables, compare results and memory

Model: 4 residual MLP blocks   h_next = h + relu(h @ W1_i) @ W2_i
       loss = sum((h_4 - t)^2) / n
"""

import torch
import torch_tpu  # noqa: F401
from torch.fx.experimental.proxy_tensor import make_fx
from torch_tpu._internal.compile.compiler import StaticCompiler

torch.manual_seed(0)
N_LAYERS, B, D, H = 4, 8, 16, 64   # tokens, model dim, hidden dim (H = 4*D)
DEV = "tpu"
aten = torch.ops.aten


def fwd_bwd(W1s, W2s, x, t, n):
  h = x
  for W1, W2 in zip(W1s, W2s):
    h = h + torch.relu(h @ W1) @ W2          # one "layer"
  loss = ((h - t) ** 2).sum() / n
  grads = torch.autograd.grad(loss, [*W1s, *W2s])
  return [loss, *grads]


# ---------------------------------------------------------------- helpers ---
_VIEW_OPS = {aten.detach.default, aten.t.default, aten.expand.default,
             aten.view.default, aten.alias.default}


def nbytes(node):
  """Bytes a node allocates. View/alias ops share their input's memory."""
  if node.target in _VIEW_OPS:
    return 0
  v = node.meta.get("val")
  return v.numel() * v.element_size() if isinstance(v, torch.Tensor) else 0


def shape(node):
  v = node.meta.get("val")
  return tuple(v.shape) if isinstance(v, torch.Tensor) else ()


def backward_start(nodes):
  return next(i for i, nd in enumerate(nodes)
              if nd.target == aten.ones_like.default)


def activation_report(gm, title):
  """Liveness analysis: which forward values stay alive into backward, and the
  peak bytes of live intermediate tensors (graph inputs excluded)."""
  nodes = list(gm.graph.nodes)
  idx = {nd: i for i, nd in enumerate(nodes)}
  bstart = backward_start(nodes)
  last_use = {}
  for nd in nodes:
    for a in nd.all_input_nodes:
      last_use[a] = max(last_use.get(a, -1), idx[nd])

  live, peak, peak_at = 0, 0, 0
  live_act, peak_act = 0, 0             # excluding graph outputs (grads, loss)
  alive = set()
  for i, nd in enumerate(nodes):
    is_out = any(u.op == "output" for u in nd.users)
    if nd.op == "call_function":
      alive.add(nd); live += nbytes(nd)
      if not is_out:
        live_act += nbytes(nd)
      if live > peak: peak, peak_at = live, i
      peak_act = max(peak_act, live_act)
    for a in list(alive):
      if last_use.get(a, -1) <= i and a.op != "output":
        if not any(u.op == "output" for u in a.users):
          alive.discard(a); live -= nbytes(a); live_act -= nbytes(a)

  print(f"\n--- {title} ---")
  print(f"graph nodes: {len(nodes)}  (backward starts at node #{bstart})")
  print("forward values kept alive into backward ('saved activations'):")
  print(f"   {'name':<12}{'op':<14}{'shape':<10}{'made at':>8}{'last use':>10}{'bytes':>8}")
  saved_bytes = 0
  for nd in nodes[:bstart]:
    if nd.op != "call_function":
      continue
    if last_use.get(nd, -1) > bstart:
      op = str(nd.target).split(".")[1]
      print(f"   {nd.name:<12}{op:<14}{str(shape(nd)):<10}{idx[nd]:>8}"
            f"{last_use[nd]:>10}{nbytes(nd):>8}")
      saved_bytes += nbytes(nd)
  print(f"total saved-activation bytes: {saved_bytes}")
  print(f"PEAK live bytes incl. outputs (grads): {peak}  (at node #{peak_at})")
  print(f"PEAK live bytes, activations only     : {peak_act}")
  return peak_act


# ------------------------------------------------------------------ pass ----
def recompute_pass(gm):
  """Simplified 'memory_policy=full': keep only layer boundaries (the residual
  `add` outputs, and the loss `sub`) alive into backward. Any other forward
  value that backward needs (here: the relu outputs, 4*D wide) is re-computed
  right before its first backward use, from the saved layer input."""
  g = gm.graph
  nodes = list(g.nodes)
  idx = {nd: i for i, nd in enumerate(nodes)}
  bstart = backward_start(nodes)
  fwd = [nd for nd in nodes[:bstart] if nd.op == "call_function"]
  bwd = set(nodes[bstart:])
  is_checkpoint = lambda nd: nd.op != "call_function" or nd.target in (
      aten.add.Tensor, aten.sub.Tensor)

  def real_bwd_users(nd):
    return [u for u in nd.users if u in bwd and u.op != "output"
            and u.target != aten.ones_like.default]   # ones_like: shape only

  def chain_of(saved):                  # forward ops back to the checkpoints
    chain, stack = set(), [saved]
    while stack:
      nd = stack.pop()
      if nd in chain or is_checkpoint(nd):
        continue
      chain.add(nd)
      stack.extend(nd.all_input_nodes)
    return chain

  to_recompute = [nd for nd in fwd if not is_checkpoint(nd) and real_bwd_users(nd)]
  # Merge saved values whose chains overlap (e.g. relu and its detach alias),
  # so each layer's intermediates are recomputed once.
  groups = []
  for saved in to_recompute:
    chain = chain_of(saved)
    for grp in groups:
      if grp["chain"] & chain:
        grp["chain"] |= chain
        grp["saved"].append(saved)
        break
    else:
      groups.append({"chain": chain, "saved": [saved]})

  for grp in groups:
    chain = sorted(grp["chain"], key=idx.get)
    users = sorted({u for s in grp["saved"] for u in real_bwd_users(s)},
                   key=idx.get)
    env = {}
    with g.inserting_before(users[0]):
      for nd in chain:
        env[nd] = g.node_copy(nd, lambda a: env.get(a, a))
        env[nd].meta["recomputed_from"] = nd.name
    for s in grp["saved"]:
      for u in real_bwd_users(s):
        u.replace_input_with(s, env[s])
    print(f"   recompute [{' -> '.join(c.name for c in chain)}] "
          f"right before backward node {users[0].name}")
  g.lint()
  gm.recompile()
  return gm


# ------------------------------------------------------------------- main ---
W1s = [(torch.randn(D, H) * 0.1).to(DEV).requires_grad_(True) for _ in range(N_LAYERS)]
W2s = [(torch.randn(H, D) * 0.1).to(DEV).requires_grad_(True) for _ in range(N_LAYERS)]
x, t = torch.randn(B, D).to(DEV), torch.randn(B, D).to(DEV)
n = torch.tensor(float(B)).to(DEV)
args = (W1s, W2s, x, t, n)

print("=" * 72); print("1. TRACE"); print("=" * 72)
gm_base = make_fx(fwd_bwd, tracing_mode="fake")(*args)
peak_before = activation_report(gm_base, "BEFORE passes")

print("\n" + "=" * 72); print("2. GRAPH PASS: activation recompute"); print("=" * 72)
gm_remat = make_fx(fwd_bwd, tracing_mode="fake")(*args)
gm_remat = recompute_pass(gm_remat)
peak_after = activation_report(gm_remat, "AFTER recompute pass")
print(f"\npeak activation memory: {peak_before} -> {peak_after} bytes "
      f"({100 * (peak_before - peak_after) / peak_before:.0f}% less), "
      f"cost: {len(list(gm_remat.graph.nodes)) - len(list(gm_base.graph.nodes))} "
      "extra nodes (re-run forward ops)")

print("\nrecomputed nodes now living inside backward (new name <- original):")
nodes_r = list(gm_remat.graph.nodes)
bs = backward_start(nodes_r)
for i, nd in enumerate(nodes_r):
  if "recomputed_from" in nd.meta:
    args_s = ", ".join(str(a) for a in nd.args)[:40]
    print(f"   #{i:<3} {nd.name:<10} = {str(nd.target):<22}({args_s})"
          f"   <- copy of forward '{nd.meta['recomputed_from']}'")
print(f"   (backward starts at node #{bs})")

print("\n" + "=" * 72); print("3. COMPILE with torch_tpu StaticCompiler"); print("=" * 72)
flat = [*W1s, *W2s, x, t, n]
exes = {}
for name, gm in (("base", gm_base), ("remat", gm_remat)):
  ex_in = [nd.meta["val"] for nd in gm.graph.nodes if nd.op == "placeholder"]
  exes[name] = StaticCompiler()(gm, ex_in, is_fwd=True)
  print(f"compiled '{name}' graph: {len(list(gm.graph.nodes))} FX nodes -> TPU executable")

print("\n" + "=" * 72); print("4. RUN both on TPU and compare"); print("=" * 72)
out_base = exes["base"](*flat)
out_remat = exes["remat"](*flat)
print(f"loss  base={out_base[0].item():.6f}  remat={out_remat[0].item():.6f}")
max_diff = max((a - b).abs().max().item() for a, b in zip(out_base, out_remat))
print(f"max |difference| over loss + all {len(out_base) - 1} gradients: {max_diff:.3e}")
print("dL/dW1_0[0,:4] base :", [round(v, 5) for v in out_base[1][0, :4].tolist()])
print("dL/dW1_0[0,:4] remat:", [round(v, 5) for v in out_remat[1][0, :4].tolist()])
