# trace_demo - GraphTrainer toy demos

Runnable toy scripts (and their captured logs) that walk through how
torchtitan's TPU `GraphTrainer` traces, transforms, compiles and runs a
training step. Copied from `~/sft/torchtitan/trace_demo/`.

Run on the TPU VM `wenjung-v6e-dev` (`--zone=europe-west4-a --project=tpu-pytorch`)
with `~/venv_torch212/bin/python` (torch 2.12, torch_tpu 0.1.1.dev20260912,
libtpu 0.0.47, jax 0.11.0) unless noted.

Related notes:
- [[TorchTitan Graph Trainer on TPU - Custom Ops, Kernels, make_fx vs Dynamo]]

| Script | Log | What it shows |
|---|---|---|
| `trace_demo.py` | `trace_demo_output.log` | CPU-only: `make_fx` traces fwd + loss + `autograd.grad` of a 1-weight linear model into one joint FX graph |
| `trace_demo_tpu.py` | `trace_demo_tpu_output.log` | Same on TPU: phase 1 trace (FakeTensors, host), phase 2 `StaticCompiler` (FX -> StableHLO -> PJRT), phase 3 run the executable for two "chips" |
| `fx_vs_mlir.py` | `fx_vs_mlir_output.log` | The same graph as FX nodes vs the lowered StableHLO/MLIR |
| `passes_demo.py` | `passes_demo_output.log` | 4-layer residual MLP: trace -> simplified activation-recompute pass -> compile both graphs -> compare loss/grads and peak activation memory |
| `dump_nodes_after_pass.py` | `dump_nodes_after_pass_output.log` | Every node after the recompute pass with full metadata: op, args, shape, phase, region (layer), seq_nr, autograd_backward, parameter-gradient tag, recompute origin, users; plus `apply_tpu_remat_schedule` boundary analysis |
| (user-run) | `dump_nodes_output.log` | Node dump before the pass |
| `barrier_demo.py` | `barrier_demo_output.log` | `torch` optimization_barrier demo (fails on this wheel: no `torch.tpu.ops.optimization_barrier`) |
| `barrier_demo_jax.py` | `barrier_demo_jax_output.log` | JAX `lax.optimization_barrier`: values unchanged, StableHLO `optimization_barrier`, removed after XLA optimization |
| `dynamo_vs_makefx.py` | `dynamo_vs_makefx_output.log`, `dynamo_vs_makefx_side_by_side.txt` | Same toy model through `torch.compile` (Dynamo -> `aot_autograd` fwd/bwd graphs) vs torchtitan's `minimal_fx_tracer` (one joint graph): op counts, saved-tensor boundary, placeholders, metadata, numerics. Needs a minimal torchtitan copy on `PYTHONPATH` (`TTMIN`, containing `graph_trainer/make_fx_tracer.py` + `dynamic_shapes.py`). This run fell back to the VM CPU because TPU init failed |

Notes:
- The VM wheel lacks `torch.compiler._patch_engine_backward`, so make_fx
  backward nodes all share the last forward `seq_nr`; `dump_nodes_after_pass.py`
  uses a fallback to infer backward regions.
- `dump_nodes_after_pass.py` imports the top half of `passes_demo.py`, so keep
  them in the same folder.
