"""Same toy as trace_demo.py, but run the way TpuGraphTrainer does on TPU.

Phases (and where each runs):
  1. TRACE    make_fx on FakeTensors            -> CPU host (Python only)
  2. COMPILE  torch_tpu StaticCompiler(gm, ...)  -> CPU host (builds XLA program)
  3. RUN      executable(*tpu_inputs)            -> TPU device
  4. READ     .item() / .tolist()                -> TPU -> host copy
"""

import time

import torch
import torch_tpu  # noqa: F401  registers the "tpu" device
from torch.fx.experimental.proxy_tensor import make_fx
from torch_tpu._internal.compile.compiler import StaticCompiler

DEV = "tpu"
python_calls = 0


def fwd_bwd_step(W, x, t, n):
  global python_calls
  python_calls += 1
  print(f"    [python body running, call #{python_calls}; "
        f"x is {type(x).__name__} on device={x.device}]")
  y = x @ W
  loss = ((y - t) ** 2).sum() / n
  (grad_W,) = torch.autograd.grad(loss, [W])
  return [loss, grad_W]


def to_dev(*ts):
  return [t.to(DEV) for t in ts]


W = torch.tensor([[0.5], [-1.0], [2.0], [0.0]]).to(DEV).requires_grad_(True)
x0, t0, n = to_dev(
    torch.tensor([[1.0, 2.0, 0.0, 1.0]]), torch.tensor([[1.0]]),
    torch.tensor(2.0))
x1, t1 = to_dev(torch.tensor([[0.0, 1.0, 1.0, 0.0]]), torch.tensor([[0.0]]))
print(f"inputs live on: W={W.device}, x0={x0.device}")

# ---- 1. TRACE (host) -------------------------------------------------------
print("\n=== PHASE 1: TRACE (CPU host, FakeTensors, no TPU compute) ===")
t_start = time.perf_counter()
gm = make_fx(fwd_bwd_step, tracing_mode="fake")(W, x0, t0, n)
print(f"trace time: {time.perf_counter() - t_start:.3f}s, "
      f"graph nodes: {len(list(gm.graph.nodes))}")
example_inputs = [
    node.meta["val"] for node in gm.graph.nodes if node.op == "placeholder"
]
print("example inputs recorded:",
      [(type(v).__name__, tuple(v.shape), str(v.device)) for v in example_inputs])

# ---- 2. COMPILE (host) -----------------------------------------------------
print("\n=== PHASE 2: COMPILE with StaticCompiler (CPU host builds XLA) ===")
t_start = time.perf_counter()
executable = StaticCompiler()(gm, example_inputs, is_fwd=True)
print(f"StaticCompiler ready in {time.perf_counter() - t_start:.3f}s "
      "(PJRT may finish compiling lazily on the first execute)")

# ---- 3. RUN (TPU) ----------------------------------------------------------
print("\n=== PHASE 3: RUN compiled executable (TPU) ===")
calls_before = python_calls
for name, (x, t) in {"chip0": (x0, t0), "chip1": (x1, t1)}.items():
  t_start = time.perf_counter()
  loss, grad = executable(W, x, t, n)          # enqueue on TPU (async)
  t_enqueue = time.perf_counter() - t_start
  # ---- 4. READ back (TPU -> host) -----------------------------------------
  t_start = time.perf_counter()
  loss_v, grad_v = loss.item(), grad.flatten().tolist()   # forces device->host
  t_read = time.perf_counter() - t_start
  print(f"{name}: outputs on device={loss.device}; enqueue {t_enqueue*1e3:.1f} ms, "
        f"read-back {t_read*1e3:.1f} ms -> loss={loss_v:.4f}, dL/dW={grad_v}")
print(f"python body calls during RUN: {python_calls - calls_before}")
