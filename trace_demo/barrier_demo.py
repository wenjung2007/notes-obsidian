"""What optimization_barrier(x_1, add_6, mm_16, mm_18) does, on real TPU tensors.

1. VALUES: outputs are bit-identical to inputs (it is an identity op).
2. COMPILED FORM: it lowers to `stablehlo.optimization_barrier`, which tells XLA
   "all outputs become available only after ALL inputs are computed; do not move
   or fuse computation across this point".
"""

import torch
import torch_tpu  # noqa: F401
from torch.fx.experimental.proxy_tensor import make_fx
from torch_tpu._internal.compile.compiler import StaticCompiler
from torch_tpu._internal.compile.debug import TpuCompileDebug

torch.manual_seed(0)
dev = "tpu"
# Same shapes as the toy graph's barrier inputs for region 0.
x_1 = torch.randn(8, 16).to(dev)     # layer 0 input (saved primal)
add_6 = torch.randn(8, 16).to(dev)   # dL/d(layer 0 output)  (cotangent)
mm_16 = torch.randn(64, 16).to(dev)  # layer 1 dW2           (ordering only)
mm_18 = torch.randn(16, 64).to(dev)  # layer 1 dW1           (ordering only)
w1s_1 = torch.randn(16, 64).to(dev)

# torch.tpu.ops.optimization_barrier (public wrapper) is missing in this wheel;
# call the underlying op it wraps: torch.ops.tpu._optimization_barrier.
barrier_op = torch.ops.tpu._optimization_barrier
if not torch._C._dispatch_has_kernel_for_dispatch_key("tpu::_optimization_barrier", "Meta"):
  # Same shim as torchtitan tpu_backend_pass._ensure_optimization_barrier_fake,
  # so make_fx can trace it with FakeTensors.
  torch.library.register_fake(
      "tpu::_optimization_barrier",
      lambda inputs: [torch.empty_strided(v.shape, v.stride(), dtype=v.dtype,
                                          device=v.device) for v in inputs])

print("=== 1. eager: values pass through unchanged ===")
outs = barrier_op([x_1, add_6, mm_16, mm_18])
for name, a, b in zip(["x_1", "add_6", "mm_16", "mm_18"],
                      [x_1, add_6, mm_16, mm_18], outs):
  print(f"{name:<6} in {tuple(a.shape)} -> out {tuple(b.shape)}  "
        f"identical={torch.equal(a.cpu(), b.cpu())}")


def layer0_entry(x_1, add_6, mm_16, mm_18, w1s_1):
  """Mimics the rewritten layer-0 entry: recompute + first backward op read the
  barrier OUTPUTS; mm_16/mm_18 outputs are unused (dependency only)."""
  x_b, add_6_b, _, _ = barrier_op([x_1, add_6, mm_16, mm_18])
  mm_23 = x_b @ w1s_1              # recompute starts from the barriered x_1
  relu_4 = torch.relu(mm_23)
  mm_20 = relu_4.t() @ add_6_b      # backward reads the barriered cotangent
  return [mm_20]


print("\n=== 2. compiled: FX graph and StableHLO ===")
args = (x_1, add_6, mm_16, mm_18, w1s_1)
try:
  gm = make_fx(layer0_entry, tracing_mode="fake")(*args)
  print(gm.code)
  dbg = TpuCompileDebug()
  ex_in = [nd.meta["val"] for nd in gm.graph.nodes if nd.op == "placeholder"]
  exe = StaticCompiler(debug=dbg)(gm, ex_in, is_fwd=True)
  for text in dbg.stablehlo_forward_text:
    body = [ln for ln in text.splitlines()
            if "stablehlo." in ln or "func.func" in ln or "return" in ln]
    print("\n".join(ln.split(" loc(")[0] for ln in body))
  got = exe(*args)[0]
  ref = torch.relu(x_1 @ w1s_1).t() @ add_6
  print(f"\ncompiled result == same math without barrier: "
        f"{torch.allclose(got.cpu(), ref.cpu())}")
except Exception as e:  # report instead of hiding
  print(f"compile path failed: {type(e).__name__}: {e}")
