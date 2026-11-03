"""Show the SAME toy step in its two representations: FX graph and MLIR.

FX   = what make_fx records (PyTorch-level graph of aten ops).
MLIR = what torch_tpu's StaticCompiler lowers it to (StableHLO), right before
       XLA compiles it into TPU machine code.
"""

import torch
import torch_tpu  # noqa: F401
from torch.fx.experimental.proxy_tensor import make_fx
from torch_tpu._internal.compile.compiler import StaticCompiler
from torch_tpu._internal.compile.debug import TpuCompileDebug


def fwd_bwd_step(W, x, t, n):
  y = x @ W
  loss = ((y - t) ** 2).sum() / n
  (grad_W,) = torch.autograd.grad(loss, [W])
  return [loss, grad_W]


W = torch.tensor([[0.5], [-1.0], [2.0], [0.0]]).to("tpu").requires_grad_(True)
x = torch.tensor([[1.0, 2.0, 0.0, 1.0]]).to("tpu")
t = torch.tensor([[1.0]]).to("tpu")
n = torch.tensor(2.0).to("tpu")

gm = make_fx(fwd_bwd_step, tracing_mode="fake")(W, x, t, n)
print("=" * 70)
print("FX GRAPH (from make_fx)")
print("=" * 70)
print(gm.code)

debug = TpuCompileDebug()
example_inputs = [nd.meta["val"] for nd in gm.graph.nodes if nd.op == "placeholder"]
exe = StaticCompiler(debug=debug)(gm, example_inputs, is_fwd=True)
print("=" * 70)
print("MLIR / StableHLO (from torch_tpu StaticCompiler)")
print("=" * 70)
for text in debug.stablehlo_forward_text:
  print(text)

loss, grad = exe(W, x, t, n)
print("run on TPU ->", loss.item(), grad.flatten().tolist())
