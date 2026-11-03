"""Toy demo: what a "traced" forward+loss+backward step looks like.

Mirrors torchtitan's graph trainer:
  - make_fwd_bwd_step  -> fwd_bwd_step below (forward, loss, autograd.grad)
  - minimal_fx_tracer  -> make_fx(...) on FakeTensors
  - run_traced         -> calling the traced GraphModule on real tensors

Uses the same numbers as the walkthrough's 2-chip example (chip 0's view):
  W = [0.5, -1, 2, 0], x = [1, 2, 0, 1], t = 1, n = 2
"""

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._subclasses.fake_tensor import FakeTensorMode

torch.manual_seed(0)


# --- 1. The "model" (one weight) and the function to be traced -------------
class Toy(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.W = torch.nn.Parameter(torch.tensor([[0.5], [-1.0], [2.0], [0.0]]))

  def forward(self, x):
    return x @ self.W


model = Toy()
python_calls = 0


def fwd_bwd_step(W, x, t, n):
  """Like graph_trainer.make_fwd_bwd_step: forward + loss + explicit backward.

  W is passed explicitly, like minimal_fx_tracer threads parameters in as
  graph inputs instead of baking them into the graph.
  """
  global python_calls
  python_calls += 1
  print(f"    [python body running, call #{python_calls}; "
        f"x is {type(x).__name__}]")
  y = x @ W                                  # forward
  loss = ((y - t) ** 2).sum() / n            # loss
  (grad_W,) = torch.autograd.grad(loss, [W])  # backward (recorded too!)
  return [loss, grad_W]


x0 = torch.tensor([[1.0, 2.0, 0.0, 1.0]])
t0 = torch.tensor([[1.0]])
n = torch.tensor(2.0)
W = model.W.detach().clone().requires_grad_(True)

# --- 2. TRACE: run once on FakeTensors (shape only, no data) ----------------
print("=" * 70)
print("STEP 1: TRACE with make_fx on FakeTensors")
print("=" * 70)
# tracing_mode="fake" swaps every input for a FakeTensor (shape, no data),
# like minimal_fx_tracer's _fakeify_input.
gm = make_fx(fwd_bwd_step, tracing_mode="fake")(W, x0, t0, n)

print("\n--- traced graph as Python code (gm.code) ---")
print(gm.code)

print("--- traced graph node table (gm.graph) ---")
print(f"{'#':>3}  {'op':<14} {'name':<16} {'target':<34} shape")
for i, node in enumerate(gm.graph.nodes):
  val = node.meta.get("val")
  shape = tuple(val.shape) if isinstance(val, torch.Tensor) else ""
  target = str(node.target).replace("aten.", "")
  print(f"{i:>3}  {node.op:<14} {node.name:<16} {target:<34} {shape}")

# --- 3. RUN: execute the recorded graph on real data (no Python body) -------
print("\n" + "=" * 70)
print("STEP 2+: RUN the traced graph on real tensors")
print("=" * 70)
calls_before = python_calls
loss, grad = gm(W, x0, t0, n)
print(f"chip0 token: loss = {loss.item():.4f}, dL/dW = {grad.flatten().tolist()}")

x1 = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
t1 = torch.tensor([[0.0]])
loss, grad = gm(W, x1, t1, n)
print(f"chip1 token: loss = {loss.item():.4f}, dL/dW = {grad.flatten().tolist()}")
print(f"python body calls during RUN: {python_calls - calls_before}  "
      f"(0 = the graph replays without re-running Python)")

# --- 4. Cross-check against normal eager autograd ---------------------------
print("\n" + "=" * 70)
print("CHECK: eager loss.backward() gives the same gradient")
print("=" * 70)
model.zero_grad()
eager_loss = ((model(x0) - t0) ** 2).sum() / n
eager_loss.backward()
print(f"eager: loss = {eager_loss.item():.4f}, "
      f"W.grad = {model.W.grad.flatten().tolist()}")
