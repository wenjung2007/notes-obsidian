"""Same toy model, two capture paths, node lists side by side.

  Path A  graph_trainer style:  minimal_fx_tracer(fwd + loss + autograd.grad)
          -> ONE joint ATen graph                      (compile.mode="aot_fx_trace")
  Path B  torch.compile(model, backend=<Dynamo graph -> aot_autograd(fw, bw)>)
          -> (1) Dynamo graph: torch-level, forward only
             (2) AOTAutograd forward graph  (ATen)
             (3) AOTAutograd backward graph (ATen)       (compile.mode="jit" style)

Model = passes_demo.py's residual MLP, as an nn.Module so both paths see params:
    h_next = h + relu(h @ W1_i) @ W2_i ;  loss = sum((h_L - t)^2) / n
Both models get module_fqn annotations exactly like
common_utils.annotate_module_fqns, so we can see whether each path keeps them.

Outputs (next to this file):
  dynamo_vs_makefx_output.log          full report (stdout)
  dynamo_vs_makefx_side_by_side.txt    make_fx joint graph | AOT fwd + bwd graphs
"""

import collections
import contextlib
import copy
import os
import sys
import traceback

import torch
import torch.nn as nn

try:
  import torch_tpu  # noqa: F401
except ImportError:
  pass
from torch.fx.traceback import annotate_fn

sys.path.insert(0, os.environ.get("TTMIN", "/tmp/ttmin"))   # minimal torchtitan copy

N_LAYERS = int(os.environ.get("N_LAYERS", "2"))
B, D, H = 8, 16, 64                # tokens, model dim, hidden dim
HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_FQN = "module_fqn"


# ------------------------------------------------------------------ model ----
class Layer(nn.Module):
  def __init__(self):
    super().__init__()
    self.w1 = nn.Parameter(torch.randn(D, H) * 0.1)
    self.w2 = nn.Parameter(torch.randn(H, D) * 0.1)

  def forward(self, h):
    return h + torch.relu(h @ self.w1) @ self.w2


class Toy(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.ModuleList(Layer() for _ in range(N_LAYERS))

  def forward(self, x):
    h = x
    for layer in self.layers:
      h = layer(h)
    return h


def loss_fn(h, t, n):
  return ((h - t) ** 2).sum() / n


def annotate_module_fqns(model):
  """Same as torchtitan common_utils.annotate_module_fqns."""
  for fqn, sub in model.named_modules():
    if fqn:
      sub.forward = annotate_fn({MODULE_FQN: fqn})(sub.forward)


# --------------------------------------------------------------- printing ----
def short_target(nd):
  if nd.op in ("placeholder", "output", "get_attr"):
    return nd.op
  t = nd.target
  if isinstance(t, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
    return str(t)                                   # e.g. aten.mm.default
  if isinstance(t, str):
    return f"method .{t}()"                         # call_method
  mod = getattr(t, "__module__", "") or ""
  name = getattr(t, "__name__", str(t))
  mod = {"_operator": "operator", "torch._C._VariableFunctions": "torch"}.get(mod, mod)
  return f"{mod}.{name}" if mod else name


def fmt(a):
  if isinstance(a, torch.fx.Node):
    return a.name
  if isinstance(a, (list, tuple)):
    return "[" + ", ".join(fmt(x) for x in a) + "]"
  return repr(a)


def val_of(nd):
  v = nd.meta.get("val", nd.meta.get("example_value"))
  if isinstance(v, torch.Tensor):
    return f"{tuple(v.shape)}"
  if isinstance(v, (list, tuple)) and v and all(isinstance(x, torch.Tensor) for x in v):
    return "(" + ", ".join(str(tuple(x.shape)) for x in v) + ")"
  return ""


def meta_flags(nd):
  m = nd.meta
  out = []
  if "seq_nr" in m:
    out.append(f"seq={m['seq_nr']}")
  if m.get("autograd_backward"):
    out.append("bwd")
  cust = m.get("custom") or {}
  if MODULE_FQN in cust:
    out.append(f"fqn={cust[MODULE_FQN]}")
  stk = m.get("nn_module_stack")
  if stk:
    out.append("nn_mod=" + list(stk.values())[-1][0])
  return " ".join(out)


def node_line(i, nd, width=None):
  if nd.op == "placeholder":
    s = f"#{i:<3}{nd.name} = placeholder {val_of(nd)}"
  elif nd.op == "output":
    s = f"#{i:<3}output({fmt(nd.args[0])})"
  else:
    s = f"#{i:<3}{nd.name} = {short_target(nd)}({', '.join(fmt(a) for a in nd.args)}) {val_of(nd)}"
  if width:
    s = s if len(s) <= width else s[: width - 1] + "~"
  return s


def dump_graph(title, gm):
  nodes = list(gm.graph.nodes)
  print(f"\n### {title}: {len(nodes)} nodes")
  for i, nd in enumerate(nodes):
    flags = meta_flags(nd)
    print(f"  {node_line(i, nd)}" + (f"    [{flags}]" if flags else ""))


def op_counter(gm):
  return collections.Counter(short_target(n) for n in gm.graph.nodes
                             if n.op in ("call_function", "call_method"))


# ------------------------------------------------------------------- paths ---
def path_a_make_fx(model, x, t, n):
  """graph_trainer: trace fwd + loss + autograd.grad as ONE graph."""
  def fwd_bwd(x, t, n):
    loss = loss_fn(model(x), t, n)
    params = [p for _, p in model.named_parameters()]
    grads = torch.autograd.grad(loss, params)
    return [loss, *grads]

  try:
    from torchtitan.experiments.graph_trainer.make_fx_tracer import (
        minimal_fx_tracer, run_traced)
    traced = minimal_fx_tracer(fwd_bwd, module=model)(x, t, n)
    outs = run_traced(traced, module=model)(x, t, n)
    return traced.gm, [o.detach().cpu() for o in outs], "torchtitan minimal_fx_tracer"
  except Exception:                                          # pragma: no cover
    print("minimal_fx_tracer failed, falling back to plain make_fx:")
    traceback.print_exc(limit=3)
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.fx.traceback import preserve_node_meta
    with preserve_node_meta(), torch.autograd.set_multithreading_enabled(False):
      gm = make_fx(fwd_bwd, tracing_mode="fake")(x, t, n)
    outs = fwd_bwd(x, t, n)
    return gm, [o.detach().cpu() for o in outs], "plain make_fx (fallback)"


def path_b_dynamo_aot(model, x, t, n):
  """torch.compile: Dynamo graph -> aot_autograd -> separate fwd / bwd graphs."""
  from functorch.compile import make_boxed_func
  from torch._dynamo.backends.common import aot_autograd
  cap = {}

  def fw_compiler(gm, example_inputs):
    cap["aot_fw"] = gm
    return make_boxed_func(gm.forward)

  def bw_compiler(gm, example_inputs):
    cap["aot_bw"] = gm
    return make_boxed_func(gm.forward)

  aot_backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)

  def my_backend(gm, example_inputs):        # Dynamo hands us its torch-level graph
    cap["dynamo"] = gm
    return aot_backend(gm, example_inputs)

  torch._dynamo.reset()
  compiled = torch.compile(model, backend=my_backend, fullgraph=True)
  h = compiled(x)                            # forward: Dynamo + AOT fwd graph
  loss = loss_fn(h, t, n)                    # loss runs EAGERLY (not compiled)
  loss.backward()                            # triggers the AOT backward graph
  grads = [p.grad.detach().cpu() for _, p in model.named_parameters()]
  return cap, [loss.detach().cpu(), *grads]


def run_on(dev):
  torch.manual_seed(0)
  base = Toy()
  model_a, model_b = copy.deepcopy(base).to(dev), copy.deepcopy(base).to(dev)
  annotate_module_fqns(model_a)
  annotate_module_fqns(model_b)
  x, t = torch.randn(B, D).to(dev), torch.randn(B, D).to(dev)
  n = torch.tensor(float(B)).to(dev)
  gm_a, outs_a, how_a = path_a_make_fx(model_a, x, t, n)
  cap_b, outs_b = path_b_dynamo_aot(model_b, x, t, n)
  return gm_a, outs_a, how_a, cap_b, outs_b


# -------------------------------------------------------------------- main ---
dev_used = None
for dev in (os.environ.get("DEV", "tpu"), "cpu"):
  try:
    gm_a, outs_a, how_a, cap_b, outs_b = run_on(dev)
    dev_used = dev
    break
  except Exception:
    print(f"!!! run on device '{dev}' failed:")
    traceback.print_exc(limit=4)
assert dev_used is not None, "both devices failed"

gm_dyn, gm_fw, gm_bw = cap_b["dynamo"], cap_b["aot_fw"], cap_b["aot_bw"]
param_names = [n for n, _ in Toy().named_parameters()]

print("=" * 78)
print(f"torch {torch.__version__} | device={dev_used} | N_LAYERS={N_LAYERS} "
      f"| B={B} D={D} H={H} | path A via {how_a}")
print("=" * 78)

print("\n" + "=" * 78 + "\n1. RAW GRAPHS\n" + "=" * 78)
dump_graph("PATH B-1  Dynamo graph (torch-level, forward only)", gm_dyn)
dump_graph("PATH B-2  AOTAutograd FORWARD graph (ATen)", gm_fw)
dump_graph("PATH B-3  AOTAutograd BACKWARD graph (ATen)", gm_bw)
dump_graph("PATH A    make_fx JOINT graph (ATen: fwd + loss + bwd)", gm_a)

# ---------------------------------------------------------------- analysis --
print("\n" + "=" * 78 + "\n2. GRAPH COUNT / SIZE\n" + "=" * 78)
def size(gm):
  ns = list(gm.graph.nodes)
  return (len(ns), sum(n.op == "placeholder" for n in ns),
          sum(n.op in ("call_function", "call_method") for n in ns))
print(f"{'graph':<34}{'nodes':>7}{'inputs':>8}{'ops':>6}")
for name, gm in (("B-1 Dynamo (torch-level)", gm_dyn), ("B-2 AOT forward", gm_fw),
                 ("B-3 AOT backward", gm_bw), ("A   make_fx joint", gm_a)):
  print(f"{name:<34}" + "".join(f"{v:>7}" if i == 0 else f"{v:>8}" if i == 1 else f"{v:>6}"
                                 for i, v in enumerate(size(gm))))

print("\n" + "=" * 78 + "\n3. OP COUNTS (call_function / call_method nodes)\n" + "=" * 78)
c_dyn, c_fw, c_bw, c_a = map(op_counter, (gm_dyn, gm_fw, gm_bw, gm_a))
c_aot = c_fw + c_bw
rows = sorted(set(c_dyn) | set(c_aot) | set(c_a))
print(f"{'op':<34}{'Dynamo':>7}{'AOT fw':>7}{'AOT bw':>7}{'AOT sum':>8}{'make_fx':>8}  note")
for op in rows:
  note = ""
  if c_aot[op] != c_a[op]:
    note = "<- differs (AOT vs make_fx)"
  print(f"{op:<34}{c_dyn[op]:>7}{c_fw[op]:>7}{c_bw[op]:>7}{c_aot[op]:>8}{c_a[op]:>8}  {note}")

print("\n" + "=" * 78 + "\n4. GRAPH INPUTS (placeholders)\n" + "=" * 78)
for name, gm in (("B-1 Dynamo", gm_dyn), ("B-2 AOT forward", gm_fw),
                 ("B-3 AOT backward", gm_bw), ("A   make_fx joint", gm_a)):
  phs = [f"{n.name}{val_of(n)}" for n in gm.graph.nodes if n.op == "placeholder"]
  print(f"{name:<20}: {', '.join(phs)}")

print("\n" + "=" * 78 + "\n5. GRAPH OUTPUTS and the fwd->bwd boundary\n" + "=" * 78)
def outputs(gm):
  out = next(n for n in gm.graph.nodes if n.op == "output")
  a = out.args[0]
  return list(a) if isinstance(a, (list, tuple)) else [a]
fw_out = outputs(gm_fw)
print(f"AOT forward returns {len(fw_out)} values:")
print(f"   [0] {fmt(fw_out[0])}{val_of(fw_out[0]) if isinstance(fw_out[0], torch.fx.Node) else ''}"
      "   <- the model output h (what the user gets)")
for k, o in enumerate(fw_out[1:], 1):
  if isinstance(o, torch.fx.Node):
    print(f"   [{k}] {o.name:<12}{val_of(o):<10} <- SAVED for backward ({short_target(o)})")
bw_in = [n for n in gm_bw.graph.nodes if n.op == "placeholder"]
print(f"AOT backward takes {len(bw_in)} inputs = {len(fw_out) - 1} saved tensors "
      f"+ {len(bw_in) - (len(fw_out) - 1)} tangent(s) (dLoss/dh, computed EAGERLY outside)")
bw_out = outputs(gm_bw)
print(f"AOT backward returns {len(bw_out)} values: "
      + ", ".join(fmt(o) for o in bw_out) + "  (grads of graph inputs; None for x)")
a_out = outputs(gm_a)
print(f"make_fx joint returns {len(a_out)} values: loss + {len(a_out) - 1} grads "
      f"({', '.join(param_names)})  -- no saved-tensor boundary: backward nodes read "
      "forward nodes directly")

print("\n" + "=" * 78 + "\n6. WHAT EACH GRAPH COVERS\n" + "=" * 78)
loss_ops = {"aten.sub.Tensor", "aten.pow.Tensor_Scalar", "aten.sum.default", "aten.div.Tensor"}
for name, c in (("Dynamo", c_dyn), ("AOT fw+bw", c_aot), ("make_fx", c_a)):
  found = sorted(op for op in loss_ops if c[op])
  print(f"{name:<10} loss ops present: {found if found else 'NONE (loss ran eagerly)'}")
print("ones_like (seed dLoss/dLoss=1, start of backward) in make_fx: "
      f"{c_a['aten.ones_like.default']}, in AOT: {c_aot['aten.ones_like.default']}")

print("\n" + "=" * 78 + "\n7. NODE METADATA (what graph passes can read)\n" + "=" * 78)
def meta_keys(gm):
  keys = collections.Counter()
  for n in gm.graph.nodes:
    if n.op == "call_function":
      keys.update(n.meta.keys())
  return keys
for name, gm in (("B-1 Dynamo", gm_dyn), ("B-2 AOT fw", gm_fw), ("B-3 AOT bw", gm_bw),
                 ("A   make_fx", gm_a)):
  k = meta_keys(gm)
  print(f"{name:<12}: {', '.join(f'{key}({cnt})' for key, cnt in sorted(k.items()))}")
def fqn_coverage(gm):
  ops = [n for n in gm.graph.nodes if n.op in ("call_function", "call_method")]
  tagged = [n for n in ops if MODULE_FQN in (n.meta.get("custom") or {})]
  return f"{len(tagged)}/{len(ops)} op nodes carry custom['module_fqn']"
for name, gm in (("B-1 Dynamo", gm_dyn), ("B-2 AOT fw", gm_fw), ("B-3 AOT bw", gm_bw),
                 ("A   make_fx", gm_a)):
  print(f"{name:<12}: {fqn_coverage(gm)}")

print("\n" + "=" * 78 + "\n8. NUMERICS: same math?\n" + "=" * 78)
print(f"loss  make_fx={outs_a[0].item():.6f}  dynamo+aot={outs_b[0].item():.6f}")
diff = max((a - b).abs().max().item() for a, b in zip(outs_a, outs_b))
print(f"max |diff| over loss + {len(outs_a) - 1} grads: {diff:.3e}")

# ------------------------------------------------------------ side by side --
W = 78
left = ["PATH A: make_fx JOINT graph (fwd + loss + bwd)", "-" * W]
left += [node_line(i, nd, W) for i, nd in enumerate(gm_a.graph.nodes)]
right = ["PATH B: AOTAutograd FORWARD graph", "-" * W]
right += [node_line(i, nd, W) for i, nd in enumerate(gm_fw.graph.nodes)]
right += ["", "      (loss + loss.backward() seed run EAGERLY here)", "",
          "PATH B: AOTAutograd BACKWARD graph", "-" * W]
right += [node_line(i, nd, W) for i, nd in enumerate(gm_bw.graph.nodes)]
right_dyn = ["PATH B-1: Dynamo graph (torch-level, before AOT)", "-" * W]
right_dyn += [node_line(i, nd, W) for i, nd in enumerate(gm_dyn.graph.nodes)]
rows_n = max(len(left), len(right))
with open(os.path.join(HERE, "dynamo_vs_makefx_side_by_side.txt"), "w") as f:
  f.write(f"device={dev_used} N_LAYERS={N_LAYERS}\n\n")
  for i in range(rows_n):
    l = left[i] if i < len(left) else ""
    r = right[i] if i < len(right) else ""
    f.write(f"{l:<{W}} | {r}\n")
  f.write("\n\n")
  f.write("\n".join(right_dyn) + "\n")
print("\nwrote dynamo_vs_makefx_side_by_side.txt")
