"""Dump every FX node AFTER the recompute pass, with ALL node metadata.

The toy model is traced the way torchtitan's GraphTrainer traces a real model,
so each node carries the same metadata the TPU passes read:

  meta["custom"]["module_fqn"]               which module made the node
                                             ("layers.0" ... "layers.3", "loss")
                                             -> annotate_module_fqns / compute_annotated_loss
  meta["autograd_backward"]                  True for backward nodes (set by autograd)
  meta["seq_nr"]                             autograd sequence number; links a backward
                                             node to the forward op it differentiates
  meta["custom"]["parameter_gradient_fqns"]  on gradient nodes: which parameter
                                             -> compute_parameter_gradients
  meta["recomputed_from"]                    on nodes inserted by recompute_pass

Region = layer id parsed from module_fqn, exactly like common_utils._get_layer_id.
The last section applies apply_tpu_remat_schedule's boundary rules to each region.
"""

import copy
import os
import traceback as tb

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import annotate, preserve_node_meta

# Reuse recompute_pass/backward_start/sizes from passes_demo.py (top half only).
_src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "passes_demo.py")
_defs = open(_src_path).read().split(
    "# ------------------------------------------------------------------- main ---")[0]
_ns = {"__name__": "passes_demo_defs"}
exec(compile(_defs, _src_path, "exec"), _ns)
recompute_pass, backward_start, nbytes = (
    _ns["recompute_pass"], _ns["backward_start"], _ns["nbytes"])
N_LAYERS, B, D, H, DEV = (_ns[k] for k in ("N_LAYERS", "B", "D", "H", "DEV"))
aten = torch.ops.aten

MODULE_FQN = "module_fqn"                       # common_utils._MODULE_FQN
GRAD_FQNS = "parameter_gradient_fqns"           # common_utils.PARAMETER_GRADIENT_FQNS_META
NOT_IN_LAYERS = -1


# ------------------------------------------------ annotated forward+backward --
def fwd_bwd(W1s, W2s, x, t, n):
  h = x
  for i, (W1, W2) in enumerate(zip(W1s, W2s)):
    with annotate({MODULE_FQN: f"layers.{i}"}):          # like annotate_module_fqns
      h = h + torch.relu(h @ W1) @ W2
  with annotate({MODULE_FQN: "loss"}):                    # like compute_annotated_loss
    loss = ((h - t) ** 2).sum() / n
  params = [*W1s, *W2s]
  fqns = [f"layers.{i}.w1" for i in range(N_LAYERS)] + \
         [f"layers.{i}.w2" for i in range(N_LAYERS)]
  grads = torch.autograd.grad(loss, params)
  tagged = []
  for fqn, g in zip(fqns, grads):                         # like compute_parameter_gradients
    with annotate({GRAD_FQNS: (fqn,)}):
      tagged.append(aten.alias.default(g))
  return [loss, *tagged]


def copy_fwd_metadata_to_bw_nodes(gm):
  """Same logic as make_fx_tracer._copy_fwd_metadata_to_bw_nodes."""
  fwd_by_seq = {}
  for nd in gm.graph.nodes:
    if nd.op == "call_function" and "seq_nr" in nd.meta and \
        not nd.meta.get("autograd_backward", False):
      fwd_by_seq.setdefault(nd.meta["seq_nr"], nd)
  for nd in gm.graph.nodes:
    if nd.op == "call_function" and "seq_nr" in nd.meta and \
        nd.meta.get("autograd_backward", False):
      fwd = fwd_by_seq.get(nd.meta["seq_nr"])
      if fwd is not None and fwd is not nd and fwd.meta.get("custom"):
        nd.meta.setdefault("custom", {}).update(copy.deepcopy(fwd.meta["custom"]))


torch.manual_seed(0)
W1s = [(torch.randn(D, H) * 0.1).to(DEV).requires_grad_(True) for _ in range(N_LAYERS)]
W2s = [(torch.randn(H, D) * 0.1).to(DEV).requires_grad_(True) for _ in range(N_LAYERS)]
x, t = torch.randn(B, D).to(DEV), torch.randn(B, D).to(DEV)
n = torch.tensor(float(B)).to(DEV)

# Same contexts as make_fx_tracer.minimal_fx_tracer (L611-628): backward must run
# on this thread for seq_nr / autograd_backward metadata to be recorded.
_non_strict_ctx = getattr(torch.compiler, "_non_strict_tracing_context", None)
_patch_bwd_ctx = getattr(torch.compiler, "_patch_engine_backward", None)
import contextlib
with contextlib.ExitStack() as _stack:
  _stack.enter_context(preserve_node_meta())             # needed for annotate() to stick
  _stack.enter_context(torch.autograd.set_multithreading_enabled(False))
  if _non_strict_ctx is not None:
    _stack.enter_context(_non_strict_ctx())
  if _patch_bwd_ctx is not None:
    _stack.enter_context(_patch_bwd_ctx())
  gm = make_fx(fwd_bwd, tracing_mode="fake", record_stack_traces=True)(
      W1s, W2s, x, t, n)
print(f"tracing contexts: non_strict={_non_strict_ctx is not None}, "
      f"patch_engine_backward={_patch_bwd_ctx is not None}")
copy_fwd_metadata_to_bw_nodes(gm)
num_static_inputs = 2 * N_LAYERS                          # weights; x, t, n are batch inputs


def remove_gradient_markers(gm):
  """Like remove_parameter_gradient_markers_pass: move each alias marker's
  parameter-gradient tag onto the real gradient node, then delete the alias."""
  for nd in list(gm.graph.nodes):
    if nd.target == aten.alias.default and GRAD_FQNS in custom(nd):
      src = nd.args[0]
      src.meta.setdefault("custom", {})[GRAD_FQNS] = custom(nd)[GRAD_FQNS]
      nd.replace_all_uses_with(src)
      gm.graph.erase_node(nd)
  gm.graph.lint()
  gm.recompile()
  return gm


def custom(nd):
  return nd.meta.get("custom", {}) or {}


gm = remove_gradient_markers(gm)

print("=== running recompute_pass ===")
gm = recompute_pass(gm)
nodes = list(gm.graph.nodes)
idx = {nd: i for i, nd in enumerate(nodes)}
bstart = backward_start(nodes)


def _layer_of_fqn(fqn):
  parts = fqn.split(".")
  return int(parts[1]) if parts[0] == "layers" and len(parts) >= 2 and parts[1].isdigit() else None


# Region source for each node, shown in the dump.
REGION_SRC = {}
for nd in nodes:
  if MODULE_FQN in custom(nd):
    REGION_SRC[nd] = ("annotate(module_fqn) during forward" if "recomputed_from" not in nd.meta
                      else "copied with the forward op by recompute_pass")

# Fallback when torch.compiler._patch_engine_backward is missing (this wheel):
# autograd does not tag backward nodes, and they all inherit the last forward
# op's seq_nr, so seq_nr cannot map them to a layer. Instead, from each node's
# direct inputs:
#   local signals (preferred):
#     weight of layer k                       -> k
#     recomputed node of layer k              -> k
#     forward output of layer k (e.g. `add`)  -> k+1  (it is layer k+1's input)
#     batch input x                           -> 0    (layer 0's input)
#   otherwise (only backward inputs): lowest layer among them
#     (the incoming cotangent comes from a higher layer, so take the lowest)
#   no layer at all -> loss region (-1)
FALLBACK_USED = _patch_bwd_ctx is None
if FALLBACK_USED:
  weight_layer = {nodes[i]: i % N_LAYERS for i in range(num_static_inputs)}
  x_input = nodes[num_static_inputs]                     # x_1
  inferred = {}
  for i, nd in enumerate(nodes):
    if i < bstart or nd.op != "call_function" or "recomputed_from" in nd.meta:
      continue
    nd.meta["autograd_backward"] = True
    local, from_bwd = [], []
    for a in nd.all_input_nodes:
      lid = _layer_of_fqn(custom(a).get(MODULE_FQN, ""))
      if a in weight_layer:
        local.append(weight_layer[a])
      elif a is x_input:
        local.append(0)
      elif "recomputed_from" in a.meta and lid is not None:
        local.append(lid)
      elif a in inferred:
        if inferred[a] is not None:
          from_bwd.append(inferred[a])
      elif lid is not None:                              # forward output of layer lid
        local.append(lid + 1)
    layer = min(local) if local else (min(from_bwd) if from_bwd else None)
    inferred[nd] = layer
    if layer is not None:
      nd.meta.setdefault("custom", {})[MODULE_FQN] = f"layers.{layer}"
      REGION_SRC[nd] = ("inferred from inputs (fallback: "
                        + ("own weight / recompute / layer input" if local
                           else "lowest backward input") + ")")
    else:
      nd.meta.setdefault("custom", {})[MODULE_FQN] = "loss"
      REGION_SRC[nd] = "inferred from inputs (fallback: no layer input -> loss)"
  print("NOTE: torch.compiler._patch_engine_backward not available in this wheel; "
        "backward regions inferred from inputs (see 'region src').")
  bad = [(nd.name, custom(nd)[GRAD_FQNS], custom(nd)[MODULE_FQN])
         for nd in nodes if GRAD_FQNS in custom(nd)
         and _layer_of_fqn(custom(nd)[GRAD_FQNS][0]) !=
             _layer_of_fqn(custom(nd).get(MODULE_FQN, ""))]
  print("check: every gradient node's region matches its parameter tag: "
        + ("OK" if not bad else f"MISMATCH {bad}"))


# ---------------------------------------------------------------- helpers ----
def custom(nd):
  return nd.meta.get("custom", {}) or {}


def region(nd):                                         # common_utils._get_layer_id
  parts = custom(nd).get(MODULE_FQN, "").split(".")
  if parts[0] == "layers" and len(parts) >= 2 and parts[1].isdigit():
    return int(parts[1])
  return NOT_IN_LAYERS


def phase(nd):
  if nd.op in ("placeholder", "output"):
    return nd.op.upper()
  if "recomputed_from" in nd.meta:
    return "RECOMPUTE (forward op copied into backward)"
  if nd.meta.get("autograd_backward", False):
    return "BACKWARD"
  return "FORWARD"


def is_recomputed(nd):
  return "recomputed_from" in nd.meta


def is_backward(nd):
  return nd.meta.get("autograd_backward", False) or is_recomputed(nd)


def ref(nd):
  return f"#{idx[nd]} {nd.name}" if isinstance(nd, torch.fx.Node) else repr(nd)


def fmt_args(nd):
  return "(" + ", ".join(ref(a) if isinstance(a, torch.fx.Node) else repr(a)
                         for a in nd.args) + ")"


def source_line(nd):
  st = nd.meta.get("stack_trace")
  if not st:
    return "-"
  lines = [ln.strip() for ln in st.strip().splitlines() if ln.strip()]
  for i in range(len(lines) - 1, -1, -1):              # last frame in this script
    if "dump_nodes_after_pass.py" in lines[i]:
      loc = lines[i].split(",")[1].strip() if "," in lines[i] else lines[i]
      code = lines[i + 1] if i + 1 < len(lines) else ""
      return f"{loc}: {code}"
  return lines[-1]


def describe_val(nd):
  v = nd.meta.get("val")
  if isinstance(v, torch.Tensor):
    return f"shape={tuple(v.shape)} dtype={str(v.dtype).replace('torch.', '')} " \
           f"device={v.device} bytes={nbytes(nd)}"
  return "-"


# ------------------------------------------------------- full node records ---
print(f"\n=== ALL NODES AFTER recompute pass: {len(nodes)} nodes "
      f"(backward starts at #{bstart}) ===")
print("legend: region = layer id from meta.custom.module_fqn (-1 = not in a layer)\n")
for nd in nodes:
  c = custom(nd)
  print(f"#{idx[nd]:<3} {nd.name}")
  print(f"      op        : {nd.op}   target: {str(nd.target).replace('torch.ops.', '')}")
  if nd.op != "placeholder":
    print(f"      args      : {fmt_args(nd) if nd.op != 'output' else ', '.join(ref(a) for a in nd.args[0])}")
  print(f"      value     : {describe_val(nd)}")
  print(f"      phase     : {phase(nd)}")
  print(f"      region    : {region(nd)}"
        + (f"   (module_fqn='{c[MODULE_FQN]}')" if MODULE_FQN in c else ""))
  if nd in REGION_SRC:
    print(f"      region src: {REGION_SRC[nd]}")
  if "seq_nr" in nd.meta:
    print(f"      seq_nr    : {nd.meta['seq_nr']}")
  if "autograd_backward" in nd.meta:
    print(f"      autograd_backward: {nd.meta['autograd_backward']}")
  if GRAD_FQNS in c:
    print(f"      grad of   : {c[GRAD_FQNS]}   (parameter-gradient tag)")
  if is_recomputed(nd):
    print(f"      recomputed: copy of forward '{nd.meta['recomputed_from']}' "
          "(inserted by recompute_pass)")
  if nd.op == "placeholder":
    kind = "static (model state)" if idx[nd] < num_static_inputs else "non-static (batch input)"
    print(f"      input kind: {kind}")
  print(f"      source    : {source_line(nd)}")
  users = [ref(u) for u in nd.users]
  print(f"      users     : {', '.join(users) if users else '-'}")
  other = sorted(k for k in nd.meta if k not in (
      "val", "tensor_meta", "stack_trace", "seq_nr", "autograd_backward",
      "custom", "recomputed_from"))
  if other:
    print(f"      other meta: {other}")
  print()


# ----------------------------------------------- region summary + remat view --
print("=== REGION SUMMARY (what apply_tpu_remat_schedule groups by) ===")
regions = sorted({region(nd) for nd in nodes if nd.op == "call_function"})
for r in regions:
  members = [nd for nd in nodes if nd.op == "call_function" and region(nd) == r]
  fwd = [ref(nd) for nd in members if phase(nd) == "FORWARD"]
  rec = [ref(nd) for nd in members if is_recomputed(nd)]
  bwd = [ref(nd) for nd in members if phase(nd) == "BACKWARD"]
  name = f"layer {r}" if r != NOT_IN_LAYERS else "not in a layer (loss / grad tags)"
  print(f"\nregion {r}  [{name}]")
  print(f"   forward   ({len(fwd)}): {', '.join(fwd) or '-'}")
  print(f"   recompute ({len(rec)}): {', '.join(rec) or '-'}")
  print(f"   backward  ({len(bwd)}): {', '.join(bwd) or '-'}")

print("\n=== apply_tpu_remat_schedule boundary analysis (same rules as "
      "remat_schedule.py L284-336) ===")
static = set(nodes[:num_static_inputs])
memo = {}


def depends_on_batch(nd):
  if nd in memo:
    return memo[nd]
  if nd.op == "placeholder":
    res = nd not in static
  else:
    res = any(depends_on_batch(a) for a in nd.all_input_nodes)
  memo[nd] = res
  return res


grads_by_region = {}
for nd in nodes:
  fqns = custom(nd).get(GRAD_FQNS)
  if fqns:
    layer = int(fqns[0].split(".")[1])
    grads_by_region.setdefault(layer, []).append(nd)

layer_regions = [r for r in regions if r != NOT_IN_LAYERS]
for k, r in enumerate(layer_regions):
  rec = {nd for nd in nodes if region(nd) == r and is_recomputed(nd)}
  bwd = {nd for nd in nodes if region(nd) == r and phase(nd) == "BACKWARD"}
  primals = sorted({a for nd in rec for a in nd.all_input_nodes
                    if a not in rec and isinstance(a.meta.get("val"), torch.Tensor)
                    and depends_on_batch(a)}, key=idx.get)
  cands = sorted({a for nd in bwd for a in nd.all_input_nodes
                  if a not in bwd and not is_recomputed(a) and is_backward(a)
                  and isinstance(a.meta.get("val"), torch.Tensor)}, key=idx.get)
  nxt = layer_regions[k + 1] if k + 1 < len(layer_regions) else None
  ordering = grads_by_region.get(nxt, []) if nxt is not None else []
  print(f"\nregion {r} (layer {r}):")
  print(f"   primal boundaries  : {[ref(p) for p in primals]}   "
        "<- saved forward values the recompute starts from")
  print(f"   cotangent candidates: {[ref(c) for c in cands]}   "
        "<- backward values entering from outside; earliest is the cotangent")
  print(f"   ordering gradients : {[ref(g) + ' ' + str(custom(g)[GRAD_FQNS]) for g in ordering]}"
        f"   <- next region ({nxt}) dW tags")
  if nxt is None:
    print("   (last region: no next layer, so no barrier is inserted)")
  else:
    print(f"   => barrier would be optimization_barrier("
          f"{', '.join([p.name for p in primals] + [cands[0].name if cands else '?'] + [g.name for g in ordering])})")
