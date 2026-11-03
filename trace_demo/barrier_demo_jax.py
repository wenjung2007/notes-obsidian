"""optimization_barrier(x_1, add_6, mm_16, mm_18), shown with JAX on the same TPU.

The VM's torch_tpu wheel (0.1.1.dev20260912) predates tpu::_optimization_barrier,
so this uses jax.lax.optimization_barrier, which lowers to the SAME op that
torch.tpu.ops.optimization_barrier lowers to: `stablehlo.optimization_barrier`.

Shows:
  1. values pass through bit-identically,
  2. the StableHLO contains one optimization_barrier taking all 4 inputs,
  3. the compiled result equals the same math without a barrier.
"""

import jax
import jax.numpy as jnp
import numpy as np

print("jax", jax.__version__, "devices:", jax.devices())
k = jax.random.split(jax.random.PRNGKey(0), 5)
x_1 = jax.random.normal(k[0], (8, 16))     # layer 0 input        (primal boundary)
add_6 = jax.random.normal(k[1], (8, 16))   # dL/d(layer 0 output) (cotangent)
mm_16 = jax.random.normal(k[2], (64, 16))  # layer 1 dW2          (ordering only)
mm_18 = jax.random.normal(k[3], (16, 64))  # layer 1 dW1          (ordering only)
w1s_1 = jax.random.normal(k[4], (16, 64))  # layer 0 W1

print("\n=== 1. values pass through unchanged ===")
outs = jax.lax.optimization_barrier((x_1, add_6, mm_16, mm_18))
for name, a, b in zip(["x_1", "add_6", "mm_16", "mm_18"],
                      [x_1, add_6, mm_16, mm_18], outs):
  print(f"{name:<6} in {a.shape} -> out {b.shape}  "
        f"identical={bool(np.array_equal(np.asarray(a), np.asarray(b)))}")


def layer0_entry(x_1, add_6, mm_16, mm_18, w1s_1):
  # Same wiring as apply_tpu_remat_schedule's rewrite of layer 0's entry:
  x_b, add_6_b, _, _ = jax.lax.optimization_barrier((x_1, add_6, mm_16, mm_18))
  mm_23 = x_b @ w1s_1               # recompute reads barriered x_1
  relu_4 = jax.nn.relu(mm_23)
  mm_20 = relu_4.T @ add_6_b         # backward reads barriered cotangent
  return mm_20


def layer0_entry_no_barrier(x_1, add_6, mm_16, mm_18, w1s_1):
  return jax.nn.relu(x_1 @ w1s_1).T @ add_6


args = (x_1, add_6, mm_16, mm_18, w1s_1)
print("\n=== 2. StableHLO of the barriered layer-0 entry ===")
lowered = jax.jit(layer0_entry).lower(*args)
for ln in lowered.as_text().splitlines():
  if "stablehlo." in ln or "func.func public" in ln or "return" in ln:
    print("  " + ln.strip().split(" loc(")[0])

print("\n=== 3. compiled results ===")
got = jax.jit(layer0_entry)(*args)
ref = jax.jit(layer0_entry_no_barrier)(*args)
print(f"barrier vs no-barrier identical: {bool(np.array_equal(np.asarray(got), np.asarray(ref)))}")
print(f"mm_20 shape {got.shape}, first row[:4] = {np.asarray(got)[0, :4].round(4).tolist()}")

print("\n=== 4. is the barrier still there after XLA optimization? ===")
hlo = jax.jit(layer0_entry).lower(*args).compile().as_text()
print("optimized HLO contains 'opt-barrier':", "opt-barrier" in hlo)
for ln in hlo.splitlines():
  if "opt-barrier" in ln:
    print("  " + ln.strip()[:160])
