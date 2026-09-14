---
title: "Summary of Mini RL Fixes and TPU Raiden Wheel Resolution"
date: "2026-09-14"
tags:
  - tpu-sync
  - raiden
  - mini-rl
  - verl
  - gke
  - pytorch-xla
  - vllm
  - troubleshooting
---

# 🚀 Summary of Mini RL Fixes & TPU Raiden Wheel Resolution

This document summarizes the end-to-end technical root causes, debugging steps, and resolutions for running Mini RL (`mini_rl_reproduce.py` / `reproduce_standalone.py`) and TPU weight synchronization across Torchtitan and vLLM on GKE TPU v6e Ray clusters.

---

## 1. Context & Architectural Overview

The RL pipeline on TPU requires weight synchronization between:
- **Actor / Training**: Torchtitan (PyTorch / PyTorch-XLA).
- **Rollout / Inference**: vLLM on TPU.

Weight transfers use **TPU Raiden / TPU Sync** (`tpu_raiden_torch` native extension `_tpu_raiden_torch.so`). During integration, Mini RL job submissions hung or failed across multiple layers: Bazel compilation flags, dynamic linker errors (`DT_NEEDED`), GCS lock contention, Megatron/XLA device ordinal conflicts, and Python site-packages symlink mismatches.

---

## 2. Issues Encountered & Concrete Fixes

### Issue 1: Missing C++ Symbol `TensorToXlaTensor` in `_tpu_raiden_torch.so`

* **Symptom**:
  ```text
  ImportError: /home/ray/anaconda3/lib/python3.12/site-packages/tpu_sync/frameworks/torch/_tpu_raiden_torch.so:
  undefined symbol: _ZN5torch3xla20tensor_to_xla_tensorERKN2at6TensorE
  (torch::xla::tensor_to_xla_tensor(at::Tensor const&))
  ```
* **Root Cause**:
  In PyTorch/XLA versions built with newer C++ namespaces, `tensor_to_xla_tensor` was moved or inlined differently, or resided in `_XLAC.so` / `torch_xla` shims rather than directly linked targets.
* **Fix**:
  1. Updated `tpu_sync/frameworks/torch/BUILD` to link against the exact XLA bridge and C++ wrappers provided by `torch_tpu`.
  2. Applied hermetic bazel build script using:
     ```bash
     export TORCH_TPU_MODULE_PATH=/home/wenjung_google_com/torch_tpu
     export BAZEL_CACHE_DIR=/mnt/pd/temp/bazel_cache
     export BAZEL_OUTPUT_BASE=/mnt/pd/temp/bazel_output_torch212
     ./build.sh torch //ci/wheel:raiden_torch_wheel
     ```

---

### Issue 2: Dynamic Linker SONAME Pollution via TPU Initialization Noise

* **Symptom**:
  ```text
  ImportError: libpywrap_Device type: tpu, Device index: default
  Initializing TPU distributed runtime
  2_12_0_common.so: cannot open shared object file
  ```
* **Root Cause**:
  When `build.sh` queried Python for `torch.__version__`, importing `torch` automatically initialized the TPU backend, emitting runtime banner messages to `stdout`. The build script captured standard output directly into the SONAME suffix variable, baking multiline strings with newline characters into the binary's `DT_NEEDED` header.
* **Fix**:
  1. Suppressed backend autoload during queries:
     ```bash
     TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -c 'import torch; ...'
     ```
  2. Sanitized existing binaries with `patchelf`:
     ```bash
     chmod u+w tpu_sync/frameworks/torch/_tpu_raiden_torch.so
     patchelf --remove-needed "$(patchelf --print-needed tpu_sync/frameworks/torch/_tpu_raiden_torch.so | grep libpywrap)" tpu_sync/frameworks/torch/_tpu_raiden_torch.so
     patchelf --add-needed "libpywrap_2_12_0_common.so" tpu_sync/frameworks/torch/_tpu_raiden_torch.so
     ```

---

### Issue 3: In-Container Symlinks for `torch_tpu` Version Glue

* **Symptom**:
  The custom wheel links against `libpywrap_torch_tpu_common.so` or `libpywrap_2_12_0_common.so`, but runtime environments or base Docker images might ship differing suffixes (e.g., `libpywrap_2_11_0_common.so` or `libpywrap_2_13_0_common.so`).
* **Fix**:
  In `Dockerfile.tpu`, dynamically inspect Python site-packages and create symlinks for all major PyTorch glue variations, followed by updating `/etc/ld.so.conf.d/torch_tpu.conf` and executing `ldconfig`:
  ```dockerfile
  RUN for PY in /home/ray/anaconda3/bin/python3 /usr/bin/python3; do \
        if [ -x "$PY" ]; then \
          SITE_DIR=$($PY -c "import site; print(site.getsitepackages()[0])") && \
          $PY -m pip install --no-cache-dir --no-deps --force-reinstall /tmp/tpu_raiden_torch-*.whl && \
          if [ -d "${SITE_DIR}/torch_tpu/common" ]; then \
            cd "${SITE_DIR}/torch_tpu/common" && \
            ln -sfn libpywrap_torch_tpu_common.so libpywrap_2_11_0_common.so && \
            ln -sfn libpywrap_torch_tpu_common.so libpywrap_2_12_0_common.so && \
            ln -sfn libpywrap_torch_tpu_common.so libpywrap_2_13_0_common.so; \
          fi && \
          echo "${SITE_DIR}/torch_tpu/common" > /etc/ld.so.conf.d/torch_tpu.conf && \
          echo "${SITE_DIR}/torch_tpu" >> /etc/ld.so.conf.d/torch_tpu.conf && \
          echo "${SITE_DIR}/torch/lib" >> /etc/ld.so.conf.d/torch_tpu.conf; \
        fi; \
      done && \
      rm -f /tmp/tpu_raiden_torch-*.whl && \
      ldconfig
  ```

---

### Issue 4: Device Ordinal & Megatron Parallel Group Initialization in Mini RL

* **Symptom**:
  When running `mini_rl_reproduce.py`, worker processes hung indefinitely or threw:
  ```text
  RuntimeError: Device ordinal 0 already acquired or PJRT client initialization failed
  ```
* **Root Cause**:
  `torch_xla` requires each host worker process to explicitly manage its TPU chip ordinal through `PJRT_DEVICE=TPU` and `TPU_VISIBLE_DEVICES` or Ray placement groups. When Torchtitan initialized distributed tensor parallelism without passing rank-to-device mapping, multiple ranks contended for Chip 0.
* **Fix**:
  1. Updated worker initialization in `mini_rl_reproduce.py` to isolate TP groups:
     ```python
     os.environ["PJRT_DEVICE"] = "TPU"
     # Map local Ray TPU resource index directly to PJRT local device
     ```
  2. Verified standalone execution with `reproduce_standalone.py` on the remote TPU VM before deploying to GKE.

---

### Issue 5: GCS Lock & Model Download Deadlock

* **Symptom**:
  Hugging Face / GCS downloads in worker pods hung waiting for `.lock` files on gcsfuse mounted volumes.
* **Root Cause**:
  GCS CSI fuse filesystem does not support standard POSIX advisory file locks across multiple pods concurrently mounting the same bucket.
* **Fix**:
  Set cache directory to local `/tmp` instead of shared GCS volume:
  ```bash
  export HF_HOME="/tmp/huggingface"
  export TRANSFORMERS_CACHE="/tmp/huggingface"
  ```

---

### Issue 6: Destination Out of Bounds in Batched Push (Trainer vs. Sampler Layer Index Desynchronization)

* **Symptom**:
  When running full GRPO training with Torchtitan Trainer and vLLM Sampler, Step 0 weight sync fails with:
  ```text
  (vLLMHttpServer) ProcessPeerRequest failed: INVALID_ARGUMENT: Destination out of bounds in batched push
  tpu_sync/transport/lib/raw_buffer_transport.cc:342
  (WorkerDict) PushWeightsResharded native execution failed: INTERNAL: tcp socket writev failed: errno=104 (Connection reset by peer)
  ```
* **Root Cause**:
  In `raiden_controller.py`, during push schedule creation, chunk metadata is assigned `entry.layer_idx = src_var.layer_idx` (the Trainer's parameter index) instead of `dst_var.layer_idx` (the Sampler's parameter index).
  In full production models (e.g. Qwen-0.6B with 226+ parameters), Torchtitan and vLLM enumerate and register parameters in different orders due to differences in weight-tying, attention projection naming, and module hierarchies.
  When the TCP packet arrives at the vLLM receiver, `raw_buffer_transport.cc` queries `GetHostSize(meta.layer_idx, ...)` using the Trainer's layer index. Looking up the wrong tensor causes the slice byte offset and length to exceed the buffer bounds, failing the bounds check and dropping the TCP connection.
  Additionally:
  - `reproduce_standalone.py` missed this because it runs single-process mock initialization without network receivers.
  - `mini_rl_reproduce.py` missed this because both Trainer and Sampler used the identical `MiniModel` with identical parameter lists, making `src_var.layer_idx == dst_var.layer_idx` by coincidence.
* **Fix**:
  1. **Protocol Decoupling in `tpu-sync`**:
     - Update `raiden_service.proto` to include `dst_layer_idx` in `ShardPushScheduleEntryProto`.
     - In `weight_synchronizer_base.cc`, read local source memory using `entry.layer_idx()` and transmit `dst_layer_idx` in `task.buffer_id` to the destination.
     - In `raiden_controller.py`, match destination variables by string `name` rather than assuming index equality, and populate `dst_var.layer_idx` in destination schedules.
  2. **Parameter Name Normalization in verl**:
     - Normalize and sort parameter names across [`raiden_checkpoint_engine.py`](file:///usr/local/google/home/wenjung/verl-upstream/verl/checkpoint_engine/raiden_checkpoint_engine.py) and [`tpu_utils.py`](file:///usr/local/google/home/wenjung/verl-upstream/verl/workers/rollout/vllm_rollout/tpu_utils.py) to ensure canonical Hugging Face naming.

---

## 3. Validation Summary

| Test Case | Environment | Result |
| :--- | :--- | :--- |
| Standalone TPU Sync Import | Remote TPU VM (`wenjung-test-tpu-...`) | **PASS** (`import tpu_sync` succeeded) |
| `reproduce_standalone.py` (Local Mock) | Remote TPU VM (`venv_torch212`) | **PASS** (Zero errors, weight sync OK) |
| Ray Job `07000000` (`mini_rl_reproduce.py`) | GKE TPU v6e-8 Cluster (`alekseyv-tpu-...`) | **PASS** (Exit code 0, 100% completed) |
| Container Build (`Dockerfile.tpu`) | Local build & Artifact Registry push | **PASS** (`v-raiden-20260914171015`) |
| Ray Job `raysubmit_C226Ast272Ffk7wN` (GRPO Run) | GKE TPU v6e-8 Cluster | **FAILED** (Root caused: Issue 6 layer index mismatch) |

