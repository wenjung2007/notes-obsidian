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

## 3. Validation Summary

| Test Case | Environment | Result |
| :--- | :--- | :--- |
| Standalone TPU Sync Import | Remote TPU VM (`wenjung-test-tpu-...`) | **PASS** (`import tpu_sync` succeeded) |
| `reproduce_standalone.py` (Local Mock) | Remote TPU VM (`venv_torch212`) | **PASS** (Zero errors, weight sync OK) |
| Ray Job `07000000` (`mini_rl_reproduce.py`) | GKE TPU v6e-8 Cluster (`alekseyv-tpu-...`) | **PASS** (Exit code 0, 100% completed) |
| Container Build (`Dockerfile.tpu`) | Local build & Artifact Registry push | **PASS** (`v-raiden-20260914050623`) |
