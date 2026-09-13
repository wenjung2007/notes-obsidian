---
title: "Troubleshooting: Recent TPU Sync Build & Import Errors"
date: "2026-09-13"
tags:
  - tpu-sync
  - pytorch
  - bazel
  - patchelf
  - debugging
  - troubleshooting
---

# 🛠️ Troubleshooting: Recent TPU Sync Build & Import Errors

This note summarizes the root causes and solutions for the last three major errors encountered while building and testing the `tpu-sync` PyTorch wheel and native extensions.

---

## 1. Bazel Target Not Found: `@torch_tpu//shims/torch:local_torch`

### ❌ Symptom
When executing `./build.sh torch //ci/wheel:raiden_torch_wheel`, Bazel aborted with:
```text
ERROR: Skipping '@torch_tpu//shims/torch:local_torch': no such target '@@torch_tpu+//shims/torch:local_torch':
target 'local_torch' not declared in package 'shims/torch' defined by .../torch_tpu+/shims/torch/BUILD.bazel (did you mean use_local_torch?)
```

### 🔍 Root Cause
1. **Module Override Missing**: `build.sh` looks for the local `torch_tpu` module at `../torch_tpu`. In this workspace (`/mnt/pd/tpu-sync`), `torch_tpu` is located at `/home/wenjung_google_com/torch_tpu`.
2. **Flag Name Mismatch**: Because Bazel could not find `../torch_tpu`, it fell back to downloading the remote pinned `torch_tpu` dependency (`v0.1.1`), where the bool flag was named `use_local_torch` instead of `local_torch`.
3. **Unactivated Virtualenv**: Running without activating the virtual environment caused `build.sh` to use the host Python 3.10 instead of Python 3.12 with PyTorch 2.12.

### ✅ Solution
Export `TORCH_TPU_MODULE_PATH`, set Bazel cache paths, and activate the virtual environment:
```bash
source /home/wenjung_google_com/venv_torch212/bin/activate
export TORCH_TPU_MODULE_PATH=/home/wenjung_google_com/torch_tpu
export BAZEL_CACHE_DIR=/mnt/pd/temp/bazel_cache
export BAZEL_OUTPUT_BASE=/mnt/pd/temp/bazel_output_torch212

./build.sh torch //ci/wheel:raiden_torch_wheel
```

---

## 2. Corrupted `DT_NEEDED` Entry via Stdout Pollution

### ❌ Symptom
Importing the compiled extension failed with:
```text
ImportError: libpywrap_Device type: tpu, Device index: default
Initializing TPU distributed runtime
2_12_0_common.so: cannot open shared object file: No such file or directory
```

### 🔍 Root Cause
During the build process, `build.sh` extracted the PyTorch version suffix using:
```bash
python3 -c "import torch; ..."
```
Upon importing `torch`, the TPU backend initialization logged `Device type: tpu, Device index: default\nInitializing TPU distributed runtime` to `stdout`. This noisy output was captured into the suffix variable (`TORCH_GLUE_SUFFIX`) and passed into `patchelf --add-needed`, creating an invalid library name with embedded newline characters in `_tpu_raiden_torch.so`.

### ✅ Solution
1. **Suppress Autoload Logging**: Set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` when querying Python for torch versions:
   ```bash
   TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -c 'import torch; ...'
   ```
2. **Clean Corrupted Shared Library**: Remove the malformed entry and inject the clean SONAME:
   ```bash
   patchelf --remove-needed "$(patchelf --print-needed tpu_sync/frameworks/torch/_tpu_raiden_torch.so | grep libpywrap)" tpu_sync/frameworks/torch/_tpu_raiden_torch.so
   patchelf --add-needed "libpywrap_2_12_0_common.so" tpu_sync/frameworks/torch/_tpu_raiden_torch.so
   ```

---

## 3. Patchelf Error with Unversioned Glue Name & Invalid Relative RPATH

### ❌ Symptom
Attempting to patch `_tpu_raiden_torch.so` with:
```bash
patchelf --add-rpath '$ORIGIN/../../../torch/lib:$ORIGIN/../../../torch_tpu/common' \
         --add-needed "libtorch_python.so" \
         --add-needed "libpywrap_torch_tpu_common.so" \
         tpu_sync/frameworks/torch/_tpu_raiden_torch.so
```
resulted in permission errors or runtime `ImportError: libpywrap_torch_tpu_common.so: cannot open shared object file`.

### 🔍 Root Cause
1. **Incorrect Library Name**: `torch_tpu` builds version-suffixed glue binaries (e.g. `libpywrap_2_12_0_common.so`), not an unversioned `libpywrap_torch_tpu_common.so`.
2. **Invalid Source Tree RPATH**: `$ORIGIN/../../../` is designed for installed wheels located in `site-packages/tpu_sync/frameworks/torch/`. When evaluated inside the repository tree (`/mnt/pd/tpu-sync/tpu_sync/frameworks/torch/`), it resolves to `/mnt/pd/` where no libraries exist.
3. **Read-Only File Permissions**: Bazel builds shared objects with `0555` (read-only) permissions. `patchelf` cannot modify the binary without write access.

### ✅ Solution
1. **Add write permission**:
   ```bash
   chmod u+w tpu_sync/frameworks/torch/_tpu_raiden_torch.so
   ```
2. **Add only the version-matched `DT_NEEDED` entry**:
   ```bash
   patchelf --add-needed "libpywrap_2_12_0_common.so" tpu_sync/frameworks/torch/_tpu_raiden_torch.so
   ```
3. **Verify with `patchelf --print-needed`**:
   ```bash
   patchelf --print-needed tpu_sync/frameworks/torch/_tpu_raiden_torch.so
   ```
