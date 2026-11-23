# ROCm Support for LlamaForge

## Overview

This document describes the implementation of full ROCm (AMD GPU) support for LlamaForge, enabling Auto-Forge and GPU optimization features on AMD hardware.

## Current State

- **GPU Detection**: Uses `nvidia-smi` exclusively for NVIDIA GPUs
- **AMD Telemetry**: Basic VRAM info via `llama-server --list-devices` and Linux sysfs (no utilization/temp)
- **Auto-Forge**: Hardcoded to CUDA device naming (`CUDA{index}`)
- **Build Flags**: Only generates CUDA flags (`GGML_CUDA`, `CMAKE_CUDA_ARCHITECTURES`)

## ROCm Tooling

### rocm-smi CLI

Query AMD GPU information in CSV format:

```bash
rocm-smi --csv
```

**Output columns**: `index,name,driver_version,pcie_link_speed,pcie_lane_count,status,vram_total,vram_used,vram_free,gpu_temp,utilization,fan_speed`

**Example output**:
```csv
index,name,driver_version,pcie_link_speed,pcie_lane_count,status,vram_total,vram_used,vram_free,gpu_temp,utilization,fan_speed
0,"AMD Radeon RX 7900 XTX",23.6.1.2,x4.0,16,active,24576,8192,16384,58,45,52
```

**Useful flags**:
- `-a, --showallinfo` - Show temperature, fan, clock values
- `-t, --showtemp` - Show current temperature
- `-u, --showuse` - Show GPU utilization (%)
- `--showmeminfo TYPE` - Show VRAM usage (vram/vis_vram/gtt)

### Fallback Detection

If `rocm-smi` is unavailable:
1. Try `rocminfo | grep gfx` - shows GPU architecture
2. Try `amd-smi --rocm-smi` (ROCm 8.x+ compatibility mode)
3. Check for `/opt/rocm` installation directory

## llama.cpp ROCm Build Flags

### Required Flags

```bash
HIPCXX="$(hipconfig -l)/clang" \
HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build \
    -DGGML_HIP=ON \
    -DGPU_TARGETS=gfx1100 \
    -DCMAKE_BUILD_TYPE=Release
```

### Key Flags

| Flag | Purpose |
|------|---------|
| `-DGGML_HIP=ON` | Enable HIP/ROCm backend (required) |
| `-DGPU_TARGETS=gfxXXXX` | Specify GPU architecture(s) to compile for |
| `-DCMAKE_BUILD_TYPE=Release` | Release build optimization |

### GPU Architecture Mapping

| GPU Product | Architecture Flag |
|-------------|-------------------|
| RX 7900 XTX/XT/GRE | `gfx1100` |
| RX 7800 XT/7700 XT | `gfx1101` |
| RX 7600 | `gfx1102` |
| RX 6800/6700 | `gfx1030` |
| MI200/MI300 | `gfx90a` |
| MI100 | `gfx908` |

Multiple architectures: `-DGPU_TARGETS=gfx1030,gfx1100`

## llama.cpp ROCm Device Naming

When built with ROCm support, llama.cpp uses:
- **`ROCm0`, `ROCm1`, `ROCm2`** - modern naming (upstream standard)
- **`HIP0`** - legacy name (backward compatibility)

**Device listing**:
```bash
./llama-cli --list-devices
# Output:
# Available devices:
#   ROCm0: AMD Radeon RX 7900 XTX (24576 MiB, 16384 MiB free)
#   ROCm1: AMD Radeon Graphics (63199 MiB, 63174 MiB free)
```

**Using specific device**:
```bash
./llama-cli -m model.gguf -ngl 99 --device ROCm0
```

**Multi-GPU**:
```bash
./llama-cli -m model.gguf -ngl 99 --device ROCm0,ROCm1
```

## Implementation Plan

### Phase 1: GPU Detection (`backend/hardware.py`)

1. Add `detect_amd_gpus()` function that queries `rocm-smi --csv`
2. Parse CSV output to extract: index, name, vram_total, vram_used, gpu_temp, utilization
3. Add fallback to `rocminfo` if `rocm-smi` unavailable
4. Update `detect_gpus()` to try both NVIDIA and AMD detection
5. Return unified GPU object format with `backend: "nvidia"` or `backend: "amd"`

### Phase 2: Build Flag Generation (`backend/hardware.py`)

1. Add AMD architecture detection from GPU name/model
2. Update `recommend()` to generate ROCm flags when AMD GPUs detected:
   - `GGML_HIP=ON`
   - `GPU_TARGETS=gfxXXXX` (derived from GPU model)
3. Handle mixed NVIDIA/AMD systems (prefer CUDA, warn about multi-backend)

### Phase 3: Auto-Forge Device Naming (`backend/forge/space.py`)

1. Replace hardcoded `f"CUDA{g['index']}"` with backend-agnostic naming
2. Create helper function `device_name(gpu)` that returns:
   - `CUDA{index}` for NVIDIA GPUs
   - `ROCm{index}` for AMD GPUs
3. Update all placement generation code to use the helper

### Phase 4: GPU Telemetry (`backend/gpuprobe.py`)

1. Enhance `other_gpus()` to use `rocm-smi` for utilization/temperature when available
2. Update telemetry output to include `util` and `temp` for ROCm devices
3. Maintain fallback to `llama-server --list-devices` and sysfs

### Phase 5: Testing & Validation

1. Test GPU detection on AMD hardware
2. Test build flag generation for various AMD GPU models
3. Test Auto-Forge candidate generation with ROCm device naming
4. Test end-to-end Auto-Forge run on AMD hardware

## Files to Modify

- `backend/hardware.py` - GPU detection and build flag generation
- `backend/forge/space.py` - Device naming for Auto-Forge candidates
- `backend/gpuprobe.py` - Enhanced ROCm telemetry
- `docs/content/config.md` - Documentation updates
- `README.md` - Update GPU support documentation

## Dependencies

- ROCm SDK 5.4+ installed on system
- `rocm-smi` tool available (part of ROCm installation)
- AMDGPU kernel driver (Linux)

## Known Limitations

- ROCm support is Linux-only (no Windows ROCm)
- Some llama.cpp features may have different performance characteristics on ROCm vs CUDA
- Flash attention support depends on ROCm version and llama.cpp build
