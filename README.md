# GPU Programming

Hands-on CUDA lessons on Windows, from first kernel launch through memory and compute patterns.

This README is the living index — each new lesson gets a short entry here.

## Setup

**Required**

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (`nvcc` on `PATH`)
- Visual Studio 2022 **Build Tools** with MSVC (`cl.exe`)
- Windows 10/11 SDK (needed for linking, e.g. `uuid.lib`)

**Optional Python helpers** (plots / checks)

```powershell
pip install -r requirements.txt
python scripts\verify_env.py
```

## Build & run

From a shell that has MSVC loaded (or use the helper, which loads `vcvars64` for you):

```powershell
# one-shot: load VS x64 tools into this terminal
cmd /k '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"'

nvcc lessons\01_hello_cuda\hello_cuda.cu -o hello_cuda
.\hello_cuda.exe
```

Or:

```powershell
.\scripts\build_cu.ps1 lessons\01_hello_cuda\hello_cuda.cu -Run
.\scripts\build_cu.ps1 lessons\02_vector_addition\vector_addition.cu -Run
```

## Lessons

| # | Folder | Idea |
|---|--------|------|
| 01 | [`lessons/01_hello_cuda`](lessons/01_hello_cuda) | Launch a kernel; print from GPU threads |
| 02 | [`lessons/02_vector_addition`](lessons/02_vector_addition) | Host ↔ device memory; simple elementwise add |

### 01 — Hello CUDA

Minimal `__global__` kernel. Grid `<<<2, 5>>>` → 2 blocks × 5 threads; each thread prints its `threadIdx.x`. `cudaDeviceSynchronize()` so host waits for device prints.

### 02 — Vector addition

CPU loop (commented) vs GPU kernel: `cudaMalloc` / `cudaMemcpy` H2D → `addVectors<<<1, N>>>` → D2H → `cudaFree`. Indexing with `blockIdx.x * blockDim.x + threadIdx.x`.

## Layout

```
lessons/          # numbered CUDA exercises (.cu)
scripts/          # build helper + env smoke test
requirements.txt  # optional numpy/matplotlib
```

## Notes

- Prefer `.cu` + `nvcc` on Windows; `gcc` is not required for these lessons.
- Build artifacts (`*.exe`, `*.obj`, `*.lib`, `*.exp`, …) are gitignored.
