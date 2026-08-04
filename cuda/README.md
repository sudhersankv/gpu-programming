# CUDA track

Progressive CUDA C++ lessons using `nvcc` on Windows. Each lesson is a small, runnable `.cu` program.

For the rest of the repository (inference lab, Colab notebooks, dashboard), see the [root README](../README.md).

---

## Prerequisites

* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (`nvcc` on `PATH`)
* Visual Studio 2022 **Build Tools** with MSVC (`cl.exe`)
* Windows 10/11 SDK (linking, e.g. `uuid.lib`)

Optional repo helpers (from repo root):

```powershell
pip install -r requirements.txt
python scripts\verify_env.py
```

---

## Build & run

Load VS x64 tools once per shell:

```powershell
cmd /k '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"'
```

Direct `nvcc`:

```powershell
nvcc cuda\01_hello_cuda\hello_cuda.cu -o hello_cuda
.\hello_cuda.exe
```

Or the helper (from repo root):

```powershell
.\scripts\build_cu.ps1 cuda\01_hello_cuda\hello_cuda.cu -Run
.\scripts\build_cu.ps1 cuda\02_vector_addition\vector_addition.cu -Run
.\scripts\build_cu.ps1 cuda\03_matrix_addition\matrix_addition.cu -Run
.\scripts\build_cu.ps1 cuda\04_matrix_multiply\matmul.cu -Run
```

Prefer `.cu` + `nvcc` on Windows for this track; `gcc` is not required. Build artifacts are gitignored.

---

## Progress

| # | Folder | Status | Idea |
|---|--------|--------|------|
| 01 | [`01_hello_cuda`](01_hello_cuda) | done | Launch a kernel; print from GPU threads |
| 02 | [`02_vector_addition`](02_vector_addition) | done | Host ↔ device memory; elementwise add |
| 03 | [`03_matrix_addition`](03_matrix_addition) | done | 2D thread indexing; flat matrix storage |
| 04 | [`04_matrix_multiply`](04_matrix_multiply) | done | Naive GEMM; one thread per output; 2D grid |
| 05 | [`05_reduction`](05_reduction) | in progress | Parallel reduce patterns |
| 06 | `06_shared_memory` | planned | Tiling with `__shared__` |

---

## Lessons

### 01 — Hello CUDA

Minimal `__global__` kernel. Grid `<<<2, 5>>>` → 2 blocks × 5 threads; each thread prints its `threadIdx.x`. `cudaDeviceSynchronize()` so the host waits for device prints.

### 02 — Vector addition

CPU loop (commented) vs GPU kernel: `cudaMalloc` / `cudaMemcpy` H2D → `addVectors<<<1, N>>>` → D2H → `cudaFree`. Indexing with `blockIdx.x * blockDim.x + threadIdx.x`.

### 03 — Matrix addition

2×3 matrices as flat device arrays (`row * cols + col`). Kernel uses 2D indices from `blockIdx` + `threadIdx`. Launch with `dim3 threads(ROWS, COLS)` and one block: `matrixAddition<<<1, threads>>>`.

### 04 — Matrix multiply

Naive `C(M×N) = A(M×K) × B(K×N)`. Host keeps 2D arrays; device uses flat row-major. Each thread owns one `C[row][col]` and loops over `K`. Launch with `dim3 block(16,16)` and a ceiling-divided 2D grid over the output.

### 05 — Reduction

WIP — parallel reduction patterns (`reduction.cu`).

### 06 — Shared memory

Planned — tiling with `__shared__`.
