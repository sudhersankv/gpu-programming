# GPU Programming

Hands-on path from CUDA kernels → profiling → Triton / HIP / TensorRT / attention & serving stacks.

This README is the living index. Add a row (and a short section) when you open a new folder.

## Layout

```
gpu-programming/
├── README.md
├── cuda/                 # CUDA C++ lessons (nvcc)
│   ├── 01_hello_cuda/
│   ├── 02_vector_addition/
│   ├── 03_matrix_addition/
│   ├── 04_matrix_multiply/
│   ├── 05_reduction/         # planned
│   ├── 06_shared_memory/     # planned
│   └── ...
├── profiling/            # Nsight, roofline, perf notes
│   ├── nsight_systems/
│   ├── nsight_compute/
│   └── roofline/
├── triton/               # Triton kernels
├── hip/                  # AMD / HIP ports
├── tensorrt/             # inference engine
├── flashattention/       # attention kernels / papers practice
├── vllm/                 # LLM serving (planned)
├── llm-inference-runtime-lab/  # HF / vLLM serving benchmarks
├── notes/                # theory, cheat sheets
└── scripts/              # Windows build + env helpers
```

## Setup (CUDA track)

**Required**

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (`nvcc` on `PATH`)
- Visual Studio 2022 **Build Tools** with MSVC (`cl.exe`)
- Windows 10/11 SDK (linking, e.g. `uuid.lib`)

**Optional Python helpers**

```powershell
pip install -r requirements.txt
python scripts\verify_env.py
```

## Build & run (CUDA)

```powershell
# load VS x64 tools into this terminal (once per shell)
cmd /k '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"'

nvcc cuda\01_hello_cuda\hello_cuda.cu -o hello_cuda
.\hello_cuda.exe
```

Or:

```powershell
.\scripts\build_cu.ps1 cuda\01_hello_cuda\hello_cuda.cu -Run
.\scripts\build_cu.ps1 cuda\02_vector_addition\vector_addition.cu -Run
.\scripts\build_cu.ps1 cuda\03_matrix_addition\matrix_addition.cu -Run
.\scripts\build_cu.ps1 cuda\04_matrix_multiply\matmul.cu -Run
```

## Progress

### CUDA

| # | Folder | Status | Idea |
|---|--------|--------|------|
| 01 | [`cuda/01_hello_cuda`](cuda/01_hello_cuda) | done | Launch a kernel; print from GPU threads |
| 02 | [`cuda/02_vector_addition`](cuda/02_vector_addition) | done | Host ↔ device memory; elementwise add |
| 03 | [`cuda/03_matrix_addition`](cuda/03_matrix_addition) | done | 2D thread indexing; flat matrix storage |
| 04 | [`cuda/04_matrix_multiply`](cuda/04_matrix_multiply) | done | Naive GEMM; one thread per output; 2D grid |
| 05 | [`cuda/05_reduction`](cuda/05_reduction) | planned | Parallel reduce patterns |
| 06 | [`cuda/06_shared_memory`](cuda/06_shared_memory) | planned | Tiling with `__shared__` |

### Other tracks

| Area | Folder | Status |
|------|--------|--------|
| Profiling | [`profiling/`](profiling/) | next (learning tools) |
| Triton | [`triton/`](triton/) | planned |
| HIP | [`hip/`](hip/) | planned |
| TensorRT | [`tensorrt/`](tensorrt/) | planned |
| FlashAttention | [`flashattention/`](flashattention/) | planned |
| vLLM | [`vllm/`](vllm/) | planned |
| Inference lab | [`llm-inference-runtime-lab/`](llm-inference-runtime-lab/) | Phases 1–5 done; next = profiling |
| Notes | [`notes/`](notes/) | planned |

### 01 — Hello CUDA

Minimal `__global__` kernel. Grid `<<<2, 5>>>` → 2 blocks × 5 threads; each thread prints its `threadIdx.x`. `cudaDeviceSynchronize()` so host waits for device prints.

### 02 — Vector addition

CPU loop (commented) vs GPU kernel: `cudaMalloc` / `cudaMemcpy` H2D → `addVectors<<<1, N>>>` → D2H → `cudaFree`. Indexing with `blockIdx.x * blockDim.x + threadIdx.x`.

### 03 — Matrix addition

2×3 matrices as flat device arrays (`row * cols + col`). Kernel uses 2D indices: `row` / `col` from `blockIdx` + `threadIdx`. Launch with `dim3 threads(ROWS, COLS)` and one block: `matrixAddition<<<1, threads>>>`.

### 04 — Matrix multiply

Naive `C(M×N) = A(M×K) × B(K×N)`. Host keeps 2D arrays; device uses flat row-major (`A[row*K+k]`, `B[k*N+col]`, `C[row*N+col]`). Each thread owns one `C[row][col]` and loops over `K`. Launch with `dim3 block(16,16)` and a ceiling-divided 2D grid over the output.

## Notes

- Prefer `.cu` + `nvcc` on Windows for the `cuda/` track; `gcc` is not required.
- Build artifacts (`*.exe`, `*.obj`, `*.lib`, `*.exp`, …) are gitignored.
- Empty planned folders use `.gitkeep` so the tree stays visible until real code lands.
