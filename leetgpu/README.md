# LeetGPU track

Kernel practice across **CUDA**, **Triton**, **PyTorch**, **JAX**, **Mojo**, and **CuTe DSL**, run through [LeetGPU](https://leetgpu.com) instead of a local `nvcc` / toolchain loop.

This folder is the source of truth for kernel drills. The [CLI](https://leetgpu.com/cli) compiles and executes remotely — no local GPU required for the practice loop.

Inference serving / Nsight work stays in [`llm-inference-runtime-lab/`](../llm-inference-runtime-lab/). Systems notebooks stay in [`inference-systems-notebooks/`](../inference-systems-notebooks/).

---

## Setup (once)

Windows (PowerShell):

```powershell
Invoke-WebRequest -Uri https://cli.leetgpu.com/install.ps1 -OutFile install.ps1; ./install.ps1
leetgpu --help
```

Verify from repo root:

```powershell
python scripts\verify_env.py
```

Optional: keep system `nvcc` + VS Build Tools if you want a local fallback (`.\scripts\build_cu.ps1`). Not required for this track.

---

## Workflow

1. Write a kernel under the matching language folder.
2. Run it with the CLI (or the helper script):

```powershell
leetgpu run leetgpu\cuda\01_hello_cuda\hello_cuda.cu
.\scripts\leetgpu_run.ps1 leetgpu\cuda\02_vector_addition\vector_addition.cu
```

3. For graded [challenges](https://leetgpu.com/challenges), save solutions under `*/challenges/` and submit on the site (CLI `run` is playground execution; the challenge judge lives on leetgpu.com).

---

## Layout

```text
leetgpu/
├── README.md                 ← you are here
├── cuda/                     ← progressive CUDA lessons + challenges/
├── triton/challenges/
├── pytorch/challenges/
├── jax/challenges/
├── mojo/challenges/
└── cute/challenges/
```

| Language | Path | Status |
|----------|------|--------|
| CUDA | [`cuda/`](cuda/) | Lessons 01–04 done; 05 in progress |
| Triton | [`triton/`](triton/) | Ready for challenges |
| PyTorch | [`pytorch/`](pytorch/) | Ready for challenges |
| JAX | [`jax/`](jax/) | Ready for challenges |
| Mojo | [`mojo/`](mojo/) | Ready for challenges |
| CuTe DSL | [`cute/`](cute/) | Ready for challenges |

### Naming

* Progressive drills: `NN_short_name/` with one main source file.
* Challenge solutions: `challenges/<slug>/` (match the LeetGPU problem name when you can).

---

## CUDA lessons

| # | Folder | Status | Idea |
|---|--------|--------|------|
| 01 | [`cuda/01_hello_cuda`](cuda/01_hello_cuda) | done | Launch a kernel; print from GPU threads |
| 02 | [`cuda/02_vector_addition`](cuda/02_vector_addition) | done | Host ↔ device memory; elementwise add |
| 03 | [`cuda/03_matrix_addition`](cuda/03_matrix_addition) | done | 2D thread indexing; flat matrix storage |
| 04 | [`cuda/04_matrix_multiply`](cuda/04_matrix_multiply) | done | Naive GEMM; one thread per output |
| 05 | [`cuda/05_reduction`](cuda/05_reduction) | in progress | Parallel reduce patterns |
| 06 | `cuda/06_shared_memory` | planned | Tiling with `__shared__` |

Run any lesson:

```powershell
.\scripts\leetgpu_run.ps1 leetgpu\cuda\01_hello_cuda\hello_cuda.cu
```

---

## Notes

* Prefer LeetGPU for iteration; use local `nvcc` only when you need machine-specific behavior or Nsight.
* Do not commit CLI installers (`install.ps1`) or large binary artifacts.
* Browser playground and challenges: [leetgpu.com](https://leetgpu.com).
