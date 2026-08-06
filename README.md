# GPU Programming and LLM Inference Systems

This is my personal repository for learning GPU programming and modern LLM inference systems by building, benchmarking, profiling, and documenting things as I go.

I use it to keep a record of what I implemented, the experiments I ran, the results I observed, and how my understanding changed over time. The documentation is mainly so I can return later and quickly reconstruct the reasoning behind the work, but the code and notes may also be useful to others exploring similar topics.

This is an evolving learning repository, not a finished framework or formal course.

---

## Current focus

I am currently building small inference-systems notebooks to understand individual concepts before returning to broader runtime comparisons.

The current sequence is:

* prefill, decode, and KV caching
* KV-cache memory scaling
* paged KV memory and block tables
* automatic prefix caching
* RadixAttention and SGLang
* continuous batching and scheduling
* memory tiers and KV-cache offloading

The real-runtime track currently covers vLLM, Nsight Systems, and LMCache. SGLang is the next planned runtime comparison, followed later by TensorRT-LLM, ROCm, concurrency experiments, and deeper profiling.

Kernel practice has moved to a [LeetGPU](https://leetgpu.com)-first workflow (CUDA, Triton, PyTorch, JAX, Mojo, CuTe) so I can iterate without a local `nvcc` loop.

---

## Repository tracks

### LeetGPU kernels

[`leetgpu/`](leetgpu/README.md)

Primary kernel track. Write under `leetgpu/<language>/`, run with the [LeetGPU CLI](https://leetgpu.com/cli) — no local GPU required for the practice loop.

```powershell
# once
Invoke-WebRequest -Uri https://cli.leetgpu.com/install.ps1 -OutFile install.ps1; ./install.ps1

# run a lesson
.\scripts\leetgpu_run.ps1 leetgpu\cuda\01_hello_cuda\hello_cuda.cu
```

Languages: CUDA, Triton, PyTorch, JAX, Mojo, CuTe DSL. CUDA lessons cover kernel launches, indexing, memory, matmul, reduction, and (planned) shared memory.

Setup, CLI usage, lesson notes, and progress live in the [LeetGPU README](leetgpu/README.md).

> Legacy path: [`cuda/`](cuda/) only redirects here. Prefer `leetgpu/`.

---

### Inference systems notebooks

[`inference-systems-notebooks/`](inference-systems-notebooks/README.md)

Small, standalone Colab notebooks used to understand one inference-system concept at a time.

The notebooks use toy implementations, calculations, simulations, and visualizations before connecting the idea to real runtimes such as vLLM, SGLang, LMCache, and TensorRT-LLM.

Available now:

* [`01_prefill_decode_and_kv_cache.ipynb`](inference-systems-notebooks/01_prefill_decode_and_kv_cache.ipynb)

Planned topics include KV-cache memory scaling, paged KV allocation, automatic prefix caching, RadixAttention, scheduling, FlashAttention, chunked prefill, speculative decoding, parallelism, quantization, and MoE inference.

The complete topic list and notebook-specific notes live in the [notebook README](inference-systems-notebooks/README.md).

---

### LLM inference runtime lab

[`llm-inference-runtime-lab/`](llm-inference-runtime-lab/README.md)

Real inference-engine experiments using Docker, Hugging Face, vLLM, Nsight Systems, and LMCache.

Completed work includes:

* Hugging Face generation baseline
* vLLM serving baseline
* reusable benchmark harness
* TTFT, end-to-end latency, and output-token throughput measurements
* deterministic Qasper shared-prefix and control workloads
* vLLM automatic prefix-caching experiments
* controlled warm cache-on versus cache-off runs
* Nsight Systems timeline analysis
* LMCache store-and-hit smoke testing

The detailed methodology, commands, benchmark artifacts, profiler screenshots, and conclusions are maintained in the [runtime-lab README](llm-inference-runtime-lab/README.md).

---

### Automatic prefix-caching case study

[Live dashboard](https://sudhersankv.github.io/gpu-programming/) · [`docs/`](docs/)

A static visualization of one controlled vLLM automatic-prefix-caching experiment.

The experiment compares warm cache-on and cache-off behavior using two requests with a shared Qasper document prefix.

The profiling observations suggest that a prefix-cache hit reuses KV already resident in GPU memory. The small host-to-device transfers observed around the request are metadata-sized and consistent with runtime metadata rather than bulk KV movement.

The Nsight timeline alone does not prove that any individual transfer is specifically a block-table copy.

---

## Progress

| Track                       | Current status                                                         |
| --------------------------- | ---------------------------------------------------------------------- |
| LeetGPU kernels             | CUDA lessons 01–04 complete; 05 in progress; other languages scaffolded |
| Inference systems notebooks | Notebook 01 available; remaining notebooks added as I study each topic |
| Inference runtime lab       | Phases 1–7 complete                                                    |
| SGLang comparison           | Planned next runtime phase                                             |
| TensorRT-LLM experiments    | Planned                                                                |
| AMD HIP / ROCm experiments  | Planned                                                                |
| Deeper profiling            | Nsight Compute and PyTorch Profiler planned                            |

### Runtime lab phases

| Phase | Topic                         | Status   |
| ----- | ----------------------------- | -------- |
| 1     | Hugging Face baseline         | Complete |
| 2     | vLLM baseline                 | Complete |
| 3     | Benchmark framework           | Complete |
| 4     | Qasper shared-prefix workload | Complete |
| 5     | vLLM automatic prefix caching | Complete |
| 6     | Nsight Systems profiling      | Complete |
| 7     | LMCache smoke test            | Complete |
| 8     | SGLang comparison             | Planned  |
| 9     | TensorRT-LLM comparison       | Planned  |
| 10    | AMD ROCm experiments          | Planned  |
| 11    | Deeper profiling              | Planned  |

---

## Repository layout

```text
gpu-programming/
├── README.md
│
├── leetgpu/
│   ├── README.md
│   ├── cuda/                 ← progressive CUDA lessons + challenges/
│   ├── triton/ pytorch/ jax/ mojo/ cute/
│
├── inference-systems-notebooks/
│   ├── README.md
│   └── 01_prefill_decode_and_kv_cache.ipynb
│
├── llm-inference-runtime-lab/
│   ├── README.md
│   ├── docker/
│   ├── prompts/
│   ├── profiling/
│   ├── results/
│   └── scripts/
│
├── docs/
│   └── GitHub Pages prefix-caching case study
│
├── profiling/
├── hip/
├── tensorrt/
├── flashattention/
├── notes/
└── scripts/                  ← leetgpu_run.ps1, optional build_cu.ps1
```

Some top-level folders are placeholders for work I plan to add later. They should not be treated as completed tracks. Kernel languages supported by LeetGPU live under `leetgpu/`.

---

## Documentation structure

The repository has multiple README files with separate responsibilities.

| README                                                                           | Purpose                                                                |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [`README.md`](README.md)                                                         | High-level repository map and current progress                         |
| [`leetgpu/README.md`](leetgpu/README.md)                                         | LeetGPU CLI setup, languages, CUDA lesson notes                        |
| [`inference-systems-notebooks/README.md`](inference-systems-notebooks/README.md) | Notebook topics, design notes, and status                              |
| [`llm-inference-runtime-lab/README.md`](llm-inference-runtime-lab/README.md)     | Runtime methodology, benchmark results, profiling, and reproducibility |
| [`docs/README.md`](docs/README.md)                                               | Dashboard generation and deployment                                    |

Detailed results stay in the relevant track README rather than being duplicated here.

---

## Notes on the results

Most experiments in this repository are run on accessible development hardware, including an NVIDIA RTX 4060 Laptop GPU.

The measurements are specific to the documented:

* model
* runtime version
* prompt workload
* GPU
* container configuration
* cache state
* warmup procedure
* concurrency level

They are intended to help me understand runtime behavior. They should not be interpreted as general datacenter-performance claims.

Some explanations and conclusions may be revised as I learn more or inspect the systems at a deeper level.

Kernel practice prefers the LeetGPU CLI; local `nvcc` (`scripts/build_cu.ps1`) is optional.
