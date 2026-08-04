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

---

## Repository tracks

### CUDA fundamentals

[`cuda/`](cuda/README.md)

Small CUDA C++ programs used to learn:

* kernel launches
* thread and block indexing
* host and device memory
* flat matrix storage
* matrix operations
* reduction patterns
* shared memory and tiling

Setup instructions, build commands, lesson notes, and progress are maintained in the [CUDA README](cuda/README.md).

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
| CUDA fundamentals           | Lessons 01–04 complete, reduction in progress                          |
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
├── cuda/
│   └── README.md
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
├── triton/
├── hip/
├── tensorrt/
├── flashattention/
├── notes/
└── scripts/
```

Some top-level folders are placeholders for work I plan to add later. They should not be treated as completed tracks.

---

## Documentation structure

The repository has multiple README files with separate responsibilities.

| README                                                                           | Purpose                                                                |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [`README.md`](README.md)                                                         | High-level repository map and current progress                         |
| [`cuda/README.md`](cuda/README.md)                                               | CUDA setup, build commands, lesson notes, and progress                 |
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
