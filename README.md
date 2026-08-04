# GPU Programming

Hands-on work across **CUDA kernels**, **LLM inference runtimes**, and **systems notebooks** — building intuition from small experiments up to real serving stacks.

This repository is a learning lab and portfolio of tracks. Each major folder owns its own documentation. **This root README is the map**, not a copy of every experiment.

---

## Why this exists

Modern GPU and LLM-serving work spans several layers:

* writing and reasoning about CUDA kernels  
* measuring inference engines (TTFT, throughput, memory, prefix reuse)  
* understanding the systems ideas behind paged KV, scheduling, and attention IO  

Those are easy to blur together. Here they stay in **separate tracks** with clear entry points.

---

## Start here

| If you want to… | Go to |
|-----------------|--------|
| Learn CUDA C++ kernels on Windows | [`cuda/`](cuda/) |
| Benchmark and profile vLLM / prefix cache / LMCache | [`llm-inference-runtime-lab/`](llm-inference-runtime-lab/) |
| Work through Colab toys (KV, paging, scheduling, …) | [`inference-systems-notebooks/`](inference-systems-notebooks/) |
| See the APC Nsight case study in the browser | [Live dashboard](https://sudhersankv.github.io/gpu-programming/) · [`docs/`](docs/) |

---

## Tracks

### CUDA kernels — [`cuda/`](cuda/README.md)

Progressive `nvcc` lessons: hello world → memory → indexing → matmul → reduction / shared memory (in progress).

**Setup, build commands, and per-lesson notes:** see the [CUDA track README](cuda/README.md).

### LLM inference runtime lab — [`llm-inference-runtime-lab/`](llm-inference-runtime-lab/README.md)

Dockerized serving experiments: Hugging Face baseline → vLLM → shared-prefix (Qasper) benchmarks → Nsight Systems → LMCache smoke test → next engines (SGLang, …).

**Methodology, results, scripts, and reproducibility** live only in that README — do not duplicate them here.

### Inference systems notebooks — [`inference-systems-notebooks/`](inference-systems-notebooks/README.md)

Standalone Colab notebooks: toy implementations first, then links to real runtimes. Complements the Docker lab; no container required.

**Curriculum, status, and notebook philosophy** live only in that README.

### Case study dashboard — [`docs/`](docs/)

Static GitHub Pages site summarizing the warm cache ON/OFF Nsight experiment from the runtime lab.

### Planned / thin folders

| Folder | Intent |
|--------|--------|
| [`profiling/`](profiling/) | Cross-cutting Nsight / roofline notes (lab Phase 6 is the main Nsight work today) |
| [`triton/`](triton/) | Triton kernels |
| [`hip/`](hip/) | AMD / HIP ports |
| [`tensorrt/`](tensorrt/) | TensorRT |
| [`flashattention/`](flashattention/) | Attention kernels / paper practice |
| [`notes/`](notes/) | Theory and cheat sheets |

Each of these should get its own README when real content lands.

---

## Progress (high level)

| Track | Status |
|-------|--------|
| CUDA kernels | Lessons 01–04 done; 05–06 planned — [details](cuda/README.md) |
| Inference runtime lab | Phases 1–7 done; Phase 8+ (SGLang, …) next — [details](llm-inference-runtime-lab/README.md) |
| Systems notebooks | Curriculum published; notebooks uploading over time — [details](inference-systems-notebooks/README.md) |
| GitHub Pages dashboard | Available — [open](https://sudhersankv.github.io/gpu-programming/) |

---

## Repository layout

```text
gpu-programming/
├── README.md                         ← you are here (navigation)
├── cuda/                             ← CUDA C++ track (+ README)
├── llm-inference-runtime-lab/        ← serving / bench / Nsight / LMCache
├── inference-systems-notebooks/      ← Colab curriculum
├── docs/                             ← GitHub Pages dashboard
├── profiling/ triton/ hip/ …         ← planned tracks
└── scripts/                          ← shared Windows helpers
```

---

## Documentation ownership

| README | Owns |
|--------|------|
| **This file** | What the repo is, track map, where to start |
| [`cuda/README.md`](cuda/README.md) | CUDA setup, builds, lesson notes |
| [`llm-inference-runtime-lab/README.md`](llm-inference-runtime-lab/README.md) | Lab methodology, results, scripts |
| [`inference-systems-notebooks/README.md`](inference-systems-notebooks/README.md) | Notebook philosophy and curriculum |
| [`docs/README.md`](docs/README.md) | How to rebuild / deploy the dashboard |

Detailed content belongs in the track README. The root only summarizes and links.

---

## Shared notes

* Prefer track-local docs over expanding this file.
* Build artifacts (`*.exe`, `*.obj`, …) and large profiler traces are gitignored.
* Empty planned folders may use `.gitkeep` so the tree stays visible.
