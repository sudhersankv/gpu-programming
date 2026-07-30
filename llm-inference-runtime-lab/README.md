# LLM Inference Runtime Lab

A hands-on exploration of modern LLM inference systems, benchmarking, profiling, and runtime optimization.

The goal of this project is to understand how different inference engines execute transformer models, identify performance bottlenecks, and evaluate techniques such as shared-prefix caching, KV-cache management, and continuous batching.

This project is being built incrementally. Each phase introduces a new inference system concept and benchmarks it before moving on to more advanced runtime optimizations.

---

## Project Roadmap

* [x] **Phase 1** — Hugging Face baseline
* [x] **Phase 2** — vLLM baseline (Docker)
* [x] **Phase 3** — Benchmark framework
* [x] **Phase 4** — Shared-prefix workload generation (Qasper)
* [x] **Phase 5** — vLLM automatic prefix caching (shared vs control)
* [ ] **Phase 6** — Profiling (Nsight / PyTorch Profiler) ← *next (learning first)*
* [ ] Phase 7 — LMCache integration
* [ ] Phase 8 — SGLang comparison
* [ ] Phase 9 — TensorRT-LLM comparison
* [ ] Phase 10 — AMD ROCm experiments

---

# Current Progress

## Phase 1 — Hugging Face Baseline

* Load `Qwen/Qwen2.5-1.5B-Instruct` with Transformers
* Stream one generation; measure load time, TTFT, E2E, output tok/s
* Script: `hf_baseline.py` (`run_once()` used by the harness)

## Phase 2 — vLLM Baseline

* Serve the same model via Docker image `llm-lab-vllm:phase2`
* Stream OpenAI-compatible chat completions from the host
* Script: `vllm_baseline.py` (`run_once()` used by the harness)

## Phase 3 — Benchmark Framework

* 10 short prompts in `prompts/short.json`
* `run_benchmark.py` — choose engine, warmup, N trials, one JSONL
* `summarize.py` — mean / p50 / p90 / p95 / p99 (skips warmup)
* `plot_compare.py` — HF vs vLLM bar charts

## Phase 4 — Shared-prefix workload (Qasper)

* `prepare_qasper_workload.py` builds a fixed, reproducible fixture from **allenai/qasper**
* Tokenizer: `Qwen/Qwen2.5-1.5B-Instruct`
* Paper truncated to **1024** tokens; `NUM_PROMPTS = 10`; `max_new_tokens = 64`
* Prompt template: paper context → question → `Answer:`

| File | Role |
|------|------|
| `prompts/qasper_shared_1024.json` | Same 1024-token paper + 10 different questions (reusable prefix) |
| `prompts/qasper_control_1024.json` | 10 different papers × 1 question each (no shared prefix) |

Regenerate:

```powershell
python prepare_qasper_workload.py
```

## Phase 5 — vLLM prefix caching

* Same harness (`run_benchmark.py`) against vLLM with automatic prefix caching enabled
* Compared **shared** vs **control** Qasper fixtures (one measured trial per prompt in the recorded run)
* `plot_qasper_prefix.py` — per-request TTFT/E2E bars, trajectory, and first-vs-rest summary

**Artifacts**

| File | Description |
|------|-------------|
| `results/qasper_shared_cache_on.jsonl` | Shared-prefix run (local; gitignored raw JSONL) |
| `results/qasper_control_cache_on.jsonl` | Control run (local; gitignored raw JSONL) |
| `results/qasper_prefix_compare.png` | Shared vs control TTFT / E2E plot |

### Prefix-cache plot

![Qasper shared vs control with vLLM prefix cache on](results/qasper_prefix_compare.png)

```powershell
python plot_qasper_prefix.py `
  --shared results\qasper_shared_cache_on.jsonl `
  --control results\qasper_control_cache_on.jsonl `
  --out results\qasper_prefix_compare.png
```

### Headline numbers (this machine, cache on)

| Slice | TTFT (s) |
|-------|----------|
| Shared — first request (cold prefix) | **2.335** |
| Shared — mean of requests 1–9 (warm prefix) | **0.039** |
| Control — mean across 10 requests | **0.172** |

**Takeaways**

* After the first shared request fills the prefix KV cache, later shared questions hit ~**0.04 s** TTFT.
* Control stays ~**0.15–0.25 s** TTFT — every prompt has a unique long paper prefix.
* Warm shared TTFT is roughly **4×** lower than control mean, and ~**60×** lower than the shared cold first request.
* E2E still includes decode; TTFT is the clearest prefix-cache signal on this fixture.

---

## Phase 3 Results (laptop, RTX 4060)

**Config**

* Model: `Qwen/Qwen2.5-1.5B-Instruct`
* Prompts: `prompts/short.json` (10 questions)
* `max_new_tokens=64`, temperature 0
* Warmup: 1 per prompt (excluded from stats)
* Measured trials: 5 per prompt → **50 measured requests per engine**
* Hardware: NVIDIA RTX 4060 Laptop (~8 GB), Windows host + vLLM in Docker

**Artifacts**

| File | Description |
|------|-------------|
| `results/bench_vllm_20260728_191524.jsonl` | Raw vLLM trials |
| `results/bench_hf_20260728_191640.jsonl` | Raw HF trials |
| `results/bench_vllm_20260728_191524.summary.json` | Percentile summary |
| `results/bench_hf_20260728_191640.summary.json` | Percentile summary |
| `results/compare_hf_vllm.png` | Side-by-side mean bars (TTFT / E2E / tok/s) |

### Comparison plot

![HF vs vLLM — mean TTFT, E2E, and output tok/s across 10 prompts](results/compare_hf_vllm.png)

Regenerate the plot:

```powershell
python plot_compare.py `
  results\bench_hf_20260728_191640.jsonl `
  results\bench_vllm_20260728_191524.jsonl `
  --out results\compare_hf_vllm.png
```

### Aggregate (mean of per-prompt means)

| Engine | TTFT (s) | E2E (s) | Output tok/s |
|--------|----------|---------|--------------|
| Hugging Face | 0.116 | 2.21 | 30.2 |
| vLLM | **0.037** | **1.06** | **62.8** |
| Speedup (HF → vLLM) | ~3.1× | ~2.1× | ~2.1× |

### One prompt detail — `kv_cache_short` (mean, n=5)

| Engine | TTFT (s) | E2E (s) | Output tok/s |
|--------|----------|---------|--------------|
| Hugging Face | 0.094 | 2.05 | 29.6 |
| vLLM | 0.036 | 1.01 | 65.9 |

### Sample percentile lines (vLLM `ttft_meaning`, n=5)

```text
ttft_s             mean=0.0375  p50=0.0378  p90=0.0396  p95=0.0399  p99=0.0402
e2e_s              mean=0.9721  p50=0.9725  p90=0.9743  p95=0.9747  p99=0.9750
output_tok_per_s   mean=68.48   p50=68.53   p90=68.77   p95=68.82   p99=68.87
```

**Takeaways**

* On short prompts, warm vLLM is clearly faster than naive HF `generate` on this laptop.
* Decode throughput roughly **doubles**; TTFT drops by about **3×**.
* Per-prompt variance is small for TTFT on vLLM (~35–40 ms means); HF TTFT spans ~88–165 ms depending on the prompt.
* These are development-scale numbers (8 GB laptop, short contexts) — not datacenter claims.
* First-request cold start is excluded via warmup; always compare warm measured trials.

---

# Next up — Profiling (learning first)

Before wiring Nsight / PyTorch Profiler into this lab, the next step is **learning the profiling tools** (timeline vs kernel metrics, how to read prefill vs decode, where TTFT shows up). Phase 6 will then profile the same HF / vLLM / Qasper setups with a consistent methodology.

---

# Environment

* Windows 11 + Docker Desktop (WSL2 / Linux engine)
* Python 3.12 (host venv for clients)
* NVIDIA RTX 4060 Laptop GPU (~8 GB)
* CUDA-enabled PyTorch + Hugging Face Transformers
* vLLM in Docker (`llm-lab-vllm:phase2`)

---

# Getting Started

## Host client

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Single-shot baselines

```powershell
python hf_baseline.py
# with vLLM server running:
python vllm_baseline.py
```

## Repeatable benchmark (Phase 3)

Start vLLM (for `--engine vllm`):

```powershell
docker run --rm -it --gpus all --ipc=host -p 8000:8000 `
  -v ${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface `
  llm-lab-vllm:phase2
```

Run harness (10 prompts × warmup × trials):

```powershell
# vLLM first (server must be up) — faster smoke of the harness
python run_benchmark.py --engine vllm --warmup 1 --trials 5 --prompts prompts/short.json

# Hugging Face (loads model once, then all prompts)
python run_benchmark.py --engine hf --warmup 1 --trials 5 --prompts prompts/short.json
```

Summarize and plot:

```powershell
python summarize.py results/bench_vllm_YYYYMMDD_HHMMSS.jsonl
python summarize.py results/bench_hf_YYYYMMDD_HHMMSS.jsonl

python plot_compare.py results/bench_hf_....jsonl results/bench_vllm_....jsonl `
  --out results/compare_hf_vllm.png
```

Example (this machine’s Phase 3 run):

```powershell
python summarize.py results\bench_vllm_20260728_191524.jsonl
python summarize.py results\bench_hf_20260728_191640.jsonl
python plot_compare.py results\bench_hf_20260728_191640.jsonl results\bench_vllm_20260728_191524.jsonl
```

Defaults: `--warmup 1`, `--trials 5`, `--max-new-tokens 64`.

## Qasper prefix-cache bench (Phases 4–5)

```powershell
python prepare_qasper_workload.py

python run_benchmark.py --engine vllm --warmup 0 --trials 1 `
  --prompts prompts/qasper_shared_1024.json

python run_benchmark.py --engine vllm --warmup 0 --trials 1 `
  --prompts prompts/qasper_control_1024.json

# rename/copy JSONLs to the names plot_qasper_prefix.py expects, then:
python plot_qasper_prefix.py
```

---

# Prompt sets

`prompts/short.json` — 10 inference-systems questions (similar length, fixed `max_new_tokens=64`):

* kv_cache_short, prefill_vs_decode, paged_attention, ttft_meaning
* continuous_batching, prefix_caching, gpu_memory_kv
* throughput_vs_latency, chunked_prefill, recompute_vs_transfer

`prompts/qasper_*_1024.json` — long-context shared-prefix vs control fixtures (Phase 4).

---

# Future Work

* Profiling methodology (Nsight Systems / Compute, PyTorch Profiler)
* Prompt-length sweeps (512 / 2048 token fixtures)
* Concurrency sweeps
* LMCache
* SGLang / TensorRT-LLM / ROCm

Each phase will introduce one new systems concept while keeping the benchmark methodology consistent.
