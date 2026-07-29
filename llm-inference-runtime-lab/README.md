# LLM Inference Runtime Lab

A hands-on exploration of modern LLM inference systems, benchmarking, profiling, and runtime optimization.

The goal of this project is to understand how different inference engines execute transformer models, identify performance bottlenecks, and evaluate techniques such as shared-prefix caching, KV-cache management, and continuous batching.

This project is being built incrementally. Each phase introduces a new inference system concept and benchmarks it before moving on to more advanced runtime optimizations.

---

## Project Roadmap

* [x] **Phase 1** — Hugging Face baseline
* [x] **Phase 2** — vLLM baseline (Docker)
* [x] **Phase 3** — Benchmark framework
* [ ] Phase 4 — Shared-prefix workload generation
* [ ] Phase 5 — vLLM Prefix Caching
* [ ] Phase 6 — LMCache integration
* [ ] Phase 7 — Profiling (Nsight / PyTorch Profiler)
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

---

# Prompt set

`prompts/short.json` — 10 inference-systems questions (similar length, fixed `max_new_tokens=64`):

* kv_cache_short, prefill_vs_decode, paged_attention, ttft_meaning
* continuous_batching, prefix_caching, gpu_memory_kv
* throughput_vs_latency, chunked_prefill, recompute_vs_transfer

---

# Future Work

* Prompt-length sweeps (exact token counts)
* Concurrency sweeps
* Shared-prefix workloads
* vLLM automatic prefix caching
* LMCache
* Nsight / PyTorch profiling
* SGLang / TensorRT-LLM / ROCm

Each phase will introduce one new systems concept while keeping the benchmark methodology consistent.
