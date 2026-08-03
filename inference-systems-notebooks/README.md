# Inference Systems Notebooks

A series of **standalone Google Colab notebooks** for learning how modern LLM inference systems work — KV caches, paging, prefix reuse, scheduling, attention IO, parallelism, and how those ideas show up in engines like vLLM and SGLang.

Notebooks are added over time. Until a file appears in this folder, treat that topic as **planned**.

---

## Who this is for

* Engineers who want intuition for **serving** LLMs, not only training
* People reading vLLM / SGLang / TensorRT-LLM docs who want a **toy model** of the underlying ideas first
* Students building a portfolio of small, reproducible systems experiments

You do **not** need a local GPU. Most notebooks are meant to run on **Colab** (CPU is fine for the toy parts).

---

## How this relates to the rest of the repo

| Track | What you do |
|-------|-------------|
| [`llm-inference-runtime-lab/`](../llm-inference-runtime-lab/) | Run real servers in Docker, measure TTFT, profile with Nsight |
| **This folder** | Implement small simulations in Colab, then map them to production systems |

They complement each other. Start either place; notebooks do not depend on the lab containers.

---

## How to use a notebook

1. Open the `.ipynb` from this folder (or its Colab link, when listed below).
2. Runtime → **Run all** (or step through cells).
3. Each notebook installs its own dependencies in the first cells — use a **fresh** Colab runtime per notebook.
4. Read the closing **questions / experiment** section; try the suggested tweak.

No notebook imports another. You can jump to any topic without finishing earlier ones (recommended order is still 01 → 16).

---

## Design principles

Every notebook in this track should:

1. Start with a **minimal toy implementation** (NumPy / small PyTorch)
2. Measure or visualize something concrete (memory, latency, tree shape, queue wait)
3. Connect the toy to at least one **real system** idea (vLLM APC, RadixAttention, LMCache, FlashAttention, …)
4. End with short **questions**, **observations**, and **one small experiment**
5. Avoid pulling in full production stacks unless that notebook’s topic requires it

---

## Curriculum

### Phase A — Core execution

| # | Topic | You will learn |
|---|--------|----------------|
| [01](#01--prefill-decode-and-kv-cache) | Prefill, decode, KV cache | Why decode without a KV cache repeats work; how caching changes cost with length |
| [02](#02--kv-cache-memory-scaling) | KV memory scaling | How layers, heads, sequence length, batch, and dtype drive GPU memory |

### Phase B — Memory management and reuse

| # | Topic | You will learn |
|---|--------|----------------|
| [03](#03--paged-kv-memory) | Paged KV | Why contiguous KV wastes memory; block tables and fragmentation |
| [04](#04--automatic-prefix-caching) | Automatic prefix caching | Block hashing, shared prefixes, miss on first mismatch |
| [05](#05--radixattention) | RadixAttention | Tree-structured prefix sharing (SGLang-style intuition) |
| [06](#06--lmcache-and-memory-tiers) | LMCache / tiers | GPU vs CPU hit vs recompute; when offload helps |

### Phase C — Scheduling and kernels

| # | Topic | You will learn |
|---|--------|----------------|
| [07](#07--continuous-batching) | Continuous batching | Why static batches hurt TTFT and utilization |
| [08](#08--inference-scheduling) | Scheduling policies | FIFO vs locality-aware policies; fairness tradeoffs |
| [09](#09--flashattention) | FlashAttention | Attention IO; tiled / online softmax *idea* (not a CUDA kernel clone) |
| [10](#10--chunked-prefill) | Chunked prefill | How long prefills can starve decode; chunking + interleave |

### Phase D — Advanced serving ideas

| # | Topic | You will learn |
|---|--------|----------------|
| [11](#11--speculative-decoding) | Speculative decoding | Draft / verify; speedup vs acceptance rate |
| [12](#12--tensor-parallelism) | Tensor parallelism | Shard matmuls; all-reduce; when TP does not pay off |
| [13](#13--pipeline-parallelism) | Pipeline parallelism | Bubbles, microbatches, latency vs throughput |
| [14](#14--quantization-for-inference) | Quantization | Weight vs KV quant; bandwidth vs quality |
| [15](#15--moe-inference-and-routing) | MoE inference | Routing, imbalance, capacity limits |

### Capstone

| # | Topic | You will learn |
|---|--------|----------------|
| [16](#16--runtime-architecture-comparison) | Runtime comparison | Map concepts across vLLM, SGLang, TensorRT-LLM, MAX |

---

## Status

| File | Status |
|------|--------|
| `01_prefill_decode_and_kv_cache.ipynb` | planned |
| `02_kv_cache_memory_scaling.ipynb` | planned |
| `03_paged_kv_memory.ipynb` | planned |
| `04_automatic_prefix_caching.ipynb` | planned |
| `05_radix_attention.ipynb` | planned |
| `06_lmcache_memory_tiers.ipynb` | planned |
| `07_continuous_batching.ipynb` | planned |
| `08_inference_scheduling.ipynb` | planned |
| `09_flash_attention.ipynb` | planned |
| `10_chunked_prefill.ipynb` | planned |
| `11_speculative_decoding.ipynb` | planned |
| `12_tensor_parallelism.ipynb` | planned |
| `13_pipeline_parallelism.ipynb` | planned |
| `14_quantization_for_inference.ipynb` | planned |
| `15_moe_inference_and_routing.ipynb` | planned |
| `16_runtime_architecture_comparison.ipynb` | planned |

When a notebook is published, its status becomes **available** and a Colab badge/link can be added next to the title.

---

## Notebook outlines

### 01 — Prefill, decode, and KV cache

Follow a prompt through prefill → first token → autoregressive decode. Implement a tiny attention path and compare decode **with** and **without** KV caching: repeated matmuls, and how latency grows with sequence length.

### 02 — KV-cache memory scaling

Derive KV memory from layers, KV heads, head dimension, sequence length, batch size, and dtype. Plug in numbers for familiar models and compare BF16 vs lower-precision storage assumptions.

### 03 — Paged KV memory

Build a small block allocator. Contrast contiguous allocation (fragmentation) with fixed-size blocks and block tables under several concurrent requests of different lengths.

### 04 — Automatic prefix caching

Implement block hashing of the form `hash(previous_block_hash, tokens)`. See exact prefix hits, the first mismatch, how block size sets reuse granularity, and simple eviction.

### 05 — RadixAttention

Implement a compressed radix tree: insert, longest-prefix match, split, branch, leaf eviction while keeping shared parents. Visualize the tree after each update.

### 06 — LMCache and memory tiers

Simulate GPU → CPU → recompute. Compare GPU hit, CPU hit (restore), and full miss. Track LRU eviction and the idea that **metadata** can live separately from where the KV tensors sit.

### 07 — Continuous batching

Simulate arrivals over time. Compare static batching, serial execution, and continuous batching on waiting time, utilization, TTFT, and throughput.

### 08 — Inference scheduling

Compare FIFO, shortest-remaining, decode-first, prefill-first, prefix-aware, and fairness-aware policies. Show a case where optimizing for cache locality worsens fairness.

### 09 — FlashAttention

Start from `QKᵀ → softmax → V`. Then simulate tiled / online softmax and reason about **memory traffic**, not a bit-exact production kernel.

### 10 — Chunked prefill

Show a long prefill blocking decode. Compare one-shot prefill vs chunked prefill with decode interleaved between chunks.

### 11 — Speculative decoding

Draft model proposes tokens; target verifies. Measure accepted vs rejected tokens and expected speedup under different acceptance rates.

### 12 — Tensor parallelism

Shard a linear layer across fake devices (column / row parallel). Account for all-reduce cost and when a small model gains little from TP.

### 13 — Pipeline parallelism

Place layers on different devices. Visualize pipeline bubbles, microbatches, and the latency/throughput tradeoff (including prefill/decode quirks).

### 14 — Quantization for inference

Compare FP32, FP16/BF16, INT8, INT4 for weights and optionally KV. Look at memory, bandwidth, dequant overhead, and quality tradeoffs.

### 15 — MoE inference and routing

Simulate router scores, top-k expert choice, token imbalance, expert capacity, and communication cost.

### 16 — Runtime architecture comparison

Map earlier concepts onto real stacks:

| Concept | vLLM | SGLang | TensorRT-LLM | MAX |
|---------|------|--------|--------------|-----|
| KV allocation | Paged KV | Paged / token pool | Runtime-managed | Runtime / compiler managed |
| Prefix reuse | Automatic prefix caching | RadixAttention | Prefix-caching features | Runtime dependent |
| Scheduling | Continuous batching | Prefix-aware serving | In-flight batching | MAX serving runtime |
| Kernels | CUDA / Triton | CUDA / Triton | NVIDIA-optimized | Mojo / MAX kernels |
| Typical strength | General-purpose serving | Prefix-heavy / structured workloads | Deep NVIDIA integration | Compiler + kernel stack |

---

## Contributing a notebook

If you are following or extending this track:

1. Keep the notebook **self-contained** (see [Design principles](#design-principles)).
2. Use the filenames in the [Status](#status) table.
3. Prefer Colab-friendly deps (`numpy`, `matplotlib`, small `torch` snippets).
4. Open a PR that adds the `.ipynb` and sets its status to **available**.

---

## License / context

Part of [gpu-programming](https://github.com/sudhersankv/gpu-programming). For hands-on Docker serving and Nsight profiling of prefix caching, see the [LLM Inference Runtime Lab](../llm-inference-runtime-lab/).
