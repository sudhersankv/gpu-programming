# Inference Systems Notebooks

Standalone Colab notebooks for learning LLM **inference systems** concepts — from prefill/decode and KV cache through scheduling, parallelism, and runtime comparison.

This track runs **in parallel** with [`llm-inference-runtime-lab/`](../llm-inference-runtime-lab/) (Docker / vLLM / Nsight). The lab measures real servers; these notebooks build **toy implementations** first, then connect to production runtimes.

Upload each `.ipynb` here as you finish it in Colab. Keep every notebook self-contained.

---

## Rules for every notebook

* Use its **own** Colab runtime
* Install **only** what that notebook needs
* Do **not** import code from another notebook
* Do **not** assume cells from another notebook were run
* Avoid giant production frameworks unless the concept specifically requires them
* Start with a **toy implementation**
* Then connect the toy idea to **vLLM / SGLang / TensorRT-LLM / LMCache / MAX** where useful
* End with **questions**, **observations**, and **one small experiment**

---

## Roadmap

```text
inference-systems-notebooks/
├── README.md                          ← this file
├── 01_prefill_decode_and_kv_cache.ipynb
├── 02_kv_cache_memory_scaling.ipynb
├── 03_paged_kv_memory.ipynb
├── 04_automatic_prefix_caching.ipynb
├── 05_radix_attention.ipynb
├── 06_lmcache_memory_tiers.ipynb
├── 07_continuous_batching.ipynb
├── 08_inference_scheduling.ipynb
├── 09_flash_attention.ipynb
├── 10_chunked_prefill.ipynb
├── 11_speculative_decoding.ipynb
├── 12_tensor_parallelism.ipynb
├── 13_pipeline_parallelism.ipynb
├── 14_quantization_for_inference.ipynb
├── 15_moe_inference_and_routing.ipynb
└── 16_runtime_architecture_comparison.ipynb
```

| # | Notebook | Status |
|---|----------|--------|
| 01 | Prefill, decode, and KV cache | planned |
| 02 | KV-cache memory scaling | planned |
| 03 | Paged KV memory | planned |
| 04 | Automatic prefix caching | planned |
| 05 | RadixAttention | planned |
| 06 | LMCache and memory tiers | planned |
| 07 | Continuous batching | planned |
| 08 | Inference scheduling | planned |
| 09 | FlashAttention | planned |
| 10 | Chunked prefill | planned |
| 11 | Speculative decoding | planned |
| 12 | Tensor parallelism | planned |
| 13 | Pipeline parallelism | planned |
| 14 | Quantization for inference | planned |
| 15 | MoE inference and routing | planned |
| 16 | Runtime architecture comparison | planned |

Mark a row **done** when the notebook is uploaded.

---

## Phase A — Core inference execution

### 01 — Prefill, decode, and KV cache

**Understand:** prompt → prefill → first token → autoregressive decode.

**Build:** tiny attention; compare decode **without** vs **with** KV caching (repeated ops, latency vs sequence length).

### 02 — KV-cache memory scaling

**Explore:** layers, KV heads, head dim, sequence length, batch size, dtype.

**Calculate:** KV memory for real models; compare BF16 / FP8 / INT8-style storage assumptions.

---

## Phase B — KV memory management and reuse

### 03 — Paged KV memory

**Build:** toy block allocator.

**Simulate:** contiguous allocation, fragmentation, fixed-size blocks, block tables, alloc/free, multi-request different lengths.

### 04 — Automatic prefix caching

**Implement:** `hash(previous_block_hash, current_block_tokens)`.

**Explore:** exact shared prefixes, first mismatch, block-aligned reuse, block size vs reuse granularity, eviction.

### 05 — RadixAttention

**Build:** compressed radix tree — insert, longest-prefix lookup, split, branch, leaf eviction, shared-parent preservation; visualize after each op.

### 06 — LMCache and memory tiers

**Build:** GPU cache → CPU cache → recompute.

**Compare:** GPU hit, CPU hit, miss; restore vs recompute; LRU; metadata on CPU while KV location changes.

---

## Phase C — Scheduling and serving

### 07 — Continuous batching

**Simulate:** requests arriving at different times.

**Compare:** static batching, one-at-a-time, continuous batching (wait time, GPU util, TTFT, throughput).

### 08 — Inference scheduling

**Compare:** FIFO, SRPT, decode-first, prefill-first, prefix-aware, fairness-aware.

**Show:** why better cache locality can hurt fairness.

### 09 — FlashAttention

**Start:** `QKᵀ → softmax → × V`.

**Then:** tiled / online-softmax simulation; measure memory. Goal = **IO reduction intuition**, not a production CUDA kernel.

### 10 — Chunked prefill

**Show:** long prefill blocking decode.

**Compare:** full prefill vs chunked prefill + decode interleaving.

---

## Phase D — Advanced decoding

### 11 — Speculative decoding

**Simulate:** draft proposes → target verifies → accept/reject → expected speedup vs acceptance rate.

### 12 — Tensor parallelism

**Split:** linear layer across simulated devices (column/row parallel, all-reduce, compute vs communication, small-model limits).

### 13 — Pipeline parallelism

**Split:** layers across devices; bubbles, microbatches, latency vs throughput, prefill/decode complications.

### 14 — Quantization for inference

**Compare:** FP32, FP16/BF16, INT8, INT4 — model/bandwidth memory, dequant overhead, quality, weight-only vs KV-cache quant.

### 15 — MoE inference and routing

**Simulate:** router scores, top-k experts, imbalance, capacity, communication.

---

## Final notebook

### 16 — Runtime architecture comparison

Map concepts to real systems:

| Concept | vLLM | SGLang | TensorRT-LLM | MAX |
|---------|------|--------|--------------|-----|
| KV allocation | Paged KV | Paged / token pool | Runtime-managed | Runtime / compiler managed |
| Prefix reuse | APC | RadixAttention | Prefix caching features | Runtime dependent |
| Scheduling | Continuous batching | Prefix-aware runtime | In-flight batching | MAX serving runtime |
| Kernels | CUDA / Triton | CUDA / Triton | NVIDIA-optimized | Mojo / MAX kernels |
| Main strength | General serving | Structured / prefix-heavy | NVIDIA specialization | Compiler + kernel integration |

---

## How to add a finished notebook

1. Download `.ipynb` from Colab (File → Download → `.ipynb`).
2. Place it in this folder with the matching name above.
3. Flip its status row to **done**.
4. Commit and push.

Optional: add a one-line “Colab link” under that notebook’s section once published.
