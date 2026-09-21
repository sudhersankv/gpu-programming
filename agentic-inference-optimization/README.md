# Agentic Inference Optimization

Experimental system for observing real LLM inference on GPUs, diagnosing performance behavior, and later using agents to propose, test, and evaluate optimizations.

All work for this project lives in this folder.

The long-term loop is:

```text
Run Inference
     ↓
Observe
     ↓
Diagnose
     ↓
Propose Optimization
     ↓
Experiment
     ↓
Measure Again
```

That full system is **not built yet**.

---

## Current phase

**Phase 1 — Exploration and System Understanding**

No substantial implementation yet. Before designing the Observation Deck, we need to understand the path from a GPU instance to generated tokens:

```text
RunPod GPU
     ↓
NVIDIA GPU / Driver
     ↓
CUDA Runtime
     ↓
PyTorch
     ↓
vLLM
     ↓
Model Loading
     ↓
Inference Request
     ↓
Scheduling / Batching
     ↓
Model Execution
     ↓
CUDA Operations / Kernels
     ↓
GPU Execution
     ↓
Generated Tokens
```

### What exists here

* This README

### What is not here yet

* Observation Deck
* Benchmarking framework
* Agents
* Bottleneck judge
* Speculative project architecture

---

## Goal 1 (later)

Inference Pipeline + Observation Deck, built incrementally after the stack is understood:

```text
Workload Definition
        ↓
Inference Pipeline
        ↓
Benchmarking
        ↓
GPU Telemetry
        ↓
GPU Profiling
        ↓
Structured Observations
        ↓
Observation Deck
```

Do not treat this as implemented.

---

## Relationship to the rest of the repo

[`llm-inference-runtime-lab/`](../llm-inference-runtime-lab/) and [`inference-systems-notebooks/`](../inference-systems-notebooks/) are prior learning on a laptop (Hugging Face, vLLM, Nsight, KV-cache concepts). They are background, not this system.

New code, experiments, RunPod notes, and public GitHub documentation for this project go **only** in `agentic-inference-optimization/`.

---

## Exploration checklist

Work through these before implementation:

1. GPU environment (and choose one GPU type)
2. Initial model (and its architecture)
3. Model loading and GPU memory layout
4. One inference request end-to-end
5. Prefill, decode, and KV cache
6. How vLLM schedules and executes the workload
7. GPU execution at a useful conceptual level
8. A manual RunPod run of this pipeline
