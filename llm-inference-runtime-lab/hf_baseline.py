"""Phase 1 Hugging Face baseline — single streamed request or run_once() for the harness."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Explain KV Cache in one short paragraph."
MAX_NEW_TOKENS = 64


def load_model(model_id: str = MODEL_ID):
    """Load tokenizer + model once; reuse across trials."""
    print(f"Loading model {model_id}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda",
    )
    load_s = time.perf_counter() - t0
    print(f"load_s={load_s:.2f}")
    print(f"device={model.device}")
    return tokenizer, model, load_s


def run_once(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    model_id: str = MODEL_ID,
    load_s: float | None = None,
) -> dict:
    """Stream one generation; return metrics dict (no file I/O)."""
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "streamer": streamer,
    }

    output_ids = None

    def run_generate():
        nonlocal output_ids
        output_ids = model.generate(**gen_kwargs)

    t0 = time.perf_counter()
    t_first = None
    thread = Thread(target=run_generate)
    thread.start()

    chunks = []
    for chunk in streamer:
        if t_first is None and chunk:
            t_first = time.perf_counter()
        chunks.append(chunk)

    thread.join()
    t_end = time.perf_counter()

    if t_first is None:
        raise RuntimeError("No streamed tokens — cannot compute TTFT")

    output_tokens = int(output_ids.shape[-1] - prompt_len)
    ttft_s = t_first - t0
    e2e_s = t_end - t0
    tok_per_s = output_tokens / (t_end - t_first)

    return {
        "engine": "huggingface",
        "model": model_id,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "load_s": load_s,
        "prompt_tokens": int(prompt_len),
        "output_tokens": output_tokens,
        "ttft_s": ttft_s,
        "e2e_s": e2e_s,
        "output_tok_per_s": tok_per_s,
        "output_text": "".join(chunks),
        "device": str(model.device),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    tokenizer, model, load_s = load_model(MODEL_ID)
    record = run_once(
        tokenizer,
        model,
        PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        model_id=MODEL_ID,
        load_s=load_s,
    )
    print(f"\n--- output ---\n{record['output_text']}\n")
    print(f"prompt_tokens={record['prompt_tokens']}")
    print(f"output_tokens={record['output_tokens']}")
    print(f"ttft_s={record['ttft_s']:.4f}")
    print(f"e2e_s={record['e2e_s']:.4f}")
    print(f"output_tok_per_s={record['output_tok_per_s']:.2f}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    to_save = {k: v for k, v in record.items() if k != "output_text"}
    out_path = results_dir / f"hf_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
