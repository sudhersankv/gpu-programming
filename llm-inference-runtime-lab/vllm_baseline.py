"""Phase 2 vLLM baseline — streamed OpenAI API request or run_once() for the harness."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000/v1"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Explain KV Cache in one short paragraph."
MAX_NEW_TOKENS = 64


def run_once(
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    model_id: str = MODEL_ID,
    base_url: str = BASE_URL,
    client: httpx.Client | None = None,
) -> dict:
    """Stream one chat completion; return metrics dict (no file I/O)."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=None)

    chunks: list[str] = []
    output_tokens = None
    prompt_tokens = None
    t_first = None

    try:
        t0 = time.perf_counter()
        with client.stream("POST", f"{base_url}/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: ") :]
                else:
                    continue

                if data == "[DONE]":
                    break

                event = json.loads(data)
                usage = event.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)

                choice0 = (event.get("choices") or [{}])[0]
                delta = (choice0.get("delta") or {}).get("content")
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    chunks.append(delta)
        t_end = time.perf_counter()
    finally:
        if own_client:
            client.close()

    if t_first is None:
        raise RuntimeError("No streamed tokens — cannot compute TTFT")
    if output_tokens is None:
        raise RuntimeError(
            "No usage.completion_tokens in stream — enable stream_options.include_usage"
        )

    ttft_s = t_first - t0
    e2e_s = t_end - t0
    tok_per_s = output_tokens / (t_end - t_first)

    return {
        "engine": "vllm",
        "model": model_id,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "load_s": None,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_s": ttft_s,
        "e2e_s": e2e_s,
        "output_tok_per_s": tok_per_s,
        "output_text": "".join(chunks),
        "device": "vllm-docker",
        "base_url": base_url,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    record = run_once(PROMPT, max_new_tokens=MAX_NEW_TOKENS)
    print(f"\n--- output ---\n{record['output_text']}\n")
    print(f"prompt_tokens={record['prompt_tokens']}")
    print(f"output_tokens={record['output_tokens']}")
    print(f"ttft_s={record['ttft_s']:.4f}")
    print(f"e2e_s={record['e2e_s']:.4f}")
    print(f"output_tok_per_s={record['output_tok_per_s']:.2f}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    to_save = {k: v for k, v in record.items() if k != "output_text"}
    out_path = results_dir / f"vllm_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
