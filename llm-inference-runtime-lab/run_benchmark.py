"""Phase 3 harness: choose engine → warmup → N trials → one JSONL file."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import httpx

from hf_baseline import MODEL_ID, load_model, run_once as hf_run_once
from vllm_baseline import BASE_URL, run_once as vllm_run_once


def load_prompts(path: Path) -> list[dict]:
    prompts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts:
        raise SystemExit(f"No prompts in {path}")
    return prompts


def append_record(fp, record: dict) -> None:
    # Drop bulky text from JSONL by default
    row = {k: v for k, v in record.items() if k != "output_text"}
    fp.write(json.dumps(row) + "\n")
    fp.flush()


def run_hf(args, prompts: list[dict], out_path: Path, run_id: str) -> None:
    tokenizer, model, load_s = load_model(args.model)
    with out_path.open("w", encoding="utf-8") as fp:
        for case in prompts:
            prompt_id = case["prompt_id"]
            text = case["text"]
            max_new = int(case.get("max_new_tokens", args.max_new_tokens))
            print(f"\n=== {prompt_id} (hf) ===")

            for w in range(args.warmup):
                print(f"  warmup {w + 1}/{args.warmup}")
                metrics = hf_run_once(
                    tokenizer,
                    model,
                    text,
                    max_new_tokens=max_new,
                    model_id=args.model,
                    load_s=load_s,
                )
                metrics.update(
                    {
                        "run_id": run_id,
                        "prompt_id": prompt_id,
                        "trial": w,
                        "is_warmup": True,
                    }
                )
                append_record(fp, metrics)

            for t in range(1, args.trials + 1):
                print(f"  trial {t}/{args.trials}")
                metrics = hf_run_once(
                    tokenizer,
                    model,
                    text,
                    max_new_tokens=max_new,
                    model_id=args.model,
                    load_s=load_s,
                )
                metrics.update(
                    {
                        "run_id": run_id,
                        "prompt_id": prompt_id,
                        "trial": t,
                        "is_warmup": False,
                    }
                )
                append_record(fp, metrics)
                print(
                    f"    ttft={metrics['ttft_s']:.4f}s "
                    f"e2e={metrics['e2e_s']:.4f}s "
                    f"tok/s={metrics['output_tok_per_s']:.1f}"
                )


def run_vllm(args, prompts: list[dict], out_path: Path, run_id: str) -> None:
    with httpx.Client(timeout=None) as client:
        # Fail fast if server is down
        r = client.get(f"{args.base_url}/models")
        r.raise_for_status()

        with out_path.open("w", encoding="utf-8") as fp:
            for case in prompts:
                prompt_id = case["prompt_id"]
                text = case["text"]
                max_new = int(case.get("max_new_tokens", args.max_new_tokens))
                print(f"\n=== {prompt_id} (vllm) ===")

                for w in range(args.warmup):
                    print(f"  warmup {w + 1}/{args.warmup}")
                    metrics = vllm_run_once(
                        text,
                        max_new_tokens=max_new,
                        model_id=args.model,
                        base_url=args.base_url,
                        client=client,
                    )
                    metrics.update(
                        {
                            "run_id": run_id,
                            "prompt_id": prompt_id,
                            "trial": w,
                            "is_warmup": True,
                        }
                    )
                    append_record(fp, metrics)

                for t in range(1, args.trials + 1):
                    print(f"  trial {t}/{args.trials}")
                    metrics = vllm_run_once(
                        text,
                        max_new_tokens=max_new,
                        model_id=args.model,
                        base_url=args.base_url,
                        client=client,
                    )
                    metrics.update(
                        {
                            "run_id": run_id,
                            "prompt_id": prompt_id,
                            "trial": t,
                            "is_warmup": False,
                        }
                    )
                    append_record(fp, metrics)
                    print(
                        f"    ttft={metrics['ttft_s']:.4f}s "
                        f"e2e={metrics['e2e_s']:.4f}s "
                        f"tok/s={metrics['output_tok_per_s']:.1f}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeatable HF/vLLM benchmark harness")
    parser.add_argument("--engine", choices=["hf", "vllm"], required=True)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("prompts/short.json"),
        help="JSON list of {prompt_id, text, max_new_tokens?}",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default: results/bench_<engine>_<timestamp>.jsonl)",
    )
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = args.out or results_dir / f"bench_{args.engine}_{run_id}.jsonl"

    print(
        f"engine={args.engine} prompts={len(prompts)} "
        f"warmup={args.warmup} trials={args.trials} out={out_path}"
    )

    if args.engine == "hf":
        run_hf(args, prompts, out_path, run_id)
    else:
        run_vllm(args, prompts, out_path, run_id)

    print(f"\nsaved={out_path}")
    print(f"next: python summarize.py {out_path}")


if __name__ == "__main__":
    main()
