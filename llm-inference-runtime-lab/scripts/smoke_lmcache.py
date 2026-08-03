"""
Phase 7 — LMCache smoke test (correctness, not a perf study).

Prereq: container llm-lab-lmcache running (scripts/run_lmcache_container.ps1).

Sends two Qasper shared-prefix prompts (same paper, different questions):
  Request 1 → expect LMCache STORE in server logs
  Request 2 → expect RETRIEVE / hit tokens in server logs

Usage (from llm-inference-runtime-lab/):
  python scripts/smoke_lmcache.py
  python scripts/smoke_lmcache.py --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vllm_baseline import MODEL_ID, run_once  # noqa: E402

SHARED = ROOT / "prompts" / "qasper_shared_1024.json"
HEALTH_TIMEOUT_S = 600
HEALTH_POLL_S = 2.0


def wait_for_health(base: str, timeout_s: float) -> None:
    health = base.rstrip("/").removesuffix("/v1") + "/health"
    deadline = time.monotonic() + timeout_s
    print(f"waiting for {health} ...", flush=True)
    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                r = client.get(health)
                if r.status_code == 200:
                    print("healthy", flush=True)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(HEALTH_POLL_S)
    raise TimeoutError(f"server not healthy within {timeout_s}s")


def load_two_shared() -> list[dict]:
    prompts = json.loads(SHARED.read_text(encoding="utf-8"))
    if len(prompts) < 2:
        raise SystemExit(f"need ≥2 prompts in {SHARED}")
    return prompts[:2]


def main() -> int:
    p = argparse.ArgumentParser(description="LMCache store/retrieve smoke test")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default=MODEL_ID)
    args = p.parse_args()

    if not SHARED.is_file():
        print(f"missing {SHARED} — run prepare_qasper_workload.py", file=sys.stderr)
        return 1

    wait_for_health(args.base_url, HEALTH_TIMEOUT_S)
    cases = load_two_shared()

    print("\n=== LMCache smoke: shared-prefix A1 then A2 ===\n")
    print("Watch server logs in parallel:")
    print("  docker logs -f llm-lab-lmcache")
    print("Expect roughly:")
    print("  A1 → LMCache INFO: Stored ... tokens")
    print("  A2 → hit tokens / Retrieved ... tokens\n")

    results = []
    with httpx.Client(timeout=None) as client:
        for i, case in enumerate(cases, start=1):
            pid = case["prompt_id"]
            text = case["text"]
            max_new = int(case.get("max_new_tokens", 64))
            print(f"--- Request {i}: {pid} (max_new_tokens={max_new}) ---", flush=True)
            m = run_once(
                text,
                max_new_tokens=max_new,
                model_id=args.model,
                base_url=args.base_url,
                client=client,
            )
            m["prompt_id"] = pid
            m["request_index"] = i
            m["engine"] = "vllm+lmcache"
            results.append(m)
            print(
                f"  ttft_s={m['ttft_s']:.4f}  e2e_s={m['e2e_s']:.4f}  "
                f"tok/s={m['output_tok_per_s']:.1f}  "
                f"prompt_tokens={m.get('prompt_tokens')}",
                flush=True,
            )

    r1, r2 = results[0], results[1]
    print("\n=== client summary ===")
    print(f"R1 TTFT={r1['ttft_s']:.4f}s   R2 TTFT={r2['ttft_s']:.4f}s")
    if r2["ttft_s"] < r1["ttft_s"]:
        print("R2 faster than R1 (consistent with prefix reuse; confirm in LMCache logs).")
    else:
        print(
            "R2 was not faster than R1 — still check logs; "
            "APC + LMCache interaction or cold graphs can blur TTFT on a tiny smoke."
        )

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "lmcache_smoke.json"
    slim = [{k: v for k, v in r.items() if k != "output_text"} for r in results]
    out.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    print(f"saved={out}")

    print("\n=== pass criteria ===")
    print("PASS if docker logs show STORE on R1 and RETRIEVE/hit on R2.")
    print("Then stop the container and move on to SGLang — no deep LMCache bench needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
