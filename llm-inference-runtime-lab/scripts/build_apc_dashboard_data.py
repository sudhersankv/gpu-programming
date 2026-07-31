"""Build docs/data.json for the APC GitHub Pages dashboard.

Merges:
  - profiling/bench/warm_cache_ab.json  (client TTFT / E2E / tok/s)
  - profiling/stats/*_warm_*.csv       (whole-session nsys stats)
  - docs/data.json sqlite section       (optional; refreshed by query_nsys_sqlite.py)

Usage (from llm-inference-runtime-lab/):
  python scripts/build_apc_dashboard_data.py
  python scripts/query_nsys_sqlite.py   # refreshes sqlite:* from local .sqlite
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
STATS = ROOT / "profiling" / "stats"
BENCH = ROOT / "profiling" / "bench" / "warm_cache_ab.json"
OUT = REPO / "docs" / "data.json"


def short_kernel(name: str, n: int = 48) -> str:
    name = name.strip('"')
    if "vectorized_elementwise" in name and "FillFunctor<int>" in name:
        return "FillFunctor<int> (elementwise)"
    if "reshape_and_cache" in name:
        return "vllm::reshape_and_cache_flash"
    if "flash_fwd" in name:
        return "flash_fwd_splitkv"
    if name.startswith("triton_"):
        return name.split("(")[0][:n]
    if "ampere_bf16" in name:
        return name[:n] + ("…" if len(name) > n else "")
    if "cutlass::Kernel2" in name:
        m = re.search(r"cutlass_80_\w+", name)
        return (m.group(0) if m else "cutlass gemm")[:n]
    return (name[:n] + "…") if len(name) > n else name


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def mem_ops(stem: str) -> list[dict]:
    time_rows = read_csv(STATS / f"{stem}_cuda_gpu_mem_time_sum.csv")
    size_rows = read_csv(STATS / f"{stem}_cuda_gpu_mem_size_sum.csv")
    size_by = {r["Operation"]: r for r in size_rows}
    out: list[dict] = []
    for r in time_rows:
        op = r["Operation"]
        s = size_by.get(op, {})
        out.append(
            {
                "operation": op.replace("[CUDA ", "").replace("]", ""),
                "count": int(float(r["Count"])),
                "total_time_ms": round(float(r["Total Time (ns)"]) / 1e6, 3),
                "med_ns": float(r["Med (ns)"]),
                "total_mb": round(float(s.get("Total (MB)", 0) or 0), 3),
            }
        )
    return out


def top_kernels(stem: str, n: int = 10) -> list[dict]:
    rows = read_csv(STATS / f"{stem}_cuda_gpu_kern_sum.csv")
    out: list[dict] = []
    for r in rows[:n]:
        out.append(
            {
                "name": short_kernel(r["Name"]),
                "full_name": r["Name"].strip('"'),
                "time_pct": float(r["Time (%)"]),
                "total_time_ms": round(float(r["Total Time (ns)"]) / 1e6, 2),
                "instances": int(float(r["Instances"])),
            }
        )
    return out


def top_api(stem: str, n: int = 8) -> list[dict]:
    rows = read_csv(STATS / f"{stem}_cuda_api_sum.csv")
    out: list[dict] = []
    for r in rows[:n]:
        out.append(
            {
                "name": r["Name"],
                "time_pct": float(r["Time (%)"]),
                "total_time_ms": round(float(r["Total Time (ns)"]) / 1e6, 2),
                "num_calls": int(float(r["Num Calls"])),
            }
        )
    return out


def find_kernel(stem: str, substr: str) -> dict | None:
    rows = read_csv(STATS / f"{stem}_cuda_gpu_kern_sum.csv")
    for r in rows:
        if substr in r["Name"]:
            return {
                "name": short_kernel(r["Name"]),
                "instances": int(float(r["Instances"])),
                "total_time_ms": round(float(r["Total Time (ns)"]) / 1e6, 2),
            }
    return None


def pack_nsys(stem: str) -> dict:
    return {
        "stem": stem,
        "scope": "whole_session",
        "mem": mem_ops(stem),
        "kernels": top_kernels(stem),
        "api": top_api(stem),
        "signals": {
            "reshape_and_cache": find_kernel(stem, "reshape_and_cache"),
            "flash_fwd": find_kernel(stem, "flash_fwd_splitkv_kernel"),
        },
    }


def derive_client(bench: dict) -> dict:
    """Compute request-level deltas from raw warm A/B timings (seconds)."""
    on = bench["cache_on"]
    off = bench["cache_off"]
    on_r1 = on["request_1"]["ttft_s"]
    on_r2 = on["request_2"]["ttft_s"]
    off_r1 = off["request_1"]["ttft_s"]
    off_r2 = off["request_2"]["ttft_s"]

    saved_s = off_r2 - on_r2
    if off_r2 <= 0 or on_r2 <= 0:
        raise ValueError("TTFT values must be positive")
    reduction_pct = 100.0 * saved_s / off_r2
    speedup = off_r2 / on_r2

    return {
        "source": bench.get("source", ""),
        "experiment_id": bench.get("experiment_id", ""),
        "cache_on": {
            "request_1": {
                "ttft_ms": round(on_r1 * 1000, 1),
                "ttft_s": on_r1,
                "e2e_s": on["request_1"]["e2e_s"],
                "output_tok_per_s": on["request_1"]["output_tok_per_s"],
            },
            "request_2": {
                "ttft_ms": round(on_r2 * 1000, 1),
                "ttft_s": on_r2,
                "e2e_s": on["request_2"]["e2e_s"],
                "output_tok_per_s": on["request_2"]["output_tok_per_s"],
            },
        },
        "cache_off": {
            "request_1": {
                "ttft_ms": round(off_r1 * 1000, 1),
                "ttft_s": off_r1,
                "e2e_s": off["request_1"]["e2e_s"],
                "output_tok_per_s": off["request_1"]["output_tok_per_s"],
            },
            "request_2": {
                "ttft_ms": round(off_r2 * 1000, 1),
                "ttft_s": off_r2,
                "e2e_s": off["request_2"]["e2e_s"],
                "output_tok_per_s": off["request_2"]["output_tok_per_s"],
            },
        },
        "request_2": {
            "ttft_saved_ms": round(saved_s * 1000, 1),
            "ttft_reduction_pct": round(reduction_pct, 1),
            "ttft_speedup": round(speedup, 1),
            "off_ttft_ms": round(off_r2 * 1000, 1),
            "on_ttft_ms": round(on_r2 * 1000, 1),
        },
        "request_1_match": {
            "on_ttft_ms": round(on_r1 * 1000, 1),
            "off_ttft_ms": round(off_r1 * 1000, 1),
            "abs_delta_ms": round(abs(on_r1 - off_r1) * 1000, 1),
        },
    }


def main() -> int:
    if not BENCH.is_file():
        print(f"missing bench record: {BENCH}")
        return 1

    for stem in ("cache_off_warm", "cache_on_warm"):
        needed = [
            f"{stem}_cuda_gpu_kern_sum.csv",
            f"{stem}_cuda_gpu_mem_time_sum.csv",
            f"{stem}_cuda_gpu_mem_size_sum.csv",
            f"{stem}_cuda_api_sum.csv",
        ]
        missing = [p for p in needed if not (STATS / p).is_file()]
        if missing:
            print(f"missing CSVs under {STATS}: {missing}")
            print("Run: python scripts/export_nsys_stats.py")
            return 1

    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    client = derive_client(bench)

    # Preserve prior sqlite section if present (refreshed by query_nsys_sqlite.py).
    prior_sqlite = None
    if OUT.is_file():
        try:
            prior_sqlite = json.loads(OUT.read_text(encoding="utf-8")).get("sqlite")
        except json.JSONDecodeError:
            prior_sqlite = None

    data = {
        "meta": {
            "title": "Profiling vLLM Automatic Prefix Caching",
            "subtitle": (
                "A controlled cache ON/OFF experiment measuring repeated-prefix "
                "TTFT and tracing CPU/GPU execution with NVIDIA Nsight Systems."
            ),
            "model": bench["model"],
            "gpu": bench["gpu"],
            "engine": bench["engine"],
            "workload": bench["workload"],
            "protocol": "Unrelated warmup → shared-prefix Request 1 → shared-prefix Request 2",
            "chips": [
                bench["model"].removeprefix("Qwen/"),
                "NVIDIA RTX 4060 Laptop GPU",
                "vLLM",
                "Qasper · 1024-token shared paper prefix",
                "NVIDIA Nsight Systems",
            ],
            "repo_url": "https://github.com/sudhersankv/gpu-programming",
            "lab_readme_url": (
                "https://github.com/sudhersankv/gpu-programming/tree/main/"
                "llm-inference-runtime-lab"
            ),
            "harness_url": (
                "https://github.com/sudhersankv/gpu-programming/blob/main/"
                "llm-inference-runtime-lab/scripts/profile_prefix_cache.py"
            ),
            "scripts_url": (
                "https://github.com/sudhersankv/gpu-programming/tree/main/"
                "llm-inference-runtime-lab/scripts"
            ),
        },
        "client": client,
        "nsight_findings": {
            "bullets": [
                "Cache OFF executed the full prefill transformer path for Request 2.",
                "Cache ON reused GPU-resident KV blocks and computed only the uncached suffix.",
                "Decode CUDA Graph execution remained similar in both runs.",
                "Output-token throughput stayed approximately unchanged (~50–53 tok/s).",
                (
                    "Small Host→Device copies near request execution are consistent with "
                    "input and runtime metadata transfers — not evidence that every copy "
                    "is specifically a KV block-table update."
                ),
                (
                    "Cached KV tensors were not restored Host→Device; reused KV remained "
                    "resident in GPU memory."
                ),
            ],
            "kernels": [
                "Tensor Core BF16 GEMMs (ampere_bf16 / Cutlass)",
                "FlashAttention forward (flash_fwd_splitkv)",
                "vllm::reshape_and_cache_flash",
                "Fused Triton SiLU and normalization kernels",
                "CUDA Graph replay during decoding",
            ],
        },
        "cache_off": pack_nsys("cache_off_warm"),
        "cache_on": pack_nsys("cache_on_warm"),
    }
    if prior_sqlite:
        data["sqlite"] = prior_sqlite

    r2 = client["request_2"]
    print(
        f"R2 TTFT  OFF={r2['off_ttft_ms']} ms  ON={r2['on_ttft_ms']} ms  "
        f"saved={r2['ttft_saved_ms']} ms  reduction={r2['ttft_reduction_pct']}%  "
        f"speedup={r2['ttft_speedup']}x"
    )
    expected = (339.3, 85.7, 7.0)
    got = (r2["ttft_saved_ms"], r2["ttft_reduction_pct"], r2["ttft_speedup"])
    if got != expected:
        print(f"WARNING: derived {got} != expected {expected}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
