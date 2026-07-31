"""Build profiling/dashboard/data.json from nsys stats CSVs.

Usage (from llm-inference-runtime-lab/):
  python scripts/build_apc_dashboard_data.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
STATS = ROOT / "profiling" / "stats"
# GitHub Pages serves /docs from the repo root.
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


def pack(stem: str) -> dict:
    return {
        "stem": stem,
        "mem": mem_ops(stem),
        "kernels": top_kernels(stem),
        "api": top_api(stem),
        "signals": {
            "reshape_and_cache": find_kernel(stem, "reshape_and_cache"),
            "flash_fwd": find_kernel(stem, "flash_fwd_splitkv_kernel"),
        },
    }


def main() -> int:
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

    data = {
        "meta": {
            "title": "vLLM Automatic Prefix Cache — Nsight A/B",
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "gpu": "NVIDIA RTX 4060 Laptop (~8 GB)",
            "protocol": "Unrelated warmup → 2 shared-prefix Qasper prompts",
            "caveat": (
                "Whole-session nsys stats include model load + init + warmup + Q1 + Q2. "
                "Session HtoD totals look alike; APC shows up in the Events timeline "
                "(tiny H→D before kernels on a hit) and in client TTFT on Q2."
            ),
        },
        "client_ttft": {
            "source": "Phase 5 shared-prefix bench (cache ON); profile JSONL was not persisted",
            "series": [
                {"label": "Q1 (miss)", "ttft_s": 2.3},
                {"label": "Q2 (hit)", "ttft_s": 0.04},
            ],
            "note": (
                "Illustrates the prefix-hit TTFT drop with APC on. "
                "Warm ON/OFF profile client timings were not saved to disk."
            ),
        },
        "cache_off": pack("cache_off_warm"),
        "cache_on": pack("cache_on_warm"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
