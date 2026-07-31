"""Export nsys stats CSVs for cache_on_warm / cache_off_warm via Docker.

Usage (from llm-inference-runtime-lab/):
  python scripts/export_nsys_stats.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "profiling" / "reports"
STATS = ROOT / "profiling" / "stats"
IMAGE = "llm-lab-vllm:profile"
STEMS = ("cache_off_warm", "cache_on_warm")
REPORTS_LIST = (
    "cuda_gpu_kern_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
    "cuda_api_sum",
)


def export_stem(stem: str) -> None:
    rep = REPORTS / f"{stem}.nsys-rep"
    if not rep.is_file():
        raise FileNotFoundError(rep)

    fmt = ",".join(["csv"] * len(REPORTS_LIST))
    outs = ",".join([f"/stats/{stem}"] * len(REPORTS_LIST))
    report_args: list[str] = []
    for name in REPORTS_LIST:
        report_args.extend(["--report", name])

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{REPORTS}:/reports",
        "-v",
        f"{STATS}:/stats",
        "--entrypoint",
        "nsys",
        IMAGE,
        "stats",
        "--force-export=true",
        *report_args,
        "--format",
        fmt,
        "--output",
        outs,
        f"/reports/{stem}.nsys-rep",
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    STATS.mkdir(parents=True, exist_ok=True)
    try:
        for stem in STEMS:
            export_stem(stem)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"CSVs in {STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
