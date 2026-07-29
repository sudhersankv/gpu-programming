"""Compute p50/p90/p95/p99 (and mean) from a benchmark JSONL file."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


METRICS = ("ttft_s", "e2e_s", "output_tok_per_s")


def percentile(sorted_vals: list[float], p: float) -> float:
    """Nearest-rank percentile; p in [0, 100]."""
    if not sorted_vals:
        raise ValueError("empty values")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def load_measured(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("is_warmup"):
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"No measured (non-warmup) rows in {path}")
    return rows


def summarize(rows: list[dict]) -> dict:
    # group by engine + prompt_id
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row.get("engine", "?"), row.get("prompt_id", "?"))
        groups[key].append(row)

    out: dict = {"n_measured": len(rows), "groups": []}
    for (engine, prompt_id), items in sorted(groups.items()):
        entry = {
            "engine": engine,
            "prompt_id": prompt_id,
            "n": len(items),
            "metrics": {},
        }
        for metric in METRICS:
            vals = sorted(float(r[metric]) for r in items)
            entry["metrics"][metric] = {
                "mean": sum(vals) / len(vals),
                "p50": percentile(vals, 50),
                "p90": percentile(vals, 90),
                "p95": percentile(vals, 95),
                "p99": percentile(vals, 99),
                "min": vals[0],
                "max": vals[-1],
            }
        out["groups"].append(entry)
    return out


def print_summary(summary: dict) -> None:
    for g in summary["groups"]:
        print(f"\n{g['engine']} / {g['prompt_id']}  (n={g['n']})")
        for metric, stats in g["metrics"].items():
            print(
                f"  {metric:18s} "
                f"mean={stats['mean']:.4f}  "
                f"p50={stats['p50']:.4f}  "
                f"p90={stats['p90']:.4f}  "
                f"p95={stats['p95']:.4f}  "
                f"p99={stats['p99']:.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, help="bench_*.jsonl from run_benchmark.py")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write summary JSON (default: <jsonl>.summary.json)",
    )
    args = parser.parse_args()

    rows = load_measured(args.jsonl)
    summary = summarize(rows)
    print_summary(summary)

    out_path = args.out or args.jsonl.with_suffix(".summary.json")
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsaved={out_path}")


if __name__ == "__main__":
    main()
