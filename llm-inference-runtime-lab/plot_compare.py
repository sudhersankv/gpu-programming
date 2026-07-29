"""Compare HF vs vLLM from one or more bench JSONL files (measured trials only)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = (
    ("ttft_s", "TTFT (s)"),
    ("e2e_s", "E2E (s)"),
    ("output_tok_per_s", "Output tok/s"),
)


def load_measured(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
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
        raise SystemExit("No measured rows found")
    return rows


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "jsonl",
        nargs="+",
        type=Path,
        help="One or more bench_*.jsonl files (e.g. hf + vllm)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/compare_hf_vllm.png"),
    )
    args = parser.parse_args()

    rows = load_measured(args.jsonl)

    # engine -> prompt_id -> metric -> values
    data: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        engine = row["engine"]
        pid = row["prompt_id"]
        for metric, _ in METRICS:
            data[engine][pid][metric].append(float(row[metric]))

    engines = sorted(data.keys())
    prompt_ids = sorted({pid for eng in data.values() for pid in eng})

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = list(range(len(prompt_ids)))
    width = 0.35 if len(engines) > 1 else 0.6

    for ax, (metric, title) in zip(axes, METRICS):
        for i, engine in enumerate(engines):
            means = []
            for pid in prompt_ids:
                vals = data[engine].get(pid, {}).get(metric, [])
                means.append(mean(vals) if vals else float("nan"))
            offset = (i - (len(engines) - 1) / 2) * width
            ax.bar([xi + offset for xi in x], means, width=width, label=engine)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(prompt_ids, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=8)

    fig.suptitle("HF vs vLLM (mean over measured trials)")
    fig.tight_layout()
    args.out.parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved={args.out}")
    plt.show()


if __name__ == "__main__":
    main()
