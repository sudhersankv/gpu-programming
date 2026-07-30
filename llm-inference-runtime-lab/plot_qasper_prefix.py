"""Plot Qasper shared vs control prefix-cache benchmark JSONLs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    # stable order by timestamp / file order
    return rows


def short_id(pid: str) -> str:
    # qasper_shared_1908.06606_q00_... -> q00
    parts = pid.split("_")
    for p in parts:
        if p.startswith("q") and p[1:].isdigit():
            return p
    return pid[-8:]


def summarize(name: str, rows: list[dict]) -> None:
    ttft = [r["ttft_s"] for r in rows]
    e2e = [r["e2e_s"] for r in rows]
    tps = [r["output_tok_per_s"] for r in rows]
    print(f"\n=== {name} (n={len(rows)}) ===")
    print(f"{'idx':>3}  {'id':6}  {'ttft_s':>8}  {'e2e_s':>8}  {'tok/s':>7}  {'out':>3}")
    for i, r in enumerate(rows):
        print(
            f"{i:3d}  {short_id(r['prompt_id']):6}  "
            f"{r['ttft_s']:8.3f}  {r['e2e_s']:8.3f}  "
            f"{r['output_tok_per_s']:7.1f}  {r['output_tokens']:3d}"
        )
    print(
        f"TTFT  mean={np.mean(ttft):.3f}  p50={np.median(ttft):.3f}  "
        f"min={np.min(ttft):.3f}  max={np.max(ttft):.3f}"
    )
    print(
        f"E2E   mean={np.mean(e2e):.3f}  p50={np.median(e2e):.3f}"
    )
    print(f"tok/s mean={np.mean(tps):.1f}")
    if len(ttft) > 1:
        print(f"TTFT first={ttft[0]:.3f}  rest_mean={np.mean(ttft[1:]):.3f}")


def annotate_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shared",
        type=Path,
        default=Path("results/qasper_shared_cache_on.jsonl"),
    )
    p.add_argument(
        "--control",
        type=Path,
        default=Path("results/qasper_control_cache_on.jsonl"),
    )
    p.add_argument("--out", type=Path, default=Path("results/qasper_prefix_compare.png"))
    args = p.parse_args()

    shared = load_measured(args.shared)
    control = load_measured(args.control)
    summarize("shared (cache on)", shared)
    summarize("control (cache on)", control)

    n = max(len(shared), len(control))
    x = np.arange(n)
    w = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1) TTFT per request (the main prefix-cache view)
    ax = axes[0, 0]
    s_ttft = [r["ttft_s"] for r in shared]
    c_ttft = [r["ttft_s"] for r in control]
    b1 = ax.bar(x[: len(s_ttft)] - w / 2, s_ttft, w, label="shared", color="#2a9d8f")
    b2 = ax.bar(x[: len(c_ttft)] + w / 2, c_ttft, w, label="control", color="#e76f51")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.set_title("TTFT by request index (lower = better)")
    ax.set_xlabel("request index")
    ax.set_ylabel("TTFT (s)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2) E2E per request
    ax = axes[0, 1]
    s_e2e = [r["e2e_s"] for r in shared]
    c_e2e = [r["e2e_s"] for r in control]
    b1 = ax.bar(x[: len(s_e2e)] - w / 2, s_e2e, w, label="shared", color="#2a9d8f")
    b2 = ax.bar(x[: len(c_e2e)] + w / 2, c_e2e, w, label="control", color="#e76f51")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.set_title("E2E latency by request index")
    ax.set_xlabel("request index")
    ax.set_ylabel("E2E (s)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 3) Line overlay TTFT (cache warm-up shape)
    ax = axes[1, 0]
    ax.plot(range(len(s_ttft)), s_ttft, "o-", label="shared", color="#2a9d8f", lw=2)
    ax.plot(range(len(c_ttft)), c_ttft, "s-", label="control", color="#e76f51", lw=2)
    for i, v in enumerate(s_ttft):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    for i, v in enumerate(c_ttft):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7)
    ax.set_title("TTFT trajectory (prefix reuse shows up after req 0)")
    ax.set_xlabel("request index")
    ax.set_ylabel("TTFT (s)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4) Aggregate means (first vs rest for shared)
    ax = axes[1, 1]
    labels = ["shared\nfirst", "shared\nrest mean", "control\nmean"]
    vals = [
        s_ttft[0],
        float(np.mean(s_ttft[1:])) if len(s_ttft) > 1 else float("nan"),
        float(np.mean(c_ttft)),
    ]
    colors = ["#264653", "#2a9d8f", "#e76f51"]
    bars = ax.bar(labels, vals, color=colors)
    annotate_bars(ax, bars)
    ax.set_title("TTFT summary")
    ax.set_ylabel("TTFT (s)")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Qasper prefix-cache: shared vs control (vLLM cache on)", fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"\nsaved={args.out}")
    plt.show()


if __name__ == "__main__":
    main()