"""Read results/hf_baseline_*.json and plot TTFT / E2E / tok/s."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

paths = sorted(Path("results").glob("hf_baseline_*.json"))
if not paths:
    raise SystemExit("No results/hf_baseline_*.json files found")

rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
xs = list(range(1, len(rows) + 1))
ttft = [r["ttft_s"] for r in rows]
e2e = [r["e2e_s"] for r in rows]
tps = [r["output_tok_per_s"] for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(xs, ttft, marker="o")
axes[0].set_title("TTFT (s)")
axes[0].set_xlabel("run")

axes[1].plot(xs, e2e, marker="o")
axes[1].set_title("E2E (s)")
axes[1].set_xlabel("run")

axes[2].plot(xs, tps, marker="o")
axes[2].set_title("Output tok/s")
axes[2].set_xlabel("run")

fig.suptitle(rows[0].get("model", "hf_baseline"))
fig.tight_layout()

out = Path("results") / "hf_baseline_plot.png"
fig.savefig(out, dpi=150)
print(f"saved={out}")
plt.show()