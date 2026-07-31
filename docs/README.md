# APC Nsight dashboard (GitHub Pages)

Recruiter-facing case study for the LLM Inference Runtime Lab warm cache ON/OFF experiment.

- `index.html` — static UI (reads `data.json` only; no hard-coded TTFT math)
- `data.json` — built from bench + nsys CSV + optional SQLite filters
- `assets/` — Nsight timeline screenshots

**URL:** https://sudhersankv.github.io/gpu-programming/

## Reproduce

```powershell
cd llm-inference-runtime-lab
python scripts\export_nsys_stats.py          # needs local .nsys-rep
python scripts\build_apc_dashboard_data.py   # reads profiling/bench/warm_cache_ab.json
python scripts\query_nsys_sqlite.py          # needs local .sqlite; merges late-trace metrics
```

Client timings live in `llm-inference-runtime-lab/profiling/bench/warm_cache_ab.json`.  
Speedup / reduction are **computed** in `build_apc_dashboard_data.py`.

Repo **Settings → Pages → Deploy from branch `main` → folder `/docs`**.
