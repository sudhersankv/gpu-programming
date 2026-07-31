# APC Nsight dashboard (GitHub Pages)

Static site for the LLM Inference Runtime Lab Phase 6 A/B:

- `index.html` — comparison UI
- `data.json` — from `nsys stats` CSVs **and** filtered SQLite queries
- `assets/` — warm ON/OFF timeline screenshots

**URL (after Pages is enabled):** https://sudhersankv.github.io/gpu-programming/

Rebuild from the lab folder (needs local `.nsys-rep` / `.sqlite` under `profiling/reports/`):

```powershell
cd llm-inference-runtime-lab
python scripts\export_nsys_stats.py
python scripts\build_apc_dashboard_data.py
python scripts\query_nsys_sqlite.py
```

Raw `.sqlite` files stay gitignored; only the small summarized `data.json` is published.

Repo **Settings → Pages → Deploy from branch `main` → folder `/docs`**.
