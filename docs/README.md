# docs/

Static GitHub Pages site for the LLM Inference Runtime Lab.

## APC Nsight A/B dashboard

Open locally: open `index.html` in a browser (needs `data.json` beside it), or after Pages is enabled:

https://sudhersankv.github.io/gpu-programming/

### Regenerate data

From `llm-inference-runtime-lab/` (requires Docker image `llm-lab-vllm:profile` and local `.nsys-rep` files):

```powershell
python scripts\export_nsys_stats.py
python scripts\build_apc_dashboard_data.py
```

Then copy refreshed screenshots if needed:

```powershell
Copy-Item profiling\screenshots\warm_cache_off_vs_cache_on.png ..\docs\assets\warm_events.png
Copy-Item profiling\screenshots\warm_cache_off_vs_cache_on_xoomed_out.png ..\docs\assets\warm_overview.png
```
