"""Smoke-test: LeetGPU CLI (primary) + optional local CUDA toolkit."""

from __future__ import annotations

import shutil
import subprocess
import sys


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    ok = True

    section("Python")
    print(f"executable: {sys.executable}")
    print(f"version:    {sys.version.split()[0]}")

    section("LeetGPU CLI (primary for leetgpu/ track)")
    leetgpu = shutil.which("leetgpu")
    if not leetgpu:
        print("leetgpu: NOT FOUND on PATH")
        print("Install (Windows PowerShell):")
        print(
            "  Invoke-WebRequest -Uri https://cli.leetgpu.com/install.ps1 "
            "-OutFile install.ps1; ./install.ps1"
        )
        print("Docs: https://leetgpu.com/cli")
        ok = False
    else:
        print(f"leetgpu: {leetgpu}")
        subprocess.run(["leetgpu", "--help"], check=False)

    section("Local nvcc (optional fallback)")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        print("nvcc: not on PATH (ok if you only use LeetGPU)")
    else:
        print(f"nvcc: {nvcc}")
        subprocess.run(["nvcc", "--version"], check=False)

    section("GPU / nvidia-smi (optional)")
    smi = shutil.which("nvidia-smi")
    if not smi:
        print("nvidia-smi: not found (ok — LeetGPU does not need a local GPU)")
    else:
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
        )

    section("numpy / matplotlib (optional helpers)")
    try:
        import matplotlib
        import numpy as np

        print(f"numpy {np.__version__}")
        print(f"matplotlib {matplotlib.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"Python helper import skipped/failed: {exc}")

    section("Result")
    if ok:
        print("OK — use: .\\scripts\\leetgpu_run.ps1 <path-to-kernel>")
        return 0
    print("Install the LeetGPU CLI, then re-run this script.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
