"""Smoke-test: system CUDA Toolkit + light Python helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    section("Python (venv helpers)")
    print(f"executable: {sys.executable}")
    print(f"version:    {sys.version.split()[0]}")

    section("System CUDA toolkit (nvcc) — source of truth")
    nvcc = shutil.which("nvcc")
    if not nvcc:
        print("nvcc: NOT FOUND on PATH")
        print("Expected something like:")
        print(r"  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
        return 1
    print(f"nvcc: {nvcc}")
    subprocess.run(["nvcc", "--version"], check=False)

    section("GPU (nvidia-smi)")
    smi = shutil.which("nvidia-smi")
    if not smi:
        print("nvidia-smi: NOT FOUND")
        return 1
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        check=False,
    )

    section("numpy / matplotlib")
    try:
        import matplotlib
        import numpy as np

        print(f"numpy {np.__version__}")
        print(f"matplotlib {matplotlib.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"Python helper import failed: {exc}")
        print("Activate .venv and: pip install -r requirements.txt")
        return 1

    section("Result")
    print("OK — compile .cu with system nvcc; use .venv only for Python helpers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
