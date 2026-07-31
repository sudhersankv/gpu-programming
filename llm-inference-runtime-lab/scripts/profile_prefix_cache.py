"""
Reproducible Nsight Systems profiling harness for vLLM prefix caching.

Orchestration (engine-agnostic):
  launch server under nsys → poll /health →
  warm with 1 unrelated short request →
  run_benchmark.py (2 shared Qasper prompts) →
  docker stop → verify .nsys-rep

Warmup uses a different prompt so APC is not seeded with the paper prefix
when --cache on.

Usage (from llm-inference-runtime-lab/):
  python scripts/profile_prefix_cache.py --cache on
  python scripts/profile_prefix_cache.py --cache off
  python scripts/profile_prefix_cache.py --cache on --output cache_on_warm
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SHARED_PROMPTS = ROOT / "prompts" / "qasper_shared_1024.json"
REPORTS_DIR = ROOT / "profiling" / "reports"
IMAGE = "llm-lab-vllm:profile"
HEALTH_URL = "http://localhost:8000/health"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
HF_CACHE = Path.home() / ".cache" / "huggingface"
NUM_PROFILE_PROMPTS = 2
HEALTH_TIMEOUT_S = 600
HEALTH_POLL_S = 2.0
DOCKER_STOP_TIMEOUT_S = 120

# Unrelated to Qasper paper — warms kernels/CUDA graphs without seeding APC.
WARMUP_PROMPT = {
    "prompt_id": "profile_warmup_unrelated",
    "text": "Say hello in one short sentence.",
    "max_new_tokens": 16,
}

# ---------------------------------------------------------------------------
# Engine-specific: only this builds the docker/nsys/server command.
# ---------------------------------------------------------------------------


def build_vllm_nsys_docker_cmd(
    *,
    container_name: str,
    report_stem: str,
    enable_prefix_cache: bool,
    reports_host: Path,
) -> list[str]:
    """Return `docker run ...` argv that starts vLLM under nsys profile."""
    reports_host.mkdir(parents=True, exist_ok=True)
    vllm_args = [
        "/usr/bin/python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_ID,
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.85",
    ]
    if enable_prefix_cache:
        vllm_args.append("--enable-prefix-caching")
    else:
        # vLLM V1 defaults prefix caching ON — must disable explicitly.
        vllm_args.append("--no-enable-prefix-caching")

    return [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        container_name,
        "--gpus",
        "all",
        "--ipc=host",
        "--cap-add=SYS_ADMIN",
        "-p",
        "8000:8000",
        "-v",
        f"{HF_CACHE}:/root/.cache/huggingface",
        "-v",
        f"{reports_host}:/reports",
        "--entrypoint",
        "nsys",
        IMAGE,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=process-tree",
        f"--output=/reports/{report_stem}",
        "--force-overwrite=true",
        *vllm_args,
    ]


# ---------------------------------------------------------------------------
# Shared orchestration helpers
# ---------------------------------------------------------------------------


def write_two_prompt_fixture(shared_path: Path, dest: Path) -> Path:
    """First two shared-prefix prompts → temporary JSON for the harness."""
    prompts = json.loads(shared_path.read_text(encoding="utf-8"))
    if len(prompts) < NUM_PROFILE_PROMPTS:
        raise SystemExit(
            f"Need >={NUM_PROFILE_PROMPTS} prompts in {shared_path}, found {len(prompts)}"
        )
    dest.write_text(
        json.dumps(prompts[:NUM_PROFILE_PROMPTS], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dest


def write_warmup_fixture(dest: Path) -> Path:
    """Single unrelated short prompt — does not share the Qasper paper prefix."""
    dest.write_text(
        json.dumps([WARMUP_PROMPT], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dest


def stream_subprocess(cmd: list[str], *, cwd: Path | None = None) -> int:
    """Run a process, stream combined stdout/stderr to the console."""
    print("+", " ".join(cmd), flush=True)
    with subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        return proc.wait()


def docker_logs_follow(container_name: str, stop_event: threading.Event) -> None:
    """Stream `docker logs -f` until stop_event is set."""
    proc = subprocess.Popen(
        ["docker", "logs", "-f", container_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        while not stop_event.is_set():
            line = proc.stdout.readline()
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
            elif proc.poll() is not None:
                break
            else:
                time.sleep(0.05)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def wait_for_health(url: str, timeout_s: float) -> None:
    """Poll GET /health until HTTP 200 or timeout (no fixed sleep-only wait)."""
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                r = client.get(url)
                if r.status_code == 200:
                    print(f"health OK ({url})", flush=True)
                    return
                last_err = RuntimeError(f"status={r.status_code} body={r.text[:200]!r}")
            except Exception as exc:  # noqa: BLE001 — poll until up
                last_err = exc
            time.sleep(HEALTH_POLL_S)
    raise TimeoutError(f"Timed out waiting for {url} after {timeout_s}s; last={last_err}")


def docker_stop(container_name: str) -> None:
    print(f"docker stop -t {DOCKER_STOP_TIMEOUT_S} {container_name}", flush=True)
    subprocess.run(
        ["docker", "stop", "-t", str(DOCKER_STOP_TIMEOUT_S), container_name],
        check=False,
    )


def run_benchmark(prompts_path: Path, out_jsonl: Path) -> None:
    """Reuse existing harness — no duplicated HTTP client logic."""
    cmd = [
        sys.executable,
        str(ROOT / "run_benchmark.py"),
        "--engine",
        "vllm",
        "--warmup",
        "0",
        "--trials",
        "1",
        "--prompts",
        str(prompts_path),
        "--out",
        str(out_jsonl),
    ]
    rc = stream_subprocess(cmd, cwd=ROOT)
    if rc != 0:
        raise RuntimeError(f"run_benchmark.py failed with exit code {rc}")


def verify_report(report_path: Path) -> None:
    if not report_path.is_file() or report_path.stat().st_size == 0:
        raise FileNotFoundError(f"Nsight report missing or empty: {report_path}")
    print(f"report OK: {report_path} ({report_path.stat().st_size} bytes)", flush=True)


def run_profile(*, cache_on: bool, output_stem: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{output_stem}.nsys-rep"
    container_name = f"llm-lab-profile-{output_stem}-{uuid.uuid4().hex[:8]}"

    with tempfile.TemporaryDirectory(prefix="qasper_profile_") as tmp:
        tmp_path = Path(tmp)
        warmup_fixture = write_warmup_fixture(tmp_path / "warmup_unrelated.json")
        fixture = write_two_prompt_fixture(
            SHARED_PROMPTS, tmp_path / "qasper_shared_2.json"
        )
        warmup_out = tmp_path / f"bench_{output_stem}_warmup.jsonl"
        bench_out = tmp_path / f"bench_{output_stem}.jsonl"

        cmd = build_vllm_nsys_docker_cmd(
            container_name=container_name,
            report_stem=output_stem,
            enable_prefix_cache=cache_on,
            reports_host=REPORTS_DIR,
        )
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        stop_logs = threading.Event()
        log_thread = threading.Thread(
            target=docker_logs_follow,
            args=(container_name, stop_logs),
            daemon=True,
        )
        log_thread.start()

        try:
            wait_for_health(HEALTH_URL, HEALTH_TIMEOUT_S)
            # Old (cold measured pair right after health) — kept for reference:
            # run_benchmark(fixture, bench_out)
            print("warmup: unrelated short request (does not seed paper APC)", flush=True)
            run_benchmark(warmup_fixture, warmup_out)
            print("measure: 2 shared-prefix Qasper prompts", flush=True)
            run_benchmark(fixture, bench_out)
        finally:
            stop_logs.set()
            docker_stop(container_name)
            log_thread.join(timeout=10)

        # nsys may finish writing shortly after stop returns
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if report_path.is_file() and report_path.stat().st_size > 0:
                break
            time.sleep(1.0)

        verify_report(report_path)
        return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile vLLM with prefix cache ON or OFF (2 Qasper shared prompts)."
    )
    parser.add_argument(
        "--cache",
        choices=["on", "off"],
        required=True,
        help="Enable or disable vLLM --enable-prefix-caching",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Report stem under profiling/reports/ (default: cache_on or cache_off)",
    )
    args = parser.parse_args()

    cache_on = args.cache == "on"
    stem = args.output or ("cache_on" if cache_on else "cache_off")

    if not SHARED_PROMPTS.is_file():
        print(f"missing prompts: {SHARED_PROMPTS}", file=sys.stderr)
        return 1

    print(
        f"cache={'on' if cache_on else 'off'} output={stem} "
        f"prompts={SHARED_PROMPTS.name}[:{NUM_PROFILE_PROMPTS}]",
        flush=True,
    )
    try:
        path = run_profile(cache_on=cache_on, output_stem=stem)
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"\nsuccess: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
