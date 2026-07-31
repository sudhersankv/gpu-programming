"""Query local Nsight .sqlite exports and merge APC-relevant metrics into docs/data.json.

Requires profiling/reports/cache_{on,off}_warm.sqlite (from nsys stats / export).

Usage (from llm-inference-runtime-lab/):
  python scripts/query_nsys_sqlite.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
REPORTS = ROOT / "profiling" / "reports"
DATA_PATH = REPO / "docs" / "data.json"
STEMS = ("cache_off_warm", "cache_on_warm")
LATE_WINDOW_S = 30


def short_name(con: sqlite3.Connection, sid: int) -> str:
    row = con.execute("SELECT value FROM StringIds WHERE id=?", (sid,)).fetchone()
    return row[0] if row else str(sid)


def htod_buckets(
    con: sqlite3.Connection, *, where_extra: str = "", params: tuple = ()
) -> dict:
    row = con.execute(
        f"""
        SELECT
          SUM(CASE WHEN bytes < 4096 THEN 1 ELSE 0 END),
          SUM(CASE WHEN bytes < 4096 THEN bytes ELSE 0 END),
          SUM(CASE WHEN bytes < 4096 THEN (end - start) ELSE 0 END),
          SUM(CASE WHEN bytes >= 1048576 THEN 1 ELSE 0 END),
          SUM(CASE WHEN bytes >= 1048576 THEN bytes ELSE 0 END),
          SUM(CASE WHEN bytes >= 1048576 THEN (end - start) ELSE 0 END),
          COUNT(*),
          SUM(bytes),
          SUM(end - start)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind = 1 {where_extra}
        """,
        params,
    ).fetchone()

    def part(n: int, b: int, d: int, mb_digits: int = 3) -> dict:
        return {
            "count": int(n or 0),
            "total_mb": round((b or 0) / 1e6, mb_digits),
            "total_time_ms": round((d or 0) / 1e6, 3),
        }

    return {
        "tiny_lt_4kb": part(row[0], row[1], row[2], mb_digits=4),
        "large_ge_1mb": part(row[3], row[4], row[5]),
        "all_htod": part(row[6], row[7], row[8]),
    }


def pack(stem: str) -> dict:
    path = REPORTS / f"{stem}.sqlite"
    if not path.is_file():
        raise FileNotFoundError(path)

    con = sqlite3.connect(path)
    t1k = con.execute("SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    t1m = con.execute("SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()[0]
    t1 = max(t1k, t1m)
    late = t1 - LATE_WINDOW_S * 1_000_000_000

    kn, kdur = con.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(end - start), 0)
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start >= ?
        """,
        (late,),
    ).fetchone()

    signals: dict[str, dict] = {}
    for key, like in (
        ("gemm_ampere", "%ampere_bf16%"),
        ("flash_fwd", "%flash_fwd%"),
        ("reshape_and_cache", "%reshape_and_cache%"),
        ("triton_silu", "%mul_silu%"),
    ):
        n, dur = con.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(k.end - k.start), 0)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            WHERE k.start >= ? AND s.value LIKE ?
            """,
            (late, like),
        ).fetchone()
        signals[key] = {
            "instances": int(n),
            "total_time_ms": round(dur / 1e6, 2),
        }

    top: list[dict] = []
    for sid, n, dur in con.execute(
        """
        SELECT shortName, COUNT(*), SUM(end - start)
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start >= ?
        GROUP BY shortName
        ORDER BY 3 DESC
        LIMIT 8
        """,
        (late,),
    ):
        name = short_name(con, sid)
        short = name if len(name) <= 48 else name[:45] + "…"
        top.append(
            {
                "name": short,
                "full_name": name,
                "instances": int(n),
                "total_time_ms": round(dur / 1e6, 2),
            }
        )

    out = {
        "sqlite_window": {
            "label": "last_30s_of_trace",
            "window_s": LATE_WINDOW_S,
            "note": (
                "Approx request era after model load; large HtoD weights finish "
                "earlier (~first half of capture)."
            ),
        },
        "htod_session": htod_buckets(con),
        "htod_late_30s": htod_buckets(con, where_extra="AND start >= ?", params=(late,)),
        "kernels_late_30s": {
            "count": int(kn),
            "total_time_ms": round(kdur / 1e6, 2),
            "top": top,
            "signals": signals,
        },
    }
    con.close()
    return out


def main() -> int:
    if not DATA_PATH.is_file():
        print(f"missing {DATA_PATH}; run build_apc_dashboard_data.py first")
        return 1

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    data["meta"]["caveat"] = (
        "Whole-session HtoD totals match because ~3GB weight upload dominates. "
        "SQLite-filtered view: split tiny vs large HtoD, and kernel time in the "
        "last 30s of each trace (after load) — that is where cache OFF does more "
        "prefill GEMM / attention work."
    )
    data["meta"]["data_sources"] = [
        "nsys stats CSVs (session kernel/API summaries)",
        "CUPTI_ACTIVITY_KIND_MEMCPY + CUPTI_ACTIVITY_KIND_KERNEL from local .sqlite",
    ]
    data["sqlite"] = {
        "cache_off": pack("cache_off_warm"),
        "cache_on": pack("cache_on_warm"),
    }

    DATA_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    off_k = data["sqlite"]["cache_off"]["kernels_late_30s"]["total_time_ms"]
    on_k = data["sqlite"]["cache_on"]["kernels_late_30s"]["total_time_ms"]
    print(f"wrote {DATA_PATH}")
    print(f"late-30s kernel ms: OFF={off_k} ON={on_k} ratio={off_k / on_k:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
