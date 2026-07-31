"""Inspect Nsight SQLite and summarize tiny vs large HtoD for APC dashboard."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "profiling" / "reports"
OUT_JSON = ROOT.parent / "docs" / "data.json"
STEMS = ("cache_off_warm", "cache_on_warm")


def connect(stem: str) -> sqlite3.Connection:
    path = REPORTS / f"{stem}.sqlite"
    if not path.is_file():
        raise FileNotFoundError(path)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def tables(con: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]


def cols(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info('{table}')")]


def find_memcpy_table(con: sqlite3.Connection) -> str | None:
    names = tables(con)
    for cand in (
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMCPY2",
        "CUDA_GPU_MEMORY_OPERATIONS",
    ):
        if cand in names:
            return cand
    for name in names:
        if "MEMCPY" in name.upper():
            return name
    return None


def summarize_memcpy(con: sqlite3.Connection, table: str) -> dict:
    c = cols(con, table)
    # Nsight schemas vary; probe common column names.
    start_col = next((x for x in ("start", "timestamp", "Start") if x in c), None)
    end_col = next((x for x in ("end", "End") if x in c), None)
    bytes_col = next(
        (x for x in ("bytes", "copySize", "size", "Bytes", "memoryBytes") if x in c),
        None,
    )
    kind_col = next(
        (
            x
            for x in (
                "copyKind",
                "kind",
                "memKind",
                "memcpyKind",
                "srcKind",
                "CopyKind",
            )
            if x in c
        ),
        None,
    )

    print(f"  table={table}")
    print(f"  cols={c}")
    print(
        f"  mapped start={start_col} end={end_col} bytes={bytes_col} kind={kind_col}"
    )

    # sample a few rows
    sample = con.execute(f"SELECT * FROM [{table}] LIMIT 3").fetchall()
    if sample:
        print("  sample0 keys:", list(sample[0].keys()))
        print("  sample0:", dict(sample[0]))

    # If copyKind is an id into ENUM/StringIds, resolve later.
    n = con.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
    print(f"  nrows={n}")

    # Discover kind values
    if kind_col and bytes_col:
        kinds = con.execute(
            f"SELECT [{kind_col}] AS k, COUNT(*) AS n, SUM([{bytes_col}]) AS b "
            f"FROM [{table}] GROUP BY [{kind_col}] ORDER BY n DESC"
        ).fetchall()
        print("  kinds:", [dict(r) for r in kinds[:20]])

    return {
        "table": table,
        "columns": c,
        "start_col": start_col,
        "end_col": end_col,
        "bytes_col": bytes_col,
        "kind_col": kind_col,
    }


def main() -> int:
    for stem in STEMS:
        print("===", stem)
        con = connect(stem)
        ts = tables(con)
        interesting = [
            t
            for t in ts
            if any(
                k in t.upper()
                for k in ("MEMCPY", "MEM_", "CUPTI", "CUDA", "STRING", "ENUM")
            )
        ]
        print("n_tables", len(ts))
        print("interesting count", len(interesting))
        for t in interesting:
            if any(k in t.upper() for k in ("MEMCPY", "MEM_OP", "MEMORY")):
                print(" ", t, cols(con, t)[:12])

        mt = find_memcpy_table(con)
        if not mt:
            print("NO MEMCPY TABLE FOUND")
            # print all table names for debugging
            print(ts[:100])
            con.close()
            continue
        summarize_memcpy(con, mt)
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
