"""
Prepare deterministic Qasper shared-prefix / control prompt JSONs
for vLLM prefix-caching benchmarks.

Outputs (match prompts/short.json schema):
  prompts/qasper_shared_1024.json
  prompts/qasper_control_1024.json

Loads allenai/qasper from the official v0.3 train/dev archive (deterministic
order). Tokenizes paper text with Qwen/Qwen2.5-1.5B-Instruct.
"""

from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PAPER_TOKEN_BUDGET = 1024
MAX_NEW_TOKENS = 64
NUM_PROMPTS = 10
MIN_QUESTIONS_SHARED = NUM_PROMPTS

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "prompts"
CACHE_DIR = ROOT / ".cache" / "qasper"
TRAIN_DEV_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
)
TRAIN_FILE = "qasper-train-v0.3.json"
DEV_FILE = "qasper-dev-v0.3.json"


def download_train_dev_archive() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / "qasper-train-dev-v0.3.tgz"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    print(f"Downloading {TRAIN_DEV_URL}")
    with urllib.request.urlopen(TRAIN_DEV_URL) as resp:
        dest.write_bytes(resp.read())
    return dest


def load_json_from_tgz(archive: Path, member: str) -> dict:
    with tarfile.open(archive, mode="r:gz") as tf:
        f = tf.extractfile(member)
        if f is None:
            raise FileNotFoundError(member)
        return json.loads(f.read().decode("utf-8"))


def load_rows() -> list[dict]:
    """Train then validation (dev), paper order = JSON object key order."""
    archive = download_train_dev_archive()
    rows: list[dict] = []
    for member in (TRAIN_FILE, DEV_FILE):
        papers = load_json_from_tgz(archive, member)
        for paper_id, paper in papers.items():
            row = dict(paper)
            row["id"] = paper_id
            rows.append(row)
    return rows


def flatten_paper(row: dict) -> str:
    """Deterministic plain-text flattening: title, abstract, then sections."""
    parts: list[str] = []
    title = (row.get("title") or "").strip()
    abstract = (row.get("abstract") or "").strip()
    if title:
        parts.append(title)
    if abstract:
        parts.append(abstract)

    for section in row.get("full_text") or []:
        name = (section.get("section_name") or "").strip()
        if name:
            parts.append(name)
        for p in section.get("paragraphs") or []:
            p = (p or "").strip()
            if p:
                parts.append(p)
    return "\n\n".join(parts)


def paper_token_len(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def format_prompt(paper_text: str, question: str) -> str:
    return (
        "You are given the following scientific paper.\n\n"
        f"{paper_text}\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Answer:"
    )


def questions_of(row: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for qa in row.get("qas") or []:
        q = (qa.get("question") or "").strip()
        qid = str(qa.get("question_id") or "")
        if q:
            out.append((qid, q))
    return out


def prompt_token_stats(prompts: list[dict], tokenizer) -> tuple[int, int, float]:
    counts = [
        len(tokenizer.encode(p["text"], add_special_tokens=False)) for p in prompts
    ]
    return min(counts), max(counts), sum(counts) / len(counts)


def main() -> None:
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading Qasper train + validation (original order)...")
    rows = load_rows()
    print(f"Papers loaded: {len(rows)}")

    prepared: list[dict] = []
    for row in rows:
        qs = questions_of(row)
        if not qs:
            continue
        flat = flatten_paper(row)
        ids = tokenizer.encode(flat, add_special_tokens=False)
        if len(ids) < PAPER_TOKEN_BUDGET:
            continue
        truncated = tokenizer.decode(ids[:PAPER_TOKEN_BUDGET], skip_special_tokens=True)
        prepared.append(
            {
                "paper_id": str(row["id"]),
                "truncated_paper": truncated,
                "questions": qs,
            }
        )

    if not prepared:
        raise SystemExit("No papers with >=1024 tokens and at least one question.")

    # First paper with >=1024 tokens and at least NUM_PROMPTS questions.
    shared_src = None
    for p in prepared:
        if len(p["questions"]) >= MIN_QUESTIONS_SHARED:
            shared_src = p
            break
    if shared_src is None:
        raise SystemExit(
            f"No paper with >={MIN_QUESTIONS_SHARED} questions and >=1024 tokens."
        )

    shared_prompts = []
    for i, (qid, question) in enumerate(shared_src["questions"][:NUM_PROMPTS]):
        shared_prompts.append(
            {
                "prompt_id": f"qasper_shared_{shared_src['paper_id']}_q{i:02d}_{qid}",
                "text": format_prompt(shared_src["truncated_paper"], question),
                "max_new_tokens": MAX_NEW_TOKENS,
            }
        )

    print(
        f"Shared: paper_id={shared_src['paper_id']} "
        f"questions={len(shared_prompts)} "
        f"paper_tokens~={paper_token_len(shared_src['truncated_paper'], tokenizer)}"
    )

    control_prompts = []
    for p in prepared:
        if p["paper_id"] == shared_src["paper_id"]:
            continue
        qid, question = p["questions"][0]
        control_prompts.append(
            {
                "prompt_id": f"qasper_control_{p['paper_id']}_q00_{qid}",
                "text": format_prompt(p["truncated_paper"], question),
                "max_new_tokens": MAX_NEW_TOKENS,
            }
        )
        if len(control_prompts) >= NUM_PROMPTS:
            break

    if len(control_prompts) < NUM_PROMPTS:
        raise SystemExit(
            f"Needed {NUM_PROMPTS} control papers, only found {len(control_prompts)}."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shared_path = OUT_DIR / "qasper_shared_1024.json"
    control_path = OUT_DIR / "qasper_control_1024.json"

    shared_path.write_text(
        json.dumps(shared_prompts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    control_path.write_text(
        json.dumps(control_prompts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    marker = "\n\nQuestion:\n"
    prefixes = [p["text"].split(marker, 1)[0] for p in shared_prompts]
    assert len(set(prefixes)) == 1, "Shared prompts must have identical paper prefixes"
    control_prefixes = [p["text"].split(marker, 1)[0] for p in control_prompts]
    assert len(set(control_prefixes)) == len(control_prefixes), (
        "Control prompts must each use a distinct paper prefix"
    )
    assert len(shared_prompts) == NUM_PROMPTS
    assert len(control_prompts) == NUM_PROMPTS

    shared_min, shared_max, shared_avg = prompt_token_stats(shared_prompts, tokenizer)
    control_min, control_max, control_avg = prompt_token_stats(control_prompts, tokenizer)

    print(f"Wrote {shared_path} ({len(shared_prompts)} prompts)")
    print(f"Wrote {control_path} ({len(control_prompts)} prompts)")
    print("Validation summary")
    print(
        f"  shared: n={len(shared_prompts)} "
        f"min={shared_min} max={shared_max} avg={shared_avg:.1f}"
    )
    print(
        f"  control: n={len(control_prompts)} "
        f"min={control_min} max={control_max} avg={control_avg:.1f}"
    )
    print("All assertions passed.")


if __name__ == "__main__":
    main()
