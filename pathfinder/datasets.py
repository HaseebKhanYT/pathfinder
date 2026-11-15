"""Report datasets: the built-in train/test splits plus any JSONL a user uploads.

An uploaded dataset has the shape of test.jsonl: one JSON object per line with
`id` and `report_text` (other keys, such as `labels`, are kept but unused). It is
saved to outputs/datasets/<name>.jsonl, and its extractions go to the usual
outputs/extractions/<name>.<model>.jsonl, so every split-based code path
(`load_split`, `cache_path`, the app's split selector) works on it unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .extraction.extractor import REPO_ROOT

BUILTIN_DIR = REPO_ROOT / "data" / "biobase" / "pack_reports"
BUILTIN = ("train", "test")
DATASETS_DIR = REPO_ROOT / "outputs" / "datasets"


def split_path(split: str) -> Path:
    """Where a split's reports live: data/ for train and test, outputs/datasets/ for uploads."""
    return (BUILTIN_DIR if split in BUILTIN else DATASETS_DIR) / f"{split}.jsonl"


def parse_reports(raw: bytes) -> list[dict]:
    """Rows of an uploaded JSONL file; ValueError naming the first bad line otherwise."""
    try:
        text = raw.decode("utf-8-sig")  # tolerate a BOM from editors that add one
    except UnicodeDecodeError as e:
        raise ValueError("the file is not UTF-8 text") from e
    rows, seen = [], set()
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"line {n} is not valid JSON ({e.msg})") from e
        if not isinstance(row, dict):
            raise ValueError(f"line {n} is not a JSON object")
        rid, txt = row.get("id"), row.get("report_text")
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError(f'line {n} has no "id" string')
        if not isinstance(txt, str) or not txt.strip():
            raise ValueError(f'line {n} ({rid}) has no "report_text" string')
        if rid in seen:
            raise ValueError(f"line {n} repeats the id {rid}")
        seen.add(rid)
        rows.append(row)
    if not rows:
        raise ValueError("the file has no reports")
    return rows


def _slug(filename: str) -> str:
    # "." would break the <split>.<model>.jsonl cache naming, so only [a-z0-9_-] survive.
    stem = Path(filename).name.rsplit(".", 1)[0].lower()
    return re.sub(r"[^a-z0-9_-]+", "-", stem).strip("-_")[:40] or "upload"


def dataset_name(filename: str, rows: list[dict]) -> str:
    """The split name an upload is stored under.

    A file named like a built-in split with the same reports *is* that split, so an
    uploaded test.jsonl reuses any test extractions already made. A same-named file
    with different reports gets a suffix instead of shadowing the built-in one.
    """
    name = _slug(filename)
    if name not in BUILTIN:
        return name
    builtin = split_path(name)
    if builtin.exists() and rows == parse_reports(builtin.read_bytes()):
        return name
    return f"{name}-upload"


def save_dataset(name: str, rows: list[dict]) -> Path:
    """Write an uploaded dataset (a no-op for a built-in split, which is read-only input)."""
    path = split_path(name)
    if name in BUILTIN:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path
