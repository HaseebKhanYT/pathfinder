"""LangChain-based extractor: report text -> `ReportExtraction` JSON.

Uses `ChatOpenAI.with_structured_output(..., method="json_schema", strict=True)`
so OpenAI constrains decoding to the schema; malformed JSON cannot come back.
Results are appended to a JSONL cache keyed by report id, so an interrupted
run resumes where it stopped and re-runs never pay twice for a report.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .postprocess import derive
from .prompt import HUMAN_TEMPLATE, SYSTEM_PROMPT
from .schema import ReportExtraction

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "gpt-5.4-mini"


def build_chain(model: str = DEFAULT_MODEL, reasoning_effort: str | None = "low"):
    """Return a runnable that maps a message list to a `ReportExtraction`."""
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing: put it in .env at the repo root")
    kwargs = dict(model=model, max_retries=6, timeout=120)
    # Reasoning models (gpt-5*, o*) take reasoning_effort, not temperature.
    if model.startswith(("gpt-5", "o")):
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = 0
    llm = ChatOpenAI(**kwargs)
    return llm.with_structured_output(ReportExtraction, method="json_schema", strict=True, include_raw=True)


def fingerprint(model: str, reasoning_effort: str | None) -> str:
    """Hash of everything that changes an extraction: model, effort, prompt, schema.

    Stored on every record; a cached record only counts when it matches, so
    editing the prompt or schema, or changing effort, re-extracts.
    """
    schema = json.dumps(ReportExtraction.model_json_schema(), sort_keys=True)
    blob = "\n".join([model, str(reasoning_effort), SYSTEM_PROMPT, schema])
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _messages(report: dict):
    return [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(HUMAN_TEMPLATE.format(report_id=report["id"], report_text=report["report_text"])),
    ]


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:  # e.g. a line cut off by a kill mid-write
            print(f"[cache] skipping unreadable line {n} in {path.name}")
            continue
        # Later lines win, but never let an error record replace a good extraction.
        if "extraction" in rec or "extraction" not in out.get(rec["id"], {}):
            out[rec["id"]] = rec
    return out


def extract_reports(
    reports: Iterable[dict],
    cache_path: Path,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str | None = "low",
    concurrency: int = 32,
) -> dict[str, dict]:
    """Extract every report not already cached; return {id: record} for all of them.

    A record is `{"id", "model", "extraction", "derived", "usage"}`, or
    `{"id", "model", "error"}` when the call failed after retries (errored ids
    are retried on the next run because they are not counted as cached).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(model, reasoning_effort)
    cache = {k: v for k, v in load_cache(cache_path).items() if "extraction" in v and v.get("fingerprint") == fp}
    reports = list(reports)
    todo = [r for r in reports if r["id"] not in cache]
    print(f"[extract] {len(reports)} reports, {len(cache)} cached, {len(todo)} to run with {model}")
    if todo:
        chain = build_chain(model, reasoning_effort)
        n_done = n_err = 0
        with cache_path.open("a") as fh:
            # Write each result the moment it arrives, so an interrupt loses at
            # most the calls still in flight, never finished (billed) ones.
            for idx, res in chain.batch_as_completed(
                [_messages(r) for r in todo],
                config={"max_concurrency": concurrency},
                return_exceptions=True,
            ):
                rec = _to_record(todo[idx]["id"], model, res)
                rec["fingerprint"] = fp
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                n_done += 1
                if "extraction" in rec:
                    cache[rec["id"]] = rec
                else:
                    n_err += 1
                if n_done % 250 == 0 or n_done == len(todo):
                    print(f"[extract] {n_done}/{len(todo)} done ({n_err} errors)", flush=True)
    return {r["id"]: cache[r["id"]] for r in reports if r["id"] in cache}


def _to_record(report_id: str, model: str, res) -> dict:
    if isinstance(res, Exception):
        return {"id": report_id, "model": model, "error": repr(res)[:500]}
    parsed: ReportExtraction | None = res.get("parsed")
    if parsed is None:
        return {"id": report_id, "model": model, "error": repr(res.get("parsing_error"))[:500]}
    usage = getattr(res.get("raw"), "usage_metadata", None) or {}
    extraction = parsed.model_dump()
    return {
        "id": report_id,
        "model": model,
        "extraction": extraction,
        "derived": derive(extraction),
        "usage": {k: usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")},
    }
