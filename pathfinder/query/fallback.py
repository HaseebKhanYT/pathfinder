"""Per-report yes/no check for the part of a question the schema cannot express.

The covered part of the question has already narrowed the candidates (e.g. to
colon reports). Each remaining report goes to the LLM with the yes/no
question; the LLM must quote the report text as evidence. A "yes" counts only
when its quote is actually found in the report; otherwise it is reported as
unverified rather than silently trusted.

This is the only part of the query layer that scales API spend with data, so
it never runs without an explicit go-ahead: the caller passes `confirm`, which
receives a cost estimate and returns True to proceed. Verdicts are cached in
outputs/fallback_cache.jsonl, keyed by the exact yes/no question text, so
re-running with the same `--rule ... --uncovered ...` (printed after every
query) costs nothing. Re-translating the question may reword the yes/no
question and miss the cache.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field as PField

from ..extraction.extractor import REPO_ROOT, build_chat_model

CACHE = REPO_ROOT / "outputs" / "fallback_cache.jsonl"
PROMPT_OVERHEAD_TOKENS = 250  # system + question, measured roughly
OUTPUT_TOKENS_PER_CALL = 120  # verdict + short quote (+ low reasoning)

SYSTEM = (
    "You answer one yes/no question about one synthetic pathology report. Answer only from the report text. "
    "If the report does not state it, the answer is no. evidence_quote must be copied verbatim from the report "
    "(a short span, under 30 words) that supports your answer; use an empty string if nothing in the report "
    "bears on the question."
)


class Verdict(BaseModel):
    matches: bool = PField(description="True only if the report clearly says yes.")
    evidence_quote: str = PField(description="Verbatim span from the report supporting the answer, or ''.")


@dataclass
class Estimate:
    reports: int
    cached: int
    calls: int
    input_tokens: int
    output_tokens: int


@dataclass
class FallbackResult:
    question: str
    matched: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)  # said yes, but quote not found in the text
    failed: list[str] = field(default_factory=list)  # call errored or unparseable: NOT checked
    rejected: list[str] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)
    estimate: Estimate | None = None
    skipped_reason: str | None = None
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


MIN_QUOTE_WORDS = 3


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.+/%-][a-z0-9]+)*", s.lower())


def quote_in_text(quote: str, text: str) -> bool:
    """True when the quote has >= 3 words and appears as a contiguous word run in the text."""
    q, t = _tokens(quote), _tokens(text)
    if len(q) < MIN_QUOTE_WORDS:
        return False
    n = len(q)
    return any(t[i:i + n] == q for i in range(len(t) - n + 1))


PROMPT_VERSION = hashlib.sha256(
    (SYSTEM + json.dumps(Verdict.model_json_schema(), sort_keys=True)).encode()).hexdigest()[:8]


def _key(question: str, model: str, rid: str, effort: str | None = "low") -> str:
    """Verdict cache key: question text, model, effort, prompt+schema version, report."""
    blob = f"{model}\n{effort}\n{PROMPT_VERSION}\n{_norm(question)}\n{rid}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _load_cache() -> dict[str, dict]:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            try:
                rec = json.loads(line)
                out[rec["key"]] = rec
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def estimate(question: str, ids: list[str], texts: dict[str, str], model: str,
             effort: str | None = "low") -> Estimate:
    cache = _load_cache()
    todo = [i for i in ids if _key(question, model, i, effort) not in cache]
    tok_in = sum(len(texts[i]) // 4 + PROMPT_OVERHEAD_TOKENS for i in todo)
    return Estimate(len(ids), len(ids) - len(todo), len(todo), tok_in, OUTPUT_TOKENS_PER_CALL * len(todo))


def run_fallback(
    question: str,
    ids: list[str],
    texts: dict[str, str],
    model: str,
    confirm: Callable[[Estimate], bool],
    max_reports: int,
    reasoning_effort: str | None = "low",
    concurrency: int = 16,
    llm=None,
) -> FallbackResult:
    res = FallbackResult(question)
    est = estimate(question, ids, texts, model, reasoning_effort)
    res.estimate = est
    if est.calls > max_reports:
        res.skipped_reason = (
            f"{est.calls} reports need an LLM check, above the cap of {max_reports}. "
            "Narrow the question or raise --max-fallback."
        )
        return res
    if est.calls and not confirm(est):
        res.skipped_reason = "not confirmed; no API calls made"
        return res

    cache = _load_cache()
    todo = [i for i in ids if _key(question, model, i, reasoning_effort) not in cache]
    if todo:
        if llm is None:
            llm = build_chat_model(model, reasoning_effort).with_structured_output(
                Verdict, method="json_schema", strict=True, include_raw=True
            )
        msgs = [
            [SystemMessage(SYSTEM), HumanMessage(f"Question: {question}\n\nReport:\n{texts[i]}")] for i in todo
        ]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CACHE.open("a") as fh:
            for idx, out in llm.batch_as_completed(msgs, config={"max_concurrency": concurrency},
                                                   return_exceptions=True):
                rid = todo[idx]
                res.usage["calls"] += 1
                if not isinstance(out, Exception):
                    u = getattr(out.get("raw"), "usage_metadata", None) or {}
                    res.usage["input_tokens"] += u.get("input_tokens") or 0
                    res.usage["output_tokens"] += u.get("output_tokens") or 0
                if isinstance(out, Exception) or out.get("parsed") is None:
                    res.failed.append(rid)  # not cached: a re-run would send (and bill) it again
                    continue
                v: Verdict = out["parsed"]
                rec = {"key": _key(question, model, rid, reasoning_effort), "id": rid, "question": question, "model": model,
                       "matches": v.matches, "evidence_quote": v.evidence_quote}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                cache[rec["key"]] = rec

    for rid in ids:
        rec = cache.get(_key(question, model, rid, reasoning_effort))
        if rec is None:
            continue
        res.evidence[rid] = rec["evidence_quote"]
        if not rec["matches"]:
            res.rejected.append(rid)
        elif quote_in_text(rec["evidence_quote"], texts[rid]):
            res.matched.append(rid)
        else:
            res.unverified.append(rid)
    return res
