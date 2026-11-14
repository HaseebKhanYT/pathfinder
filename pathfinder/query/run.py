"""Question -> rule -> answer over the saved extractions, shown with the rule.

Steps: (1) translate the question into a rule (one LLM call, plus repairs),
(2) validate it, (3) run the engine over the saved records (no API calls),
(4) if part of the question is outside the schema, ask the LLM yes/no per
remaining report, only after confirmation, (5) print the rule with the answer.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from ..criteria.engine import evaluate
from .catalog import Field, build_catalog
from .fallback import Estimate, FallbackResult, run_fallback
from .translate import Translation, translate
from .validate import validate_rule


@dataclass
class QueryResult:
    question: str
    rule: dict | list | None
    interpretation: str
    answer_type: str
    breakdown_by: str | None
    uncovered: str | None
    ids: list[str] = field(default_factory=list)  # final matches (after fallback, if any)
    rule_ids: list[str] = field(default_factory=list)  # matches of the rule alone
    breakdown: list[tuple] = field(default_factory=list)
    translation: Translation | None = None
    fallback: FallbackResult | None = None
    error: str | None = None

    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def answered(self) -> bool:
        """False when the uncovered part was not checked: there is no final answer."""
        return not (self.fallback and self.fallback.skipped_reason)

    @property
    def complete(self) -> bool:
        """False when some fallback calls failed: the count is a lower bound."""
        return self.answered and not (self.fallback and self.fallback.failed)


def breakdown(ids: list[str], records: dict[str, dict], by: str) -> list[tuple]:
    c = Counter()
    for i in ids:
        v = records[i].get(by)
        for item in (v if isinstance(v, list) and v else [v]):
            c["(none)" if item in (None, []) else item] += 1
    return c.most_common()


def run_query(
    question: str,
    records: dict[str, dict],
    texts: dict[str, str],
    model: str,
    confirm: Callable[[Estimate], bool],
    max_fallback: int,
    reasoning_effort: str | None = "low",
    rule: dict | list | None = None,
    answer_type: str = "ids",
    breakdown_by: str | None = None,
    uncovered: str | None = None,
    catalog: dict[str, Field] | None = None,
    translator_llm=None,
    fallback_llm=None,
) -> QueryResult:
    catalog = catalog or build_catalog(records)
    t = None
    if rule is None:
        t = translate(question, catalog, model, reasoning_effort, llm=translator_llm)
        if not t.ok:
            last = t.attempts[-1]["errors"] if t.attempts else ["no attempt"]
            return QueryResult(question, None, "", "ids", None, None, translation=t,
                               error="could not produce a valid rule: " + "; ".join(last))
        d = t.draft
        rule, interpretation, answer_type, breakdown_by, uncovered = (
            t.rule, d.interpretation, d.answer, d.breakdown_by, d.uncovered)
    else:
        errors = validate_rule(rule, catalog)
        if answer_type == "breakdown" and not breakdown_by:
            errors.append("answer 'breakdown' needs a field to group by (--by)")
        if breakdown_by and breakdown_by not in catalog:
            errors.append(f"breakdown_by {breakdown_by!r} is not a catalog field")
        if errors:
            return QueryResult(question, rule, "", answer_type, breakdown_by, None,
                               error="invalid rule: " + "; ".join(errors))
        interpretation = "rule supplied directly"

    rule_ids = sorted(rid for rid, rec in records.items() if evaluate(rule, rec))
    res = QueryResult(question, rule, interpretation, answer_type, breakdown_by, uncovered,
                      ids=rule_ids, rule_ids=rule_ids, translation=t)
    if uncovered:
        fb = run_fallback(uncovered, rule_ids, texts, model, confirm, max_fallback, reasoning_effort,
                          llm=fallback_llm)
        res.fallback = fb
        res.ids = [] if fb.skipped_reason else sorted(fb.matched)
    if answer_type == "breakdown" and breakdown_by:
        res.breakdown = breakdown(res.ids, records, breakdown_by)
    return res


def to_dict(res: QueryResult) -> dict:
    """The saved form of a result (CLI --out, UI download). No final answer -> nulls."""
    return {
        "question": res.question, "interpretation": res.interpretation, "rule": res.rule,
        "uncovered": res.uncovered,
        "answered": res.answered, "complete": res.complete,
        "fallback_skipped": res.fallback.skipped_reason if res.fallback else None,
        "fallback_failed": res.fallback.failed if res.fallback else [],
        "rule_ids": res.rule_ids,
        "count": res.count if res.answered else None,
        "ids": res.ids if res.answered else None,
        "breakdown": res.breakdown if res.answered else None,
        "evidence": res.fallback.evidence if res.fallback else {},
    }


def format_rule(rule, indent: int = 0) -> str:
    """One condition per line, groups indented: easier to read than json indent."""
    pad = "  " * indent
    if isinstance(rule, list):
        return pad + json.dumps(rule)
    (key, body), = rule.items()
    if key == "not":
        return f"{pad}NOT\n{format_rule(body, indent + 1)}"
    return f"{pad}{key.upper()} of:\n" + "\n".join(format_rule(r, indent + 1) for r in body)


def render(res: QueryResult, max_ids: int = 50) -> str:
    out = [f"Question: {res.question}"]
    t = res.translation
    if t:
        fixes = len(t.attempts) - 1
        out.append(f"Translation: {t.usage['calls']} LLM call(s), {t.usage['input_tokens']:,} in / "
                   f"{t.usage['output_tokens']:,} out tokens" + (f", {fixes} repair(s)" if fixes else ""))
        for i, a in enumerate(t.attempts[:-1] if t.ok else t.attempts, 1):
            out.append(f"  attempt {i} rejected: " + "; ".join(a["errors"]))
    if res.error:
        out.append(f"ERROR: {res.error}")
        return "\n".join(out)
    out.append(f"Read as: {res.interpretation}")
    out.append("Rule:\n" + format_rule(res.rule, 1))
    if res.uncovered:
        out.append(f"Outside the schema: {res.uncovered}")
        out.append(f"  The rule alone (covered part) matches {len(res.rule_ids)} report(s).")
        fb = res.fallback
        if fb and fb.estimate:
            e = fb.estimate
            out.append(f"  Fallback: {e.reports} candidate(s), {e.cached} cached, {e.calls} needing a call "
                       f"(~{e.input_tokens:,} in / ~{e.output_tokens:,} out tokens estimated).")
        if fb and fb.skipped_reason:
            out.append(f"  Fallback NOT run: {fb.skipped_reason}")
            out.append("  No final answer: the uncovered part was not checked.")
            return "\n".join(out)
        if fb:
            out.append(f"  Fallback used {fb.usage['calls']} call(s), {fb.usage['input_tokens']:,} in / "
                       f"{fb.usage['output_tokens']:,} out tokens. yes={len(fb.matched)} "
                       f"no={len(fb.rejected)} unverified-quote={len(fb.unverified)} failed={len(fb.failed)}")
    out.append(f"Count: {res.count}" + ("" if res.complete else
               f"  (LOWER BOUND: {len(res.fallback.failed)} report(s) could not be checked)"))
    if res.answer_type == "breakdown" and res.breakdown_by:
        out.append(f"Breakdown by {res.breakdown_by}:")
        for value, n in res.breakdown:
            out.append(f"  {str(value):40} {n}")
    elif res.answer_type == "ids":
        shown = res.ids[:max_ids]
        out.append("Ids: " + (", ".join(shown) if shown else "(none)")
                   + (f" ... (+{len(res.ids) - max_ids} more)" if len(res.ids) > max_ids else ""))
    if res.fallback and res.fallback.matched:
        out.append("Evidence (first 5):")
        for rid in res.fallback.matched[:5]:
            out.append(f"  {rid}: \"{res.fallback.evidence.get(rid, '')}\"")
    if res.uncovered and res.translation:
        out.append("Re-run this exact query (no translation; cached verdicts are free):\n"
                   f"  query {json.dumps(res.question)} --rule '{json.dumps(res.rule)}' "
                   f"--uncovered {json.dumps(res.uncovered)} --answer {res.answer_type}"
                   + (f" --by {res.breakdown_by}" if res.breakdown_by else ""))
    if res.fallback and res.fallback.unverified:
        out.append(f"Unverified (said yes, quote not found in text, excluded): {', '.join(res.fallback.unverified[:10])}")
    return "\n".join(out)
