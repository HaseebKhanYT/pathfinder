"""Question -> validated rule, via an LLM with a repair loop.

The LLM sees the rule grammar, the field catalog (with allowed values), and the
30 hand-written criteria as worked examples, so it inherits their conventions
(e.g. "stage III" colon = positive nodes OR pN1/pN2; "ER low positive" is
neither ER+ nor ER-). Its rule is validated; any errors go back to it to fix,
up to `max_repairs` times.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field as PField

from ..criteria.engine import load_rules
from ..extraction.extractor import build_chat_model
from .catalog import Field, render_catalog
from .validate import validate_rule


class RuleDraft(BaseModel):
    interpretation: str = PField(description="One or two sentences: how you read the question, in plain English.")
    rule_json: str = PField(description="The rule as a JSON string in the engine format.")
    answer: Literal["ids", "count", "breakdown"] = PField(
        description="ids: list matching patients; count: how many; breakdown: counts grouped by a field."
    )
    breakdown_by: Optional[str] = PField(description="Field to group by when answer is breakdown, else null.")
    uncovered: Optional[str] = PField(
        description=(
            "The part of the question NO catalog field can express, phrased as a yes/no question about one "
            "report (e.g. 'Does the report state MMR deficiency?'). Null when the rule covers the whole question."
        )
    )


GRAMMAR = """\
# Rule format (JSON)
A rule is a condition or a group.
- condition: [field, op, value]  or  [field, "is_null"]  or  [field, "not_null"]
- group: {"all": [rules...]}  |  {"any": [rules...]}  |  {"not": rule}
Ops: eq, ne, in, not_in (value is a list), gt, gte, lt, lte (numbers), startswith (text; value may be a list
of prefixes), contains (substring of text, or element-substring of a list field such as involved_margins).
A null field fails every comparison except is_null.

# Conventions
- Always pin the organ when the question names one: ["organ", "eq", "colon"].
- Use only fields and values from the catalog. Enum values must be copied exactly.
- Status fields: "present" / "not_identified" / "not_reported". "No X" means "not_identified" (an explicit
  negative), as in the examples, not "ne present".
- Sizes: use tumor_size_mm (already converted to mm). "5 cm or larger" -> ["tumor_size_mm", "gte", 50].
- Ages come only as bands: "under 60" -> ["age_max", "lt", 60]; "70 or older" -> ["age_min", "gte", 70].
- Positive nodes: ["lymph_nodes_positive", "gte", 1]. pT4 etc.: ["pT", "startswith", "pT4"].
- If part of the question cannot be expressed with catalog fields (e.g. MMR status, PD-L1, smoking history,
  necrosis, anything not in the catalog), DO NOT invent a field. Encode only the covered part (at least
  the organ) in the rule and put the rest in `uncovered` as a yes/no question about a single report.
- answer: "count" for "how many"; "breakdown" for "by/per/broken down by <field>" (set breakdown_by to a
  catalog field); otherwise "ids".
"""


def _examples(exclude: set[str]) -> str:
    rows = []
    for cid, spec in load_rules().items():
        if cid in exclude:
            continue
        rows.append(f'Q: {spec["text"]}\nRULE: {json.dumps(spec["rule"])}')
    return "\n\n".join(rows)


def system_prompt(catalog: dict[str, Field], exclude_examples: set[str] = frozenset()) -> str:
    return (
        "You translate a cohort question about pathology reports into a rule that a rule engine runs over "
        "already-extracted report records.\n\n"
        f"{GRAMMAR}\n# Field catalog\n{render_catalog(catalog)}\n\n"
        f"# Worked examples (house conventions)\n{_examples(set(exclude_examples))}\n"
    )


@dataclass
class Translation:
    question: str
    draft: RuleDraft | None
    rule: dict | list | None
    attempts: list[dict] = field(default_factory=list)  # [{"rule_json", "errors"}]
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0})

    @property
    def ok(self) -> bool:
        return self.rule is not None


def translate(
    question: str,
    catalog: dict[str, Field],
    model: str,
    reasoning_effort: str | None = "low",
    max_repairs: int = 3,
    exclude_examples: set[str] = frozenset(),
    llm=None,
) -> Translation:
    """`llm` may be injected (anything with .invoke(messages) -> {"parsed", "raw"}) for offline tests."""
    if llm is None:
        llm = build_chat_model(model, reasoning_effort).with_structured_output(
            RuleDraft, method="json_schema", strict=True, include_raw=True
        )
    messages = [SystemMessage(system_prompt(catalog, exclude_examples)), HumanMessage(f"Question: {question}")]
    t = Translation(question, None, None)
    for _ in range(max_repairs + 1):
        res = llm.invoke(messages)
        t.usage["calls"] += 1
        usage = getattr(res.get("raw"), "usage_metadata", None) or {}
        t.usage["input_tokens"] += usage.get("input_tokens") or 0
        t.usage["output_tokens"] += usage.get("output_tokens") or 0
        draft: RuleDraft | None = res.get("parsed")
        if draft is None:
            errors = [f"output did not match the schema: {res.get('parsing_error')!r}"]
            rule = None
            raw_json = ""
        else:
            raw_json = draft.rule_json
            errors, rule = _check_draft(draft, catalog)
        t.attempts.append({"rule_json": raw_json, "errors": errors})
        if not errors:
            t.draft, t.rule = draft, rule
            return t
        messages += [
            AIMessage(draft.model_dump_json() if draft else "(unparseable)"),
            HumanMessage("Your rule failed validation:\n- " + "\n- ".join(errors) + "\nReturn a corrected answer."),
        ]
    t.draft = draft
    return t


def _check_draft(draft: RuleDraft, catalog: dict[str, Field]) -> tuple[list[str], dict | list | None]:
    try:
        rule = json.loads(draft.rule_json)
    except json.JSONDecodeError as e:
        return [f"rule_json is not valid JSON: {e}"], None
    errors = validate_rule(rule, catalog)
    if draft.answer == "breakdown":
        if not draft.breakdown_by:
            errors.append("answer is 'breakdown' but breakdown_by is null")
        elif draft.breakdown_by not in catalog:
            errors.append(f"breakdown_by {draft.breakdown_by!r} is not a catalog field")
    return errors, (rule if not errors else None)
