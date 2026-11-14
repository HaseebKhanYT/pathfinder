"""Tiny declarative rule engine for `criteria_rules.json`.

A rule is either a condition triple `[field, op, value]` or a group
`{"all": [...]}` / `{"any": [...]}` / `{"not": rule}`. A criterion matches a
report when its top-level rule is true for that report's flat record.

A missing/null field makes every comparison false (except `is_null`), so a
report without, say, a tumor size never satisfies `tumor_size_mm > 20`.
"""

from __future__ import annotations

import json
from pathlib import Path

RULES_PATH = Path(__file__).with_name("criteria_rules.json")


def _cmp(actual, op: str, expected) -> bool:
    if op == "is_null":
        return actual is None
    if op == "not_null":
        return actual is not None
    if actual is None:
        return False
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "in":
        return actual in expected
    if op == "not_in":
        return actual not in expected
    if op == "gt":
        return actual > expected
    if op == "gte":
        return actual >= expected
    if op == "lt":
        return actual < expected
    if op == "lte":
        return actual <= expected
    if op == "startswith":
        return isinstance(actual, str) and actual.startswith(tuple(expected) if isinstance(expected, list) else expected)
    if op == "contains":  # substring of a string, or element of a list
        if isinstance(actual, list):
            return any(expected in str(a) for a in actual)
        return expected.lower() in str(actual).lower()
    raise ValueError(f"unknown op {op!r}")


def evaluate(rule, record: dict) -> bool:
    if isinstance(rule, list):
        field, op, *rest = rule
        return _cmp(record.get(field), op, rest[0] if rest else None)
    if "all" in rule:
        return all(evaluate(r, record) for r in rule["all"])
    if "any" in rule:
        return any(evaluate(r, record) for r in rule["any"])
    if "not" in rule:
        return not evaluate(rule["not"], record)
    raise ValueError(f"bad rule {rule!r}")


def load_rules(path: Path = RULES_PATH) -> dict:
    return json.loads(path.read_text())


def match_all(records: dict[str, dict], rules: dict | None = None) -> dict[str, list[str]]:
    """{criterion id: sorted matching report ids} for flat records keyed by id."""
    rules = rules or load_rules()
    return {
        cid: sorted(rid for rid, rec in records.items() if evaluate(spec["rule"], rec))
        for cid, spec in rules.items()
    }
