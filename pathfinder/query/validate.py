"""Static checks on a rule before it runs.

Every problem becomes one plain-English error string; the translator sends
these back to the LLM to repair. An empty list means the rule is valid.
"""

from __future__ import annotations

from .catalog import Field

GROUP_KEYS = {"all", "any", "not"}


def validate_rule(rule, catalog: dict[str, Field], path: str = "rule") -> list[str]:
    errors: list[str] = []
    if isinstance(rule, list):
        errors += _check_condition(rule, catalog, path)
    elif isinstance(rule, dict):
        keys = set(rule)
        if len(keys) != 1 or not keys <= GROUP_KEYS:
            return [f"{path}: a group must have exactly one key out of all/any/not, got {sorted(keys)}"]
        (key,) = keys
        body = rule[key]
        if key == "not":
            errors += validate_rule(body, catalog, f"{path}.not")
        elif not isinstance(body, list) or not body:
            errors.append(f"{path}.{key}: must be a non-empty list of rules")
        else:
            for i, sub in enumerate(body):
                errors += validate_rule(sub, catalog, f"{path}.{key}[{i}]")
    else:
        errors.append(f"{path}: expected a condition [field, op, value] or a group, got {type(rule).__name__}")
    return errors


def _check_condition(cond: list, catalog: dict[str, Field], path: str) -> list[str]:
    if len(cond) not in (2, 3) or not isinstance(cond[0], str) or not isinstance(cond[1], str):
        return [f"{path}: a condition is [field, op, value] (or [field, 'is_null'/'not_null']), got {cond!r}"]
    name, op = cond[0], cond[1]
    f = catalog.get(name)
    if f is None:
        near = [k for k in catalog if name.split(".")[-1] in k][:5]
        hint = f"; did you mean one of {near}?" if near else ""
        return [f"{path}: unknown field {name!r}{hint}"]
    if op not in f.ops:
        return [f"{path}: op {op!r} is not valid for {f.kind} field {name!r}; use one of {sorted(f.ops)}"]
    if op in ("is_null", "not_null"):
        return [] if len(cond) == 2 else [f"{path}: {op!r} takes no value"]
    if len(cond) != 3:
        return [f"{path}: op {op!r} needs a value"]
    value = cond[2]
    if op in ("in", "not_in"):
        if not isinstance(value, list) or not value:
            return [f"{path}: op {op!r} needs a non-empty list"]
        values = value
    elif op == "startswith" and isinstance(value, list):
        if not value:
            return [f"{path}: op 'startswith' needs a non-empty prefix list"]
        values = value
    else:
        values = [value]
    errs = []
    for v in values:
        errs += _check_value(f, op, v, path)
    return errs


def _check_value(f: Field, op: str, v, path: str) -> list[str]:
    if f.kind == "enum" and v not in f.allowed:
        return [f"{path}: {v!r} is not an allowed value of {f.name!r}; allowed: {f.allowed}"]
    if f.kind in ("int", "float"):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return [f"{path}: {f.name!r} is numeric; got {v!r}"]
        if f.kind == "int" and isinstance(v, float) and not v.is_integer():
            return [f"{path}: {f.name!r} is an integer field; got {v!r}"]
    if f.kind == "bool" and not isinstance(v, bool):
        return [f"{path}: {f.name!r} is true/false; got {v!r}"]
    if f.kind in ("string", "list") and not isinstance(v, str):
        return [f"{path}: {f.name!r} compares against text; got {v!r}"]
    if f.kind == "list" and op == "contains" and f.seen_all and not any(v in s for s in f.seen_all):
        # The engine matches list elements case-sensitively; a value that is a
        # substring of no stored element can never match.
        return [f"{path}: {v!r} is not part of any value of {f.name!r}; values include {f.seen[:12]}"]
    if f.kind == "string" and op in ("eq", "in") and f.seen_all and v not in f.seen_all:
        # Exact matches on free text only work for values that exist.
        return [f"{path}: {v!r} never occurs in {f.name!r}; seen values include {f.seen[:12]}. "
                "Use an existing value, or 'startswith'/'contains'."]
    return []
