"""The field catalog: every flat field a rule may use, with its type and allowed values.

Built by introspecting `ReportExtraction` (enum values come straight from the
schema's `Literal`s, so the catalog cannot drift from what the extractor
produces) plus the fields `postprocess.derive` computes. For free-text fields
the catalog also lists the values actually seen in the saved records, so the
LLM writes values that exist (e.g. "pT4a", not "T4a").
"""

from __future__ import annotations

import typing
from collections import Counter
from dataclasses import dataclass, field

from pydantic import BaseModel

from ..extraction.postprocess import ORGAN_BLOCKS
from ..extraction.schema import ReportExtraction

NUMERIC_OPS = {"eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "is_null", "not_null"}
ENUM_OPS = {"eq", "ne", "in", "not_in", "is_null", "not_null"}
STRING_OPS = {"eq", "ne", "in", "not_in", "startswith", "contains", "is_null", "not_null"}
LIST_OPS = {"contains", "is_null", "not_null"}
BOOL_OPS = {"eq", "ne", "is_null", "not_null"}


@dataclass
class Field:
    name: str
    kind: str  # "enum" | "int" | "float" | "string" | "list" | "bool"
    description: str = ""
    allowed: list | None = None  # enum values
    seen: list = field(default_factory=list)  # most common observed values, shown to the LLM
    seen_all: set = field(default_factory=set)  # every observed value, used by the validator

    @property
    def ops(self) -> set[str]:
        return {"enum": ENUM_OPS, "int": NUMERIC_OPS, "float": NUMERIC_OPS, "string": STRING_OPS,
                "list": LIST_OPS, "bool": BOOL_OPS}[self.kind]


def _unwrap(tp):
    """Optional[X] -> X."""
    if typing.get_origin(tp) is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        return args[0] if len(args) == 1 else tp
    return tp


def _kind(tp) -> tuple[str, list | None]:
    tp = _unwrap(tp)
    origin = typing.get_origin(tp)
    if origin is typing.Literal:
        return "enum", list(typing.get_args(tp))
    if origin is list:
        return "list", None
    return {int: "int", float: "float", str: "string", bool: "bool"}.get(tp, "string"), None


def _model_fields(model: type[BaseModel], prefix: str = "") -> list[Field]:
    out = []
    for name, info in model.model_fields.items():
        kind, allowed = _kind(info.annotation)
        out.append(Field(prefix + name, kind, info.description or "", allowed))
    return out


# Fields computed in postprocess.derive (not asked of the model).
DERIVED = [
    Field("age_min", "int", "Lower bound of the age band (e.g. 60 for '60-69', 90 for '90+')."),
    Field("age_max", "int", "Upper bound of the age band (69 for '60-69'); null for '90+'."),
    Field("tumor_size_mm", "float", "Tumor size converted to mm from value+unit as written."),
    Field("tumor_size_text", "string", "Tumor size as written, e.g. '2.5 cm'."),
    Field("node_status", "enum", "From lymph_nodes_positive.", ["positive", "negative", "not_examined"]),
    Field("pT_base", "string", "pT without suffixes: 'pT1c(m)' -> 'pT1c'."),
    Field("pN_base", "string", "pN without suffixes: 'pN0(sn)' -> 'pN0', 'pN1a(sn)' -> 'pN1a'."),
    Field("her2_status", "enum", "Breast or stomach HER2, whichever applies.",
          ["positive", "low_positive", "negative", "equivocal", "not_reported"]),
]


def build_catalog(records: dict[str, dict] | None = None, top_n: int = 25) -> dict[str, Field]:
    hints = typing.get_type_hints(ReportExtraction)
    fields: list[Field] = []
    for section in ("header", "specimen", "tumor"):
        fields += _model_fields(_unwrap(hints[section]))
    for organ in ORGAN_BLOCKS:
        fields += _model_fields(_unwrap(hints[organ]), prefix=f"{organ}.")
    fields += DERIVED
    catalog = {f.name: f for f in fields}
    if records:
        for f in catalog.values():
            if f.kind in ("string", "list"):
                c = Counter()
                for rec in records.values():
                    v = rec.get(f.name)
                    for item in (v if isinstance(v, list) else [v]):
                        if item is not None:
                            c[item] += 1
                f.seen = [v for v, _ in c.most_common(top_n)]
                f.seen_all = set(c)
    return catalog


def render_catalog(catalog: dict[str, Field]) -> str:
    """Compact text listing for the prompt."""
    lines = []
    for f in catalog.values():
        s = f"- {f.name} ({f.kind})"
        if f.allowed:
            s += f" allowed: {', '.join(map(repr, f.allowed))}"
        elif f.seen:
            s += f" e.g.: {', '.join(map(repr, f.seen[:12]))}"
        if f.description:
            s += f" — {f.description}"
        lines.append(s)
    return "\n".join(lines)
