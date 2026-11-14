"""Turn a nested `ReportExtraction` dict into the flat record the rules read.

Everything that needs arithmetic or normalisation lives here, in code, so the
LLM only ever copies what the report says.

Flat field names (the vocabulary `criteria_rules.json` may use):

  header:    sex, age_band, age_min, age_max, report_year
  specimen:  organ, site, laterality, procedure
  tumor:     histologic_type, histologic_category, differentiation_grade,
             tumor_size_value, tumor_size_unit, tumor_size_mm, tumor_size_text,
             lymphovascular_invasion, perineural_invasion, margin_status,
             closest_margin_mm, involved_margins, lymph_nodes_positive,
             lymph_nodes_examined, node_status, pT, pN, pT_base, pN_base
  organ blocks, prefixed: breast.er_status, prostate.grade_group,
             kidney.extent_of_invasion, skin.group, ...
  her2_status: breast or stomach HER2, whichever applies
"""

from __future__ import annotations

import re

ORGAN_BLOCKS = ("breast", "lung", "colon", "stomach", "prostate", "kidney", "skin")


def _age_bounds(band: str | None) -> tuple[int | None, int | None]:
    if not band:
        return None, None
    m = re.match(r"\s*(\d+)\s*-\s*(\d+)", band)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"\s*(\d+)\s*\+", band)
    if m:
        return int(m.group(1)), None
    return None, None


def _size_mm(value, unit) -> float | None:
    if value is None or unit is None:
        return None
    return round(float(value) * (10.0 if unit == "cm" else 1.0), 3)


def _stage_base(stage: str | None, prefix: str) -> str | None:
    """'pT1c(m)' -> 'pT1c'; 'pN0(sn)' -> 'pN0'; 'pN1a(sn)' -> 'pN1a'."""
    if not stage:
        return None
    m = re.match(rf"\s*(y?{prefix}[0-9A-Za-z]*?(?:is)?)(?:\(|\s|$)", stage)
    return m.group(1) if m else stage.strip()


def derive(extraction: dict) -> dict:
    """Return the flat record for one extraction."""
    h, s, t = extraction["header"], extraction["specimen"], extraction["tumor"]
    age_min, age_max = _age_bounds(h.get("age_band"))
    pos = t.get("lymph_nodes_positive")
    if pos is None:
        node_status = "not_examined"
    else:
        node_status = "positive" if pos >= 1 else "negative"

    flat: dict = {
        "sex": h.get("sex"),
        "age_band": h.get("age_band"),
        "age_min": age_min,
        "age_max": age_max,
        "report_year": h.get("report_year"),
        "organ": s.get("organ"),
        "site": s.get("site"),
        "laterality": s.get("laterality"),
        "procedure": s.get("procedure"),
        **{k: v for k, v in t.items()},
        # Consistency guard: a model that says margins are negative but still lists
        # margin names has copied the margin names, not involvement (seen on train).
        "involved_margins": (
            []
            if t.get("margin_status") == "negative"
            else [m.lower().strip() for m in (t.get("involved_margins") or [])]
        ),
        "tumor_size_mm": _size_mm(t.get("tumor_size_value"), t.get("tumor_size_unit")),
        "tumor_size_text": (
            f"{t['tumor_size_value']:g} {t['tumor_size_unit']}"
            if t.get("tumor_size_value") is not None and t.get("tumor_size_unit")
            else None
        ),
        "node_status": node_status,
        "pT_base": _stage_base(t.get("pT"), "pT"),
        "pN_base": _stage_base(t.get("pN"), "pN"),
        "her2_status": None,
    }
    for organ in ORGAN_BLOCKS:
        block = extraction.get(organ) or {}
        for k, v in block.items():
            flat[f"{organ}.{k}"] = v
    if extraction.get("breast"):
        flat["her2_status"] = extraction["breast"].get("her2_status")
    elif extraction.get("stomach"):
        flat["her2_status"] = extraction["stomach"].get("her2_status")
    return flat
