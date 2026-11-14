"""Gold `ReportExtraction` dicts built from train `labels` (no LLM).

Two uses:
1. Run the criteria rules on gold records to check the RULES alone against
   `criteria_train_answers.json` (any error there is a rule bug, not an
   extraction bug).
2. Compare LLM extractions field by field against gold.

Parsing covers the field vocabulary observed in train.jsonl; anything
unrecognised is left null rather than guessed.
"""

from __future__ import annotations

import re

STATUS_PRESENT = re.compile(r"^\s*present", re.I)
STATUS_ABSENT = re.compile(r"^\s*(not identified|absent|negative)", re.I)


def _status(v):
    if v is None:
        return "not_reported"
    if STATUS_PRESENT.match(v):
        return "present"
    if STATUS_ABSENT.match(v):
        return "not_identified"
    return "not_reported"


def _num(pattern, text, cast=float):
    if not text:
        return None
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def _receptor(text, name):
    m = re.search(rf"{name} (low positive|positive|negative|equivocal)", text or "")
    return m.group(1).replace(" ", "_") if m else "not_reported"


def _grade_g(text):
    return _num(r"G([123])", text, int)


def _margins(organ, text):
    text = text or ""
    low = text.lower()
    involved = []
    for m in re.finditer(r"(?:^|; |\b)([a-z ()]+?) margins? (?:involved|positive)", low):
        involved.append(re.sub(r"\s*\(.*?\)", "", m.group(1)).strip())
    m = re.search(r"involves the inked (\w+) margin", low)
    if m:
        involved.append(m.group(1))
    m = re.search(r"positive at the ([a-z ]+?) margin", low)
    if m:
        involved.append(m.group(1))
    m = re.search(r"extends to the ((?:deep|peripheral)(?: and (?:deep|peripheral))?) margin", low)
    if m:
        involved.extend(x.strip() for x in m.group(1).split(" and "))
    involved = [i.split(";")[-1].strip() for i in involved]
    involved = [part.strip() for i in involved for part in re.split(r" and |, ", i) if part.strip()]
    positive = bool(involved) or "involved" in low or low.startswith("positive")
    if positive:
        status = "positive"
    elif "negative" in low or "completely excised" in low:
        status = "negative"
    else:
        status = "not_reported"
    closest = None
    if status == "negative":
        closest = _num(r"invasive carcinoma is ([\d.]+) mm from the closest margin", text) or _num(
            r"closest distance ([\d.]+) mm", text
        )
    return status, closest, involved


KIDNEY_EXTENT = [
    ("inferior vena cava above", "ivc_above_diaphragm"),
    ("inferior vena cava below", "ivc_below_diaphragm"),
    ("segmental branches", "renal_vein_segmental_branches"),
    ("main renal vein", "main_renal_vein"),
    ("renal sinus fat", "renal_sinus_fat"),
    ("perinephric fat", "perinephric_fat"),
    ("gerota", "beyond_gerota_fascia"),
    ("adrenal", "ipsilateral_adrenal"),
    ("confined", "confined_to_kidney"),
]


def _histo_category(organ, h):
    h = (h or "").lower()
    if organ == "breast":
        return "invasive_lobular" if "lobular" in h else "invasive_ductal_nst"
    if organ == "colon":
        return "mucinous_adenocarcinoma" if "mucinous" in h else "adenocarcinoma"
    if organ == "lung":
        if "squamous" in h:
            return "squamous_cell_carcinoma"
        return "lung_adenocarcinoma" if "adenocarcinoma" in h else "lung_other"
    if organ == "stomach":
        if h.startswith("mixed"):
            return "mixed_glandular_poorly_cohesive"
        if h.startswith("poorly cohesive"):
            return "poorly_cohesive_diffuse"
        return "intestinal_tubular_or_nos_adenocarcinoma"
    if organ == "kidney":
        return "clear_cell_rcc" if "clear cell" in h else "other_rcc"
    if organ == "prostate":
        return "acinar_adenocarcinoma"
    return None


def gold_extraction(labels: dict) -> dict:
    d = labels["diagnosis"]
    organ = labels["organ"]
    size = d.get("Tumor size")
    m = re.match(r"([\d.]+)\s*(mm|cm)", size or "")
    margin_status, closest, involved = _margins(organ, d.get("Margins"))
    pos = exam = None
    ln = d.get("Lymph nodes")
    if ln is None:
        sn = [v for k, v in d.items() if k.startswith("Part B, sentinel")]
        ln = sn[0] if sn else None
    if ln:
        mm = re.search(r"(\d+) of (\d+)", ln)
        if mm:
            pos, exam = int(mm.group(1)), int(mm.group(2))
    histo = d.get("Histologic type") or (d.get("Diagnosis") if organ == "skin" else None)
    out = {
        "header": {"sex": labels["sex"], "age_band": labels["age_band"], "report_year": labels["report_year"]},
        "specimen": {"organ": organ, "site": None, "laterality": "not_applicable", "procedure": d.get("Specimen")},
        "tumor": {
            "histologic_type": histo,
            "histologic_category": _histo_category(organ, histo),
            "differentiation_grade": _grade_g(d.get("Histologic grade") or (histo if organ == "skin" else None)),
            "tumor_size_value": float(m.group(1)) if m else None,
            "tumor_size_unit": m.group(2) if m else None,
            "lymphovascular_invasion": _status(d.get("Lymphovascular invasion")),
            "perineural_invasion": _status(d.get("Perineural invasion")),
            "margin_status": margin_status,
            "closest_margin_mm": closest,
            "involved_margins": involved,
            "lymph_nodes_positive": pos,
            "lymph_nodes_examined": exam,
            "pT": labels.get("pT"),
            "pN": labels.get("pN"),
        },
        **{k: None for k in ("breast", "lung", "colon", "stomach", "prostate", "kidney", "skin")},
    }
    if organ == "breast":
        bm = d.get("Biomarkers")
        out["breast"] = {
            "nottingham_grade": _num(r"^(\d)", d.get("Nottingham grade"), int),
            "er_status": _receptor(bm, "ER"),
            "pr_status": _receptor(bm, "PR"),
            "her2_status": _receptor(bm, "HER2"),
            "tumor_focality": "multifocal" if "multifocal" in (d.get("Tumor focality") or "") else "unifocal",
            "dcis_present": _status(d.get("Ductal carcinoma in situ")) == "present",
        }
    elif organ == "lung":
        out["lung"] = {"visceral_pleural_invasion": _status(d.get("Visceral pleural invasion"))}
    elif organ == "colon":
        out["colon"] = {
            "tumor_deposits": _status(d.get("Tumor deposits")),
            "macroscopic_perforation": _status(d.get("Macroscopic tumor perforation")),
            "depth_of_invasion": d.get("Depth of invasion"),
        }
    elif organ == "stomach":
        her2 = (d.get("HER2") or "").split(" ")[0]
        if "not amplified" in (d.get("HER2") or ""):
            her2 = "negative"
        out["stomach"] = {"her2_status": her2 or "not_reported", "depth_of_invasion": d.get("Depth of invasion")}
    elif organ == "prostate":
        g = d.get("Gleason score") or ""
        mm = re.match(r"(\d)\+(\d)", g)
        epe = (d.get("Extraprostatic extension") or "").lower()
        out["prostate"] = {
            "gleason_primary": int(mm.group(1)) if mm else None,
            "gleason_secondary": int(mm.group(2)) if mm else None,
            "grade_group": _num(r"Grade Group (\d)", g, int),
            "extraprostatic_extension": "focal" if epe.startswith("focal") else "established" if epe.startswith("established") else "not_identified",
            "seminal_vesicle_invasion": _status(d.get("Seminal vesicle invasion")),
            "cribriform_glands": _status(d.get("Cribriform glands")),
            "intraductal_carcinoma": _status(d.get("Intraductal carcinoma")),
        }
    elif organ == "kidney":
        ext = (d.get("Extent of invasion") or "").lower()
        out["kidney"] = {
            "isup_grade": _num(r"(\d)", d.get("ISUP grade"), int),
            "sarcomatoid_or_rhabdoid": _status(d.get("Sarcomatoid or rhabdoid features")),
            "extent_of_invasion": next((v for k, v in KIDNEY_EXTENT if k in ext), None),
        }
    elif organ == "skin":
        out["skin"] = {
            "group": labels.get("group"),
            "breslow_thickness_mm": _num(r"([\d.]+) mm", d.get("Breslow thickness")),
            "ulceration": _status(d.get("Ulceration")),
        }
    return out
