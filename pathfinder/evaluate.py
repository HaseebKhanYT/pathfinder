"""Scoring: per-criterion precision/recall/F1 and field-level accuracy."""

from __future__ import annotations


def prf(pred: set, gold: set) -> tuple[float, float, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else (1.0 if not pred else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score_criteria(pred: dict[str, list[str]], gold: dict[str, list[str]], universe: set[str]) -> dict:
    """Score only within `universe` (the ids actually extracted), so a sample run is fair."""
    rows = {}
    for cid in sorted(gold):
        g = set(gold[cid]) & universe
        p = set(pred.get(cid, [])) & universe
        pr, rc, f1 = prf(p, g)
        rows[cid] = {"precision": pr, "recall": rc, "f1": f1, "n_pred": len(p), "n_gold": len(g),
                     "fp": sorted(p - g), "fn": sorted(g - p)}
    # A criterion with no gold ids inside the universe would score a free 1.0 when
    # nothing is predicted, so it is reported but left out of the mean.
    scored = [r for r in rows.values() if r["n_gold"] > 0]
    rows["_mean_f1"] = sum(r["f1"] for r in scored) / len(scored) if scored else 0.0
    rows["_n_scored"] = len(scored)
    return rows


def field_accuracy(llm: dict[str, dict], gold: dict[str, dict], fields: list[str]) -> dict:
    """Exact-match accuracy per flat field over reports present in both maps."""
    out = {}
    ids = [i for i in llm if i in gold]
    for f in fields:
        relevant = [i for i in ids if gold[i].get(f) is not None or llm[i].get(f) is not None]
        if not relevant:
            continue
        ok = sum(1 for i in relevant if _eq(llm[i].get(f), gold[i].get(f)))
        out[f] = {"accuracy": ok / len(relevant), "n": len(relevant),
                  "mismatch_ids": [i for i in relevant if not _eq(llm[i].get(f), gold[i].get(f))][:20]}
    return out


def _eq(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return False
    if isinstance(a, list) and isinstance(b, list):
        return sorted(a) == sorted(b)
    return a == b
