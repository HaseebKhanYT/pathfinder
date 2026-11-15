"""Command line entry point.

  python -m pathfinder.cli extract  --split train [--limit N] [--model M]
  python -m pathfinder.cli evaluate --split train [--model M]
  python -m pathfinder.cli submit   [--model M] [--out outputs/submission.json]
  python -m pathfinder.cli query    "question" [--split train] [--rule JSON] [--yes] [--max-fallback N]

Extractions are cached in outputs/extractions/<split>.<model>.jsonl ("/" in the model id
becomes "__").
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from .criteria.engine import match_all
from .evaluate import field_accuracy, score_criteria
from .extraction.extractor import DEFAULT_MODEL, REPO_ROOT, extract_reports, fingerprint, load_cache, model_slug
from .extraction.postprocess import derive
from .oracle import gold_extraction

DATA = REPO_ROOT / "data" / "biobase" / "pack_reports"
OUT = REPO_ROOT / "outputs"

FIELDS = [
    "sex", "age_band", "report_year", "organ", "histologic_category", "differentiation_grade",
    "tumor_size_mm", "lymphovascular_invasion", "perineural_invasion", "margin_status",
    "closest_margin_mm", "involved_margins", "lymph_nodes_positive", "lymph_nodes_examined",
    "pT", "pN", "breast.nottingham_grade", "breast.er_status", "breast.pr_status",
    "breast.her2_status", "breast.tumor_focality", "lung.visceral_pleural_invasion",
    "colon.tumor_deposits", "stomach.her2_status", "prostate.grade_group",
    "prostate.extraprostatic_extension", "prostate.seminal_vesicle_invasion",
    "kidney.isup_grade", "kidney.sarcomatoid_or_rhabdoid", "kidney.extent_of_invasion",
    "skin.group",
]


def load_split(split: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / f"{split}.jsonl").read_text().splitlines() if l.strip()]


def stratified(rows: list[dict], n: int, seed: int = 7) -> list[dict]:
    """Organ-balanced sample, so a small run still exercises every criterion."""
    by = defaultdict(list)
    for r in rows:
        by[r["labels"]["organ"]].append(r)
    rng = random.Random(seed)
    organs = sorted(by)
    quota = {o: n // len(organs) + (1 if i < n % len(organs) else 0) for i, o in enumerate(organs)}
    out = []
    for organ in organs:
        out += rng.sample(by[organ], min(quota[organ], len(by[organ])))
    return out


def cache_path(split: str, model: str) -> Path:
    return OUT / "extractions" / f"{split}.{model_slug(model)}.jsonl"


def cmd_extract(a):
    rows = load_split(a.split)
    if a.limit:
        rows = stratified(rows, a.limit) if a.split == "train" else rows[: a.limit]
    res = extract_reports(rows, cache_path(a.split, a.model), a.model, a.reasoning_effort, a.concurrency)
    tok_in = sum((r.get("usage") or {}).get("input_tokens") or 0 for r in res.values())
    tok_out = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in res.values())
    print(f"[extract] {len(res)} extracted; tokens in={tok_in:,} out={tok_out:,}")


def _records(split: str, model: str, reasoning_effort: str | None) -> dict[str, dict]:
    # Only records made with the current prompt/schema/effort; re-derive from the raw
    # extraction so postprocess changes apply without re-calling the API.
    fp = fingerprint(model, reasoning_effort)
    return {
        i: derive(r["extraction"])
        for i, r in load_cache(cache_path(split, model)).items()
        if "extraction" in r and r.get("fingerprint") == fp
    }


def cmd_evaluate(a):
    recs = _records("train", a.model, a.reasoning_effort)
    if not recs:
        raise SystemExit("no cached train extractions; run extract first")
    gold_answers = json.loads((DATA / "criteria_train_answers.json").read_text())
    crit = score_criteria(match_all(recs), gold_answers, set(recs))
    labels = {r["id"]: r["labels"] for r in load_split("train") if r["id"] in recs}
    gold_flat = {i: derive(gold_extraction(l)) for i, l in labels.items()}
    fields = field_accuracy(recs, gold_flat, FIELDS)
    print(f"\nCriteria on {len(recs)} train reports (model {a.model})")
    print(f"{'id':5} {'P':>6} {'R':>6} {'F1':>6} {'pred':>5} {'gold':>5}")
    for cid, r in crit.items():
        if not cid.startswith("_"):
            print(f"{cid:5} {r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f} {r['n_pred']:5d} {r['n_gold']:5d}")
    print(f"mean F1 = {crit['_mean_f1']:.4f} over {crit['_n_scored']} criteria with gold ids\n\nField accuracy vs labels")
    for f, r in fields.items():
        print(f"  {f:38} {r['accuracy']:.3f}  (n={r['n']})")
    report = OUT / f"eval.train.{a.model}.json"
    report.write_text(json.dumps({"criteria": crit, "fields": fields}, indent=1))
    print(f"\nwrote {report}")


def cmd_submit(a):
    rows = load_split("test")
    extract_reports(rows, cache_path("test", a.model), a.model, a.reasoning_effort, a.concurrency)
    recs = _records("test", a.model, a.reasoning_effort)
    sub = match_all(recs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sub, indent=1))
    print(f"wrote {out} ({len(recs)}/{len(rows)} test reports extracted)")


def _confirm_interactive(est) -> bool:
    import sys

    msg = (f"\nThe question has a part outside the schema. Checking it needs {est.calls} LLM call(s) "
           f"(~{est.input_tokens:,} input / ~{est.output_tokens:,} output tokens, {est.cached} already cached).")
    print(msg, file=sys.stderr)
    if not sys.stdin.isatty():  # never spend without an explicit yes from a person
        print("Not a terminal, so not asking: re-run with --yes to allow these calls.", file=sys.stderr)
        return False
    print("Spend Nebius credits on this? [y/N] ", end="", file=sys.stderr, flush=True)
    try:
        return sys.stdin.readline().strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def cmd_query(a):
    from .query.run import render, run_query, to_dict

    recs = _records(a.split, a.model, a.reasoning_effort)
    if not recs:
        raise SystemExit(f"no saved {a.split} extractions for {a.model}; run extract --split {a.split} first")
    texts = {r["id"]: r["report_text"] for r in load_split(a.split) if r["id"] in recs}
    confirm = (lambda est: True) if a.yes else _confirm_interactive
    rule = json.loads(a.rule) if a.rule else None
    res = run_query(a.question, recs, texts, a.llm_model, confirm, a.max_fallback, a.reasoning_effort,
                    rule=rule, answer_type=a.answer, breakdown_by=a.by, uncovered=a.uncovered)
    print(f"[{len(recs)} saved {a.split} records]")
    print(render(res))
    if a.out and not res.error:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(to_dict(res), indent=1))
        print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser(prog="pathfinder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "evaluate", "submit"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--reasoning-effort", default="low")
        p.add_argument("--concurrency", type=int, default=16)
        if name == "extract":
            p.add_argument("--split", default="train", choices=["train", "test"])
            p.add_argument("--limit", type=int, default=0)
        if name == "submit":
            p.add_argument("--out", default=str(OUT / "submission.json"))
    q = sub.add_parser("query", help="answer a cohort question over saved extractions")
    q.add_argument("question")
    q.add_argument("--split", default="train", choices=["train", "test"])
    q.add_argument("--model", default=DEFAULT_MODEL, help="whose saved extractions to query")
    q.add_argument("--llm-model", default=DEFAULT_MODEL, help="model for translation and fallback calls")
    q.add_argument("--reasoning-effort", default="low")
    q.add_argument("--rule", help="skip translation and run this rule JSON (no API calls)")
    q.add_argument("--answer", default="ids", choices=["ids", "count", "breakdown"], help="with --rule only")
    q.add_argument("--by", help="breakdown field, with --rule only")
    q.add_argument("--uncovered", help="with --rule: the yes/no question for the per-report fallback")
    q.add_argument("--yes", action="store_true", help="allow fallback API calls without asking")
    q.add_argument("--max-fallback", type=int, default=100,
                   help="refuse the fallback if more than this many reports need an LLM call")
    q.add_argument("--out", help="also write the result as JSON")
    a = ap.parse_args()
    {"extract": cmd_extract, "evaluate": cmd_evaluate, "submit": cmd_submit, "query": cmd_query}[a.cmd](a)


if __name__ == "__main__":
    main()
