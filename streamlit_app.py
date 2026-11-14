"""Streamlit UI for testing the query layer, the 30 criteria, and the extractions.

    .venv/bin/streamlit run streamlit_app.py

Reads the saved extractions in outputs/extractions/ (run `cli extract` first).
Writing a rule by hand, scoring the criteria, and browsing reports make no API
calls. Translating a question makes 1 to 4 calls. The per-report fallback runs
only after you click the button that shows its cost estimate, and only if the
fresh estimate is no larger than the one on the button.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pathfinder.cli import DATA, FIELDS, OUT, _records, cache_path, load_split
from pathfinder.criteria.engine import load_rules, match_all
from pathfinder.evaluate import _eq, score_criteria
from pathfinder.extraction.extractor import REPO_ROOT, fingerprint
from pathfinder.extraction.postprocess import derive
from pathfinder.oracle import gold_extraction
from pathfinder.query.catalog import build_catalog
from pathfinder.query.run import format_rule, run_query, to_dict

st.set_page_config(page_title="pathfinder", layout="wide")

EFFORTS = ["low", "medium", "high", ""]  # "" = no reasoning_effort (non-reasoning models)
EXAMPLES = [
    "",
    "Kidney cancers larger than 7 cm, broken down by extent of invasion",
    "How many breast cancers are triple negative?",
    "Colon cancers with 1 or more positive lymph nodes, by pT",
    "Women aged 70 or older with lung adenocarcinoma and visceral pleural invasion",
    "Prostate cancers with Grade Group 4 or 5 and seminal vesicle invasion",
    "Colon cancers with MMR deficiency (needs the per-report fallback)",
]
# Shown in the results table next to the id; the full record is one click away.
SUMMARY_COLS = ["organ", "sex", "age_band", "histologic_category", "tumor_size_text", "pT", "pN",
                "lymph_nodes_positive", "margin_status"]


# ---------- data (cached per split, model, effort, and cache-file mtime) ----------

@st.cache_data(show_spinner=False)
def usable_efforts(path: str, model: str, mtime: float) -> dict[str, int]:
    """{effort: number of rows whose fingerprint matches it} for one extraction file."""
    with open(path) as fh:
        fps = Counter(re.findall(r'"fingerprint": "(\w+)"', fh.read()))
    return {e: fps[fingerprint(model, e or None)] for e in EFFORTS if fps[fingerprint(model, e or None)]}


def available() -> tuple[dict[str, dict[str, dict[str, int]]], list[str]]:
    """({split: {model: usable_efforts}}, [files with no usable rows]) from outputs/extractions/."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    unusable = []
    for p in sorted((OUT / "extractions").glob("*.jsonl")):
        split, _, model = p.name[: -len(".jsonl")].partition(".")
        efforts = usable_efforts(str(p), model, p.stat().st_mtime)
        if efforts:
            out.setdefault(split, {})[model] = efforts
        else:
            unusable.append(p.name)
    return out, unusable


@st.cache_data(show_spinner="Loading saved extractions...")
def load_records(split: str, model: str, effort: str, mtime: float) -> dict[str, dict]:
    return _records(split, model, effort or None)


@st.cache_data(show_spinner="Loading report text...")
def load_rows(split: str) -> dict[str, dict]:
    return {r["id"]: r for r in load_split(split)}


@st.cache_resource(show_spinner="Building field catalog...")
def load_catalog(split: str, model: str, effort: str, mtime: float):
    return build_catalog(load_records(split, model, effort, mtime))


@st.cache_data(show_spinner="Building gold records from train labels...")
def load_gold() -> dict[str, dict]:
    return {i: derive(gold_extraction(r["labels"])) for i, r in load_rows("train").items()}


# ---------- sidebar ----------

avail, unusable = available()
if not avail:
    st.error("No usable saved extractions in outputs/extractions/. Run "
             "`.venv/bin/python -m pathfinder.cli extract --split train --limit 210` first.")
    st.stop()

with st.sidebar:
    st.header("pathfinder")
    splits = sorted(avail)
    split = st.selectbox("Split", splits, index=splits.index("train") if "train" in splits else 0)
    models = sorted(avail[split])
    model = st.selectbox("Extraction model", models,
                         index=models.index("gpt-5.4-mini") if "gpt-5.4-mini" in models else 0)
    efforts = list(avail[split][model])
    effort = st.selectbox("Reasoning effort", efforts, format_func=lambda e: e or "(none)",
                          help="Only efforts that some saved extractions were made with are listed (the "
                               "fingerprint includes the effort). Also used for translation and fallback calls.")
    mtime = cache_path(split, model).stat().st_mtime
    records = load_records(split, model, effort, mtime)
    rows = load_rows(split)
    texts = {i: rows[i]["report_text"] for i in records if i in rows}
    catalog = load_catalog(split, model, effort, mtime)
    st.caption(f"{len(records):,} saved {split} records ({len(rows):,} reports in {split}.jsonl)")
    if unusable:
        st.caption("Not listed (no rows match the current prompt and schema; re-run extract): "
                   + ", ".join(unusable))
    max_fallback = st.number_input("Max fallback calls", min_value=0, max_value=5000, value=100, step=25,
                                   help="The fallback is refused if more reports than this need an LLM call.")
    load_dotenv(REPO_ROOT / ".env")
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    (st.success if has_key else st.warning)(
        "OPENAI_API_KEY found" if has_key else "No OPENAI_API_KEY: only hand-written rules work")

gold = load_gold() if split == "train" else {}
CTX = (split, model, effort, mtime)  # a result is only shown under the settings it was computed with


# ---------- shared pieces ----------

def _cell(v):
    return "" if v is None else (", ".join(map(str, v)) if isinstance(v, list) else str(v))


def show_report(rid: str, key: str):
    """Report text next to its extracted record (and the gold record on train)."""
    if rid not in records:
        st.warning(f"{rid} has no saved record")
        return
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"**{rid}**")
        st.code(texts.get(rid, ""), language="text", wrap_lines=True, height=560)
    with right:
        rec = records[rid]
        df = pd.DataFrame({"field": list(rec)})
        if rid in gold:
            g = gold[rid]
            df["match"] = ["" if f not in FIELDS else ("ok" if _eq(rec.get(f), g.get(f)) else "DIFF")
                           for f in rec]
            df["extracted"] = [_cell(v) for v in rec.values()]
            df["gold"] = [_cell(g.get(f)) for f in rec]
            st.caption("gold = parsed from train labels by oracle.py; match covers only the scored fields")
            if st.toggle("Only fields that differ", key=f"{key}-diff-{rid}"):
                df = df[df["match"] == "DIFF"]
        else:
            df["extracted"] = [_cell(v) for v in rec.values()]
        st.dataframe(df, hide_index=True, height=520, width="stretch",
                     column_config={"field": st.column_config.TextColumn(width="medium"),
                                    "match": st.column_config.TextColumn(width="small")})


def id_table(ids: list[str], key: str, extra: dict[str, dict[str, str]] | None = None) -> str | None:
    """Table of ids with summary fields; returns the id of the clicked row.

    The key includes a digest of the ids: Streamlit keeps a keyed table's
    selection when its data changes, so a row picked in an earlier result
    would otherwise point at the wrong report, or past the end of the list.
    """
    if not ids:
        st.info("No reports.")
        return None
    digest = hashlib.sha1("\n".join(ids).encode()).hexdigest()[:10]
    data = [{"id": i, **{c: _cell(records.get(i, {}).get(c)) for c in SUMMARY_COLS},
             **{name: col.get(i, "") for name, col in (extra or {}).items()}} for i in ids]
    ev = st.dataframe(pd.DataFrame(data), hide_index=True, width="stretch",
                      on_select="rerun", selection_mode="single-row", key=f"{key}-{digest}", height=320)
    sel = ev.selection.rows if ev and ev.selection else []
    return ids[sel[0]] if sel and sel[0] < len(ids) else None


# ---------- tab: ask ----------

def run(question, rule=None, answer_type="ids", by=None, uncovered=None, approved=None):
    """Run a query. The fallback spends only when `approved` (the estimate the user clicked)
    is given and the fresh estimate is no larger; otherwise it returns its estimate unspent."""
    def confirm(est):
        return approved is not None and est.calls <= approved.calls and est.input_tokens <= approved.input_tokens

    try:
        res = run_query(question, records, texts, model, confirm, int(max_fallback), effort or None,
                        rule=rule, answer_type=answer_type, breakdown_by=by, uncovered=uncovered,
                        catalog=catalog)
    except Exception as e:  # missing key, network, rate limit: show it, keep the app alive
        st.session_state.ask_error = f"{type(e).__name__}: {e}"
        return None
    st.session_state.ask_error = None
    return res


def set_result(res, inputs):
    st.session_state.result = res
    st.session_state.result_ctx = CTX
    st.session_state.result_inputs = inputs


def editor_inputs(question):
    s = st.session_state
    return ("write", question, s.rule_text, s.answer_type, s.by, s.uncovered)


def load_rule_into_editor():
    res = st.session_state.result
    st.session_state.mode = "Write the rule"
    st.session_state.rule_text = json.dumps(res.rule, indent=1)
    st.session_state.answer_type = res.answer_type
    st.session_state.by = res.breakdown_by or "(none)"
    st.session_state.uncovered = res.uncovered or ""


def tab_ask():
    for k, v in {"mode": "Translate with the LLM", "rule_text": '{"all": [["organ", "eq", "breast"]]}',
                 "answer_type": "ids", "by": "(none)", "uncovered": "", "result": None,
                 "result_ctx": None, "result_inputs": None, "ask_error": None}.items():
        st.session_state.setdefault(k, v)
    if st.session_state.result is not None and st.session_state.result_ctx != CTX:
        st.session_state.result = None
        st.info("Cleared the previous result: the split, model, effort or extraction file changed.")

    ex = st.selectbox("Example questions", EXAMPLES, format_func=lambda s: s or "(pick one, or type below)")
    question = st.text_area("Question", value=ex, height=80, placeholder="e.g. LVI-positive breast cancers over 2 cm")
    mode = st.radio("How to get the rule", ["Translate with the LLM", "Write the rule"], horizontal=True,
                    key="mode", help="Writing the rule yourself makes no API calls (unless you fill in the "
                                     "fallback question and approve it).")
    if mode == "Write the rule":
        c1, c2 = st.columns([3, 2])
        with c1:
            st.text_area("Rule JSON", key="rule_text", height=180, persist_state="page",
                         help='Condition: [field, op, value]. Group: {"all": [...]}, {"any": [...]}, {"not": rule}.')
        with c2:
            st.radio("Answer", ["ids", "count", "breakdown"], key="answer_type", horizontal=True,
                     persist_state="page")
            st.selectbox("Breakdown by", ["(none)"] + list(catalog), key="by", persist_state="page")
            st.text_input("Outside the schema (yes/no question per report, optional)", key="uncovered",
                          placeholder="Does the report state MMR deficiency?", persist_state="page")
        with st.expander("Field catalog"):
            st.dataframe(pd.DataFrame([{"field": f.name, "type": f.kind, "ops": ", ".join(sorted(f.ops)),
                                        "values": ", ".join(map(str, f.allowed or f.seen[:12])),
                                        "description": f.description} for f in catalog.values()]),
                         hide_index=True, width="stretch")
        inputs = editor_inputs(question)
    else:
        inputs = ("translate", question)

    if st.button("Run", type="primary"):
        if mode == "Translate with the LLM":
            if not question.strip():
                st.warning("Type a question first.")
                return
            with st.spinner("Translating the question into a rule..."):
                set_result(run(question), inputs)
        else:
            try:
                rule = json.loads(st.session_state.rule_text)
            except json.JSONDecodeError as e:
                st.session_state.result = None
                st.error(f"Rule is not valid JSON: {e}")
                return
            by = None if st.session_state.by == "(none)" else st.session_state.by
            set_result(run(question or "(hand-written rule)", rule, st.session_state.answer_type, by,
                           st.session_state.uncovered.strip() or None), inputs)

    if st.session_state.ask_error:
        st.error(st.session_state.ask_error)
    res = st.session_state.result
    if res is not None:
        if st.session_state.result_inputs != inputs:
            st.caption(":orange[The inputs above changed since this result was computed. Click Run to update it.]")
        show_result(res)


def show_result(res):
    st.divider()
    t = res.translation
    if t:
        fixes = len(t.attempts) - 1
        st.caption(f"Translation: {t.usage['calls']} LLM call(s), {t.usage['input_tokens']:,} in / "
                   f"{t.usage['output_tokens']:,} out tokens" + (f", {fixes} repair(s)" if fixes else ""))
        rejected = t.attempts[:-1] if t.ok else t.attempts
        if rejected:
            with st.expander(f"{len(rejected)} rejected attempt(s)"):
                for i, a in enumerate(rejected, 1):
                    st.markdown(f"**Attempt {i}**")
                    st.code(a["rule_json"] or "(no rule)", language="json")
                    for err in a["errors"]:
                        st.markdown(f"- {err}")
    if res.error:
        st.error(res.error)
        return

    st.markdown(f"**Read as:** {res.interpretation}")
    c1, c2 = st.columns(2)
    c1.code(format_rule(res.rule), language="text")
    c2.code(json.dumps(res.rule), language="json", wrap_lines=True)
    st.button("Edit this rule", on_click=load_rule_into_editor,
              help="Copy the rule into the editor to tweak and re-run without a translation call.")

    fb = res.fallback
    if res.uncovered:
        st.markdown(f"**Outside the schema:** {res.uncovered}  \n"
                    f"The rule alone (covered part) matches **{len(res.rule_ids)}** report(s).")
        if fb and fb.estimate:
            e = fb.estimate
            st.caption(f"Fallback: {e.reports} candidate(s), {e.cached} cached, {e.calls} needing a call "
                       f"(~{e.input_tokens:,} in / ~{e.output_tokens:,} out tokens estimated).")
        if fb and fb.skipped_reason:
            e = fb.estimate
            st.warning(f"No final answer yet: checking the uncovered part needs {e.calls} LLM call(s) to "
                       f"{model}. Nothing has been sent.")
            if e.calls > max_fallback:
                st.info(f"That is above the cap of {int(max_fallback)}. Narrow the question, or raise "
                        "**Max fallback calls** in the sidebar to get an approve button.")
            elif st.button(f"Spend ~{e.input_tokens + e.output_tokens:,} tokens on {e.calls} call(s)",
                           type="primary"):
                with st.spinner("Checking each report..."):
                    new = run(res.question, res.rule, res.answer_type, res.breakdown_by, res.uncovered,
                              approved=e)
                if new is not None:
                    new.translation, new.interpretation = res.translation, res.interpretation
                    set_result(new, st.session_state.result_inputs)
                st.rerun()
            st.markdown("The covered part alone:")
            pick = id_table(res.rule_ids, "rule-only")
            if pick:
                show_report(pick, "ask")
            return
        if fb:
            st.caption(f"Fallback used {fb.usage['calls']} call(s), {fb.usage['input_tokens']:,} in / "
                       f"{fb.usage['output_tokens']:,} out tokens. yes={len(fb.matched)} no={len(fb.rejected)} "
                       f"unverified-quote={len(fb.unverified)} failed={len(fb.failed)}")

    m1, m2 = st.columns([1, 3])
    m1.metric("Count", res.count)
    if not res.complete:
        m2.warning(f"Lower bound: {len(fb.failed)} report(s) could not be checked.")
    st.download_button("Download result JSON", json.dumps(to_dict(res), indent=1),
                       file_name="query_result.json", mime="application/json")

    if res.answer_type == "breakdown" and res.breakdown_by:
        df = pd.DataFrame(res.breakdown, columns=["value", "count"])
        df["value"] = df["value"].astype(str)
        b1, b2 = st.columns([2, 3])
        b1.dataframe(df.rename(columns={"value": res.breakdown_by}), hide_index=True, width="stretch")
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("count:Q", title="reports"),
            y=alt.Y("value:N", sort="-x", title=res.breakdown_by, axis=alt.Axis(labelLimit=0)),
            tooltip=["value", "count"])
        b2.altair_chart(chart, width="stretch")
    extra = {"evidence": fb.evidence} if fb else {}
    with st.expander(f"Matching reports ({res.count})", expanded=res.answer_type == "ids"):
        pick = id_table(res.ids, "result-ids", extra)
    if pick:
        show_report(pick, "ask")
    if fb and fb.unverified:
        with st.expander(f"Unverified: {len(fb.unverified)} report(s) said yes, but the quote was not found "
                         "in the text, so they are excluded"):
            pick_u = id_table(fb.unverified, "unverified-ids", {"evidence": fb.evidence})
        if pick_u:
            show_report(pick_u, "unverified")


# ---------- tab: criteria ----------

def tab_criteria():
    if split != "train":
        st.info("Gold answers exist only for train. Switch the split to train.")
        return
    rules = load_rules()
    texts_by_id = {c["id"]: c["text"] for c in json.loads((DATA / "criteria.json").read_text())}
    gold_answers = json.loads((DATA / "criteria_train_answers.json").read_text())
    pred = match_all(records, rules)
    scores = score_criteria(pred, gold_answers, set(records))
    st.metric(f"Mean F1 over {scores['_n_scored']} criteria with gold ids ({len(records):,} train records)",
              f"{scores['_mean_f1']:.4f}")
    cids = [c for c in scores if not c.startswith("_")]
    table = pd.DataFrame([{"id": c, "F1": round(scores[c]["f1"], 3), "P": round(scores[c]["precision"], 3),
                           "R": round(scores[c]["recall"], 3), "pred": scores[c]["n_pred"],
                           "gold": scores[c]["n_gold"], "criterion": texts_by_id.get(c, "")} for c in cids])
    ev = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                      selection_mode="single-row", key=f"crit-table-{len(records)}", height=400,
                      column_config={"criterion": st.column_config.TextColumn(width="large")})
    sel = ev.selection.rows if ev and ev.selection else []
    if not sel or sel[0] >= len(cids):
        st.caption("Click a row to see its rule and its false positives and negatives.")
        return
    cid = cids[sel[0]]
    s = scores[cid]
    lists = {"fp": s["fp"], "fn": s["fn"],
             "tp": sorted(set(pred[cid]) & set(gold_answers[cid]) & set(records))}
    names = {"fp": "False positives", "fn": "False negatives", "tp": "True positives"}
    st.subheader(f"{cid}: {texts_by_id.get(cid, '')}")
    st.code(format_rule(rules[cid]["rule"]), language="text")
    first = next((i for i, k in enumerate(lists) if lists[k]), 0)
    kind = st.radio("Show", list(lists), index=first, horizontal=True, key=f"kind-{cid}",
                    format_func=lambda k: f"{names[k]} ({len(lists[k])})")
    pick = id_table(lists[kind], f"crit-ids-{cid}-{kind}")
    if pick:
        show_report(pick, "crit")


# ---------- tab: reports ----------

def tab_reports():
    ids = sorted(records)
    q = st.text_input("Report id (or part of it)", placeholder="SYN-a985f38840").strip().lower()
    match = [i for i in ids if q in i.lower()] if q else ids
    if not match:
        st.warning("No saved record has that id.")
        return
    rid = st.selectbox(f"{len(match):,} report(s)", match)
    show_report(rid, "browse")


tabs = st.tabs(["Ask", "Criteria", "Reports"])
with tabs[0]:
    tab_ask()
with tabs[1]:
    tab_criteria()
with tabs[2]:
    tab_reports()
