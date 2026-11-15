"""Streamlit UI for testing the query layer, the 30 criteria, and the extractions.

    .venv/bin/streamlit run streamlit_app.py

Reads the saved extractions in outputs/extractions/ (run `cli extract` first).
Scoring the criteria and browsing reports make no API calls. Answering a
question makes 1 to 4 translation calls; the rule is never shown to the user. The per-report fallback runs
only after you click the button that shows its cost estimate, and only if the
fresh estimate is no larger than the one on the button.

The look follows the "Pathfinder Redesign" Claude Design file: a centered
search on the home view, then one question and its answer card. The theme is
in .streamlit/config.toml; the CSS below styles the keyed containers
(`st.container(key="pf-...")` renders as the class `st-key-pf-...`).
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
from collections import Counter
from contextlib import contextmanager

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from pathfinder.cli import DATA, FIELDS, OUT, _records, cache_path, load_split
from pathfinder.criteria.engine import load_rules, match_all
from pathfinder.evaluate import _eq, score_criteria
from pathfinder.extraction.extractor import DEFAULT_MODEL, REPO_ROOT, fingerprint, model_from_slug
from pathfinder.extraction.postprocess import derive
from pathfinder.oracle import gold_extraction
from pathfinder.query.catalog import build_catalog
from pathfinder.query.run import format_rule, run_query, to_dict

st.set_page_config(page_title="pathfinder", layout="wide", initial_sidebar_state="collapsed")

EFFORTS = ["low", "medium", "high", ""]  # "" = no reasoning_effort (non-reasoning models)
# (chip label, question sent)
EXAMPLES = [
    ("Kidney > 7 cm by extent", "Kidney cancers larger than 7 cm, broken down by extent of invasion"),
    ("Triple-negative breast", "How many breast cancers are triple negative?"),
    ("Node-positive colon by pT", "Colon cancers with 1 or more positive lymph nodes, by pT"),
    ("Prostate GG 4–5 with SVI", "Prostate cancers with Grade Group 4 or 5 and seminal vesicle invasion"),
]
# Results table columns (record field -> header); the full record is one click away.
SUMMARY_COLS = {"organ": "organ", "sex": "sex", "age_band": "age", "histologic_category": "histology",
                "tumor_size_text": "size", "pT": "pT", "pN": "pN", "margin_status": "margin"}
# Home-page image band: (Commons file, credit, height in px, flex grow). Hotlinked from
# Wikimedia Commons with the credit shown on the image, as in the design file.
COMMONS = "https://commons.wikimedia.org/wiki/"
BAND = [
    ("Microscope_(inverted).jpg", "National Cancer Institute / Wikimedia Commons", 170, 1),
    ("Medical_stethoscope.jpg", "Wikimedia Commons", 230, 1),
    ("Test_tubes_in_a_laboratory.jpg", "Vit Almer / Wikimedia Commons", 270, 1.2),
    ("Sterile_surgical_instruments.jpg", "Wikimedia Commons", 230, 1),
    ("Histologic_Slide_under_MIcroscope.jpg", "Ed Uthman / Wikimedia Commons", 170, 1),
]

CSS = """
<style>
/* ---- page chrome: hide Streamlit's header bar, sidebar, and padding ---- */
header[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stDecoration"] { display: none !important; }
.stMainBlockContainer, [data-testid="stMainBlockContainer"] { padding: 20px 32px 0 !important; max-width: none; }
a { color: #3C7A6E; } a:hover { color: #2A5C52; }
[data-testid="InputInstructions"] { display: none; }
/* Streamlit pulls markdown up by 1rem to collapse paragraph margins; undo it for the app's own HTML blocks */
[data-testid="stMarkdownContainer"]:has(> [class*="pf-"]) { margin-bottom: 0; }
/* style-only st.markdown calls would otherwise each add a flex gap */
[data-testid="stElementContainer"]:has(style) { display: none; }
.st-key-pf-hero, .st-key-pf-thread, .st-key-pf-dock, .st-key-pf-page, .st-key-pf-chips { margin-inline: auto; }
@keyframes pfPulse { 0%,100% { opacity: .25; transform: scale(.8); } 50% { opacity: 1; transform: scale(1); } }
.pf-serif { font-family: "Newsreader", Georgia, serif; }
.pf-mono { font-family: "JetBrains Mono", ui-monospace, monospace; }

/* ---- header: logo (resets the app), nav, settings ---- */
.st-key-pf-header { align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.st-key-pf-logo button { background: none; border: 0; padding: 6px 8px; margin: -6px -8px; border-radius: 10px;
  color: #1E3A35; gap: 10px; min-height: 0; box-shadow: none; }
.st-key-pf-logo button:hover { background: #E3ECE8; color: #1E3A35; }
.st-key-pf-logo button:focus:not(:active) { color: #1E3A35; border: 0; }
.st-key-pf-logo button::before { content: ""; flex: none; width: 26px; height: 26px; border-radius: 50%;
  background: radial-gradient(circle, #F3F6F4 0 5px, #3C7A6E 5.5px); }
.st-key-pf-logo button p { white-space: nowrap; font-family: "Newsreader", Georgia, serif; font-size: 24px; font-weight: 500;
  letter-spacing: -0.01em; }
.st-key-pf-right { align-items: center; gap: 4px; flex-wrap: wrap; justify-content: flex-end; }
.st-key-pf-right button { color: #5E726D; font-size: 13px; min-height: 0; padding: 6px 10px; border-radius: 10px; }
.st-key-pf-right button p { font-size: 13px; }
.st-key-pf-right button:hover { color: #1E3A35; background: #E3ECE8; }
.st-key-pf-right [data-testid="stPopover"] button { border: 0; background: none; }
.st-key-pf-right [data-testid="stPopover"] button:hover { background: #E3ECE8; }
/* Settings is an icon-only button: hide the popover's expand chevron. */
.st-key-pf-right [data-testid="stPopoverButton"] > div > div[aria-hidden="true"] { display: none; }

/* ---- home ---- */
.st-key-pf-home { align-items: center; gap: 56px; padding: 48px 0 64px; }
.st-key-pf-hero { width: 100%; max-width: 680px; align-items: center; gap: 20px; }
.pf-h1 { margin: 0; font-family: "Newsreader", Georgia, serif; font-weight: 400; font-size: clamp(40px, 6vw, 64px);
  line-height: 1.05; letter-spacing: -0.02em; text-align: center; text-wrap: balance; color: #1E3A35; }
.st-key-pf-chips { justify-content: center; gap: 8px; max-width: 640px; flex-wrap: wrap; }
.st-key-pf-chips button { border: 1px solid #D5E2DD; background: #FAFCFB; color: #3B5550; padding: 8px 14px;
  min-height: 0; }
.st-key-pf-chips button p { font-size: 14px; }
.st-key-pf-chips button:hover { background: #E3ECE8; border-color: #BCD1C9; color: #3B5550; }
.pf-band { display: flex; align-items: flex-end; gap: 16px; width: 100%; max-width: 1180px; margin: 0 auto; }
.pf-band figure { position: relative; margin: 0; min-width: 0; border-radius: 20px; overflow: hidden;
  background: #E3ECE8; }
.pf-band img { width: 100%; height: 100%; object-fit: cover; display: block; }
.pf-band figcaption { position: absolute; left: 8px; bottom: 8px; max-width: calc(100% - 16px); font-size: 11px;
  line-height: 1.3; padding: 3px 8px; border-radius: 999px; background: rgba(30,58,53,.55); opacity: 0;
  transition: opacity .15s; }
.pf-band figcaption a { color: #fff; text-decoration: none; }
.pf-band figure:hover figcaption, .pf-band figure:focus-within figcaption { opacity: 1; }
@media (max-width: 760px) { .pf-band figure:first-child, .pf-band figure:last-child { display: none; } }

/* ---- search bar (home and the docked bar in a thread) ---- */
[data-testid="stForm"]:has(.st-key-pf-bar), [data-testid="stForm"]:has(.st-key-pf-dockbar) { border: 0; padding: 0; }
.st-key-pf-bar, .st-key-pf-dockbar { align-items: center; gap: 12px; width: 100%; }
.st-key-pf-bar [data-testid="stTextInputRootElement"], .st-key-pf-dockbar [data-testid="stTextInputRootElement"] { background: #fff !important;
  border: 1px solid #D5E2DD !important; border-radius: 28px; padding: 0 24px;
  box-shadow: 0 1px 2px rgba(30,58,53,.04), 0 8px 24px rgba(30,58,53,.06); }
.st-key-pf-bar [data-testid="stTextInputRootElement"] { min-height: 60px; }
.st-key-pf-dockbar [data-testid="stTextInputRootElement"] { min-height: 56px; box-shadow: 0 8px 24px rgba(30,58,53,.06); }
.st-key-pf-bar input, .st-key-pf-dockbar input { font-size: 17px; line-height: 1.4; color: #1E3A35;
  padding: 0 !important; background: transparent !important; height: auto; }
.st-key-pf-dockbar input { font-size: 16px; }
.st-key-pf-bar input::placeholder, .st-key-pf-dockbar input::placeholder { color: #8A9B96; opacity: 1; }
.st-key-pf-bar [data-testid="stTextInputRootElement"]:focus-within, .st-key-pf-dockbar [data-testid="stTextInputRootElement"]:focus-within {
  border-color: #7FA89B !important; }
.st-key-pf-send button, .st-key-pf-docksend button { flex: none; width: 60px; height: 60px; min-height: 0;
  padding: 0; border-radius: 50%; border: 0; background: #3C7A6E; color: #fff;
  box-shadow: 0 6px 18px rgba(60,122,110,.28); }
.st-key-pf-docksend button { width: 56px; height: 56px; box-shadow: none; }
.st-key-pf-send button:hover, .st-key-pf-docksend button:hover { background: #2F675C; color: #fff; }
.st-key-pf-send button:focus:not(:active), .st-key-pf-docksend button:focus:not(:active) { color: #fff; }
.st-key-pf-send [data-testid="stIconMaterial"], .st-key-pf-docksend [data-testid="stIconMaterial"] { font-size: 22px; }
.st-key-pf-send [data-testid="stMarkdownContainer"], .st-key-pf-docksend [data-testid="stMarkdownContainer"] {
  position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }

/* ---- thread: question bubble, working indicator, answer card, docked bar ---- */
.st-key-pf-thread { width: 100%; max-width: 960px; margin: 0 auto; gap: 24px; padding: 4px 0 40px; }
.pf-bubble { margin-left: auto; width: fit-content; max-width: 80%; background: #3C7A6E; color: #fff;
  border-radius: 22px 22px 6px 22px; padding: 14px 20px; font-size: 17px; line-height: 1.45;
  overflow-wrap: anywhere; }
.pf-working { display: flex; align-items: center; gap: 12px; color: #4E635E; font-size: 15px; }
.pf-dots { display: inline-flex; gap: 5px; }
.pf-dots span { width: 7px; height: 7px; border-radius: 50%; background: #3C7A6E;
  animation: pfPulse 1s infinite ease-in-out; }
.pf-dots span:nth-child(2) { animation-delay: .15s; } .pf-dots span:nth-child(3) { animation-delay: .3s; }
.st-key-pf-card, .st-key-pf-page-card { background: #fff; border: 1px solid #DCE6E1;
  border-radius: 24px; padding: 32px; gap: 28px; }
@media (max-width: 640px) { .st-key-pf-card, .st-key-pf-page-card { padding: 20px; } }
.pf-head { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 24px; }
.pf-eyebrow { font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: #3C7A6E; font-weight: 600; }
.pf-read { display: flex; flex-direction: column; gap: 8px; max-width: 560px; }
.pf-interp { font-family: "Newsreader", Georgia, serif; font-size: 26px; line-height: 1.25; text-wrap: pretty; }
.pf-count { display: flex; align-items: baseline; gap: 8px; }
.pf-count b { font-family: "Newsreader", Georgia, serif; font-weight: 400; font-size: 64px; line-height: 1;
  color: #2A5C52; }
.pf-count span { font-size: 15px; color: #4E635E; }
.pf-rule { background: #EEF4F1; border-radius: 14px; padding: 14px 18px; font-family: "JetBrains Mono", ui-monospace,
  monospace; font-size: 14px; line-height: 1.6; color: #244741; overflow-x: auto; white-space: pre-wrap; }
.pf-head + .pf-rule, .pf-read + .pf-rule { margin-top: 28px; }
/* Key findings panel next to the report text. */
.pf-find { display: flex; flex-direction: column; gap: 22px; background: #fff; border: 1px solid #DCE6E1;
  border-radius: 20px; padding: 24px; }
.pf-fhead { display: flex; flex-direction: column; gap: 4px; }
.pf-dx { font-family: "Newsreader", Georgia, serif; font-size: 24px; line-height: 1.2; color: #1E3A35; }
.pf-where { font-size: 14px; color: #3B5550; }
.pf-who { font-size: 13px; color: #5E726D; }
.pf-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.pf-tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
@media (max-width: 900px) { .pf-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.pf-tile { background: #F5F9F7; border: 1px solid #E6EEEA; border-radius: 14px; padding: 12px 14px;
  display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.pf-tile span { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #5E726D; font-weight: 600; }
.pf-tile b { font-family: "Newsreader", Georgia, serif; font-weight: 500; font-size: 22px; color: #2A5C52;
  line-height: 1.15; overflow-wrap: anywhere; }
.pf-tile.flag { background: #FBF0EA; border-color: #F0D5C6; }
.pf-tile.flag b { color: #A2482A; }
.pf-fsect { display: flex; flex-direction: column; }
.pf-fsect .pf-eyebrow { margin-bottom: 6px; }
.pf-frow { display: flex; justify-content: space-between; align-items: center; gap: 16px; padding: 10px 0;
  border-top: 1px solid #E6EEEA; font-size: 14px; color: #1E3A35; }
.pf-frow > span:first-child { color: #3B5550; flex: 0 0 auto; max-width: 55%; text-align: left; }
.pf-frow > :last-child { text-align: right; flex: 0 1 auto; }
.pf-fval { display: inline-flex; align-items: center; gap: 10px; }
.pf-fval em { font-style: normal; font-size: 13px; color: #5E726D; }
.pf-pill { display: inline-block; border-radius: 999px; padding: 3px 11px; font-size: 13px; font-weight: 500;
  white-space: nowrap; border: 1px solid transparent; }
.pf-pill.flag { background: #FBF0EA; color: #A2482A; border-color: #F0D5C6; }
.pf-pill.pos { background: #E3F0EC; color: #245C50; border-color: #C5DED6; }
.pf-pill.mid { background: #FBF5E3; color: #8A6A1C; border-color: #EEDFB4; }
.pf-pill.neg { background: #F3F6F4; color: #4E635E; border-color: #DCE6E1; }
.pf-pill.na { background: transparent; color: #8A9B96; border-color: #E6EEEA; }
.pf-anchor { scroll-margin-top: 24px; }
[class*="st-key-pf-ids-"] canvas, .st-key-pf-crit-list canvas { cursor: pointer !important; }
/* Report text and findings side by side: the left column stretches to the right one's height. */
[class*="st-key-pf-report-"] [data-testid="stHorizontalBlock"] { align-items: stretch; }
[class*="st-key-pf-report-"] [data-testid="stColumn"]:first-child > [data-testid="stVerticalBlock"] { height: 100%; }
/* The text's element gets no intrinsic height (flex-basis 0, height 0), so the row's height comes
   from the findings panel and the text fills it, scrolling inside. */
[class*="st-key-pf-report-"] [data-testid="stColumn"]:first-child [data-testid="stElementContainer"]:has([data-testid="stCode"]) {
  flex: 1 1 0; height: 0; min-height: 320px; }
[class*="st-key-pf-report-"] [data-testid="stColumn"]:first-child [data-testid="stCode"],
[class*="st-key-pf-report-"] [data-testid="stColumn"]:first-child [data-testid="stCode"] pre { height: 100% !important; }
.pf-label { font-size: 14px; font-weight: 600; color: #1E3A35; }
.pf-bars { display: flex; flex-direction: column; gap: 12px; }
.pf-bar { display: grid; grid-template-columns: minmax(0, 180px) minmax(0, 1fr) 36px; align-items: center; gap: 14px;
  font-size: 14px; }
.pf-bar > span:first-child { color: #3B5550; overflow-wrap: anywhere; }
.pf-bar > span:last-child { text-align: right; font-variant-numeric: tabular-nums; }
.pf-track { height: 10px; border-radius: 5px; background: #E6EEEA; }
.pf-track div { height: 10px; border-radius: 5px; background: #7FA89B; }
.pf-note { font-size: 13px; color: #5E726D; }
.pf-callout { background: #F5F9F7; border: 1px solid #DCE6E1; border-radius: 14px; padding: 14px 18px;
  font-size: 14px; line-height: 1.5; color: #3B5550; }
.st-key-pf-actions { gap: 4px; flex-wrap: wrap; }
.st-key-pf-actions button { color: #3C7A6E; min-height: 0; padding: 4px 10px; border-radius: 10px; }
.st-key-pf-actions button p { font-size: 14px; }
.st-key-pf-actions button:hover { background: #EEF4F1; color: #2A5C52; }
/* sticky goes on Streamlit's wrapper: the keyed block is exactly as tall as its wrapper, so it can't stick */
[data-testid="stLayoutWrapper"]:has(> .st-key-pf-dock) { position: sticky; bottom: 0; z-index: 10;
  padding: 16px 0 28px; background: linear-gradient(to bottom, rgba(243,246,244,0), #F3F6F4 30%); }
.st-key-pf-dock { width: 100%; max-width: 960px; margin: 0 auto; }

/* ---- criteria and reports pages ---- */
.st-key-pf-page { width: 100%; max-width: 1180px; margin: 0 auto; gap: 24px; padding: 16px 0 48px; }
.pf-title { margin: 0; padding: 0 !important; font-family: "Newsreader", Georgia, serif; font-weight: 400; font-size: 40px;
  line-height: 1.1; letter-spacing: -0.02em; }
.pf-sub { margin: 6px 0 0; color: #5E726D; font-size: 15px; }
</style>
"""


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
        split, _, slug = p.name[: -len(".jsonl")].partition(".")
        model = model_from_slug(slug)
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


# ---------- settings and app state ----------

st.markdown(CSS, unsafe_allow_html=True)
s = st.session_state
for k, v in {"page": "ask", "view": "home", "asked": "", "pending": None,
             "result": None, "result_ctx": None, "ask_error": None}.items():
    s.setdefault(k, v)

avail, _ = available()
if not avail:
    st.error("No usable saved extractions in outputs/extractions/. Run "
             "`.venv/bin/python -m pathfinder.cli extract --split train --limit 210` first.")
    st.stop()


def _keep(key: str, options: list, preferred) -> None:
    """Point a settings widget's state at a valid option before the widget is drawn.

    The split selector lives in a popover drawn late in the page, while the data
    it selects is loaded first, so the choice is resolved here from session state.
    """
    if s.get(key) not in options:
        s[key] = preferred if preferred in options else options[0]


splits = sorted(avail)
_keep("split", splits, "train")
split = s.split
# The model and effort are fixed: the preferred ones when this split has extractions for them,
# otherwise the first available, so the app still opens on a split extracted differently.
# Live calls (translation, fallback) always go to Nebius with DEFAULT_MODEL, whichever file is read.
MODEL, EFFORT = DEFAULT_MODEL, "low"
# Files to read, best first: Nebius extractions, then the older OpenAI gpt-5.4-mini ones.
model = next((m for m in (MODEL, "gpt-5.4-mini") if m in avail[split]), sorted(avail[split])[0])
effort = EFFORT if EFFORT in avail[split][model] else next(iter(avail[split][model]))
mtime = cache_path(split, model).stat().st_mtime
records = load_records(split, model, effort, mtime)
rows = load_rows(split)
texts = {i: rows[i]["report_text"] for i in records if i in rows}
catalog = load_catalog(split, model, effort, mtime)
load_dotenv(REPO_ROOT / ".env")
has_key = bool(os.environ.get("NEBIUS_API_KEY"))
gold = load_gold() if split == "train" else {}
MAX_FALLBACK = 20  # the fallback is refused when more reports than this need an LLM call
CTX = (split, model, effort, mtime)  # a result is only shown under the settings it was computed with


def reset():
    """Logo click: back to the empty home view (settings are kept)."""
    s.page, s.view, s.asked, s.pending, s.result, s.ask_error = "ask", "home", "", None, None, None


def go(page: str):
    s.page = page


def settings_panel():
    st.markdown('<span class="pf-label">Settings</span>', unsafe_allow_html=True)
    st.selectbox("Split", splits, key="split")
    st.caption(f"{len(records):,} saved {split} records ({len(rows):,} reports in {split}.jsonl)")
    if not has_key:
        st.warning("No NEBIUS_API_KEY: questions cannot be answered")


def header():
    # The active tab is darker and bold; Streamlit gives the button the class st-key-nav-<page>.
    st.markdown(f"<style>.st-key-nav-{s.page} button {{ color: #1E3A35; }} .st-key-nav-{s.page} button p "
                "{ font-weight: 600; }</style>", unsafe_allow_html=True)
    with st.container(horizontal=True, key="pf-header"):
        with st.container(key="pf-logo", width="content"):
            st.button("pathfinder", on_click=reset, help="Back to start")
        with st.container(horizontal=True, key="pf-right", width="content"):
            for page, label in (("ask", "Find"), ("criteria", "Criteria"), ("reports", "Reports")):
                st.button(label, type="tertiary", on_click=go, args=(page,), key=f"nav-{page}")
            with st.popover("", icon=":material/tune:", help="Settings", key="pf-settings"):
                settings_panel()


# ---------- shared pieces ----------

def _cell(v):
    return "" if v is None else (", ".join(map(str, v)) if isinstance(v, list) else str(v))


def esc(v) -> str:
    return html.escape(str(v))


@contextmanager
def working(message: str):
    """The design's three pulsing dots with a message, shown while a slow call runs."""
    slot = st.empty()
    slot.markdown(f'<div class="pf-working"><span class="pf-dots"><span></span><span></span><span></span></span>'
                  f'<span>{esc(message)}</span></div>', unsafe_allow_html=True)
    try:
        yield
    finally:
        slot.empty()


def _nice(v) -> str:
    """Display form of a coded value: not_identified -> Not identified, low_positive -> Low positive."""
    if v is True:
        return "Present"
    if v is False:
        return "Absent"
    s_ = str(v).replace("_", " ")
    return s_[:1].upper() + s_[1:]


# Tone of a status pill. "flag" marks an adverse finding (spread, invasion, a positive margin);
# "pos" a positive biomarker; "neg" a negative or absent result; "na" not reported.
_FLAG_IF_PRESENT = {"lymphovascular_invasion", "perineural_invasion", "prostate.extraprostatic_extension",
                    "prostate.seminal_vesicle_invasion", "prostate.cribriform_glands",
                    "prostate.intraductal_carcinoma", "kidney.sarcomatoid_or_rhabdoid", "colon.tumor_deposits",
                    "colon.macroscopic_perforation", "lung.visceral_pleural_invasion", "skin.ulceration"}
_BIOMARKERS = {"breast.er_status", "breast.pr_status", "breast.her2_status", "stomach.her2_status"}


def _tone(field: str, v) -> str:
    sv = str(v).lower()
    if v is None or sv in ("not_reported", "not_assessed", "unknown"):
        return "na"
    if field in _BIOMARKERS:
        return {"positive": "pos", "low_positive": "pos", "equivocal": "mid"}.get(sv, "neg")
    if field in _FLAG_IF_PRESENT:
        return "flag" if sv in ("present", "positive") or v is True else "neg"
    if field == "breast.dcis_present":
        return "pos" if v is True else "neg"
    return "plain"


# Organ-specific rows: (label, field). Grades go in the headline tiles instead.
_ORGAN_ROWS = {
    "breast": [("ER", "breast.er_status"), ("PR", "breast.pr_status"), ("HER2", "breast.her2_status"),
               ("Focality", "breast.tumor_focality"), ("DCIS", "breast.dcis_present")],
    "prostate": [("Extraprostatic extension", "prostate.extraprostatic_extension"),
                 ("Seminal vesicle invasion", "prostate.seminal_vesicle_invasion"),
                 ("Cribriform glands", "prostate.cribriform_glands"),
                 ("Intraductal carcinoma", "prostate.intraductal_carcinoma")],
    "kidney": [("Sarcomatoid / rhabdoid", "kidney.sarcomatoid_or_rhabdoid"),
               ("Extent of invasion", "kidney.extent_of_invasion")],
    "colon": [("Depth of invasion", "colon.depth_of_invasion"), ("Tumor deposits", "colon.tumor_deposits"),
              ("Macroscopic perforation", "colon.macroscopic_perforation")],
    "stomach": [("HER2", "stomach.her2_status"), ("Depth of invasion", "stomach.depth_of_invasion")],
    "lung": [("Visceral pleural invasion", "lung.visceral_pleural_invasion")],
    "skin": [("Diagnosis group", "skin.group"), ("Ulceration", "skin.ulceration")],
}


def _grade(rec: dict) -> tuple[str, str] | None:
    """(label, value) for the organ's grading system, if the report gives one."""
    o = rec.get("organ")
    if o == "prostate" and rec.get("prostate.grade_group") is not None:
        p_, s2 = rec.get("prostate.gleason_primary"), rec.get("prostate.gleason_secondary")
        gl = f" · Gleason {p_}+{s2}={p_ + s2}" if p_ is not None and s2 is not None else ""
        return "Grade group", f"{rec['prostate.grade_group']}{gl}"
    if o == "breast" and rec.get("breast.nottingham_grade") is not None:
        return "Nottingham grade", str(rec["breast.nottingham_grade"])
    if o == "kidney" and rec.get("kidney.isup_grade") is not None:
        return "ISUP grade", str(rec["kidney.isup_grade"])
    if o == "skin" and rec.get("skin.breslow_thickness_mm") is not None:
        return "Breslow", f"{rec['skin.breslow_thickness_mm']:g} mm"
    if rec.get("differentiation_grade") is not None:
        return "Grade", f"G{rec['differentiation_grade']}"
    return None


def findings_panel(rec: dict) -> str:
    """HTML for the report's key findings: diagnosis, headline tiles, and status rows."""
    where = " · ".join(_nice(v) for v in (rec.get("organ"), rec.get("site"), rec.get("laterality"),
                                          rec.get("procedure")) if v and v != "not_applicable")
    who = " · ".join(str(v) for v in (rec.get("sex"), rec.get("age_band") and f"age {rec['age_band']}",
                                      rec.get("report_year")) if v)

    stage = " ".join(v for v in (rec.get("pT"), rec.get("pN")) if v)
    npos, nex = rec.get("lymph_nodes_positive"), rec.get("lymph_nodes_examined")
    nodes = f"{npos} / {nex}" if npos is not None and nex is not None else (str(npos) if npos is not None else "")
    g = _grade(rec)
    tiles = [("Stage", stage, ""), ("Size", rec.get("tumor_size_text") or "", ""),
             (g[0] if g else "Grade", g[1] if g else "", ""),
             ("Nodes positive", nodes, "flag" if npos else "")]
    if not any(val for _, val, _ in tiles):
        tiles = []  # e.g. a benign skin biopsy: no stage, size, grade or nodes to show
    tiles_html = "".join(
        f'<div class="pf-tile {tone}"><span>{esc(lbl)}</span><b>{esc(val) if val else "—"}</b></div>'
        for lbl, val, tone in tiles)

    def rows(items):
        out = []
        for label, field in items:
            v = rec.get(field)
            if v is None or v == "not_applicable":
                continue
            tone = _tone(field, v)
            val = f'<span class="pf-pill {tone}">{esc(_nice(v))}</span>' if tone != "plain" else esc(_nice(v))
            out.append(f'<div class="pf-frow"><span>{esc(label)}</span>{val}</div>')
        return "".join(out)

    margin = rec.get("margin_status")
    margin_row = ""
    if margin is not None:
        tone = {"positive": "flag", "negative": "neg"}.get(margin, "na")
        extra = ""
        if rec.get("closest_margin_mm") is not None:
            extra = f'<em>closest {rec["closest_margin_mm"]:g} mm</em>'
        if rec.get("involved_margins"):
            extra = f'<em>{esc(_cell(rec["involved_margins"]))}</em>'
        margin_row = (f'<div class="pf-frow"><span>Margins</span><span class="pf-fval">{extra}'
                      f'<span class="pf-pill {tone}">{esc(_nice(margin))}</span></span></div>')
    organ_rows = rows(_ORGAN_ROWS.get(rec.get("organ"), []))
    spread_rows = rows([("Lymphovascular invasion", "lymphovascular_invasion"),
                        ("Perineural invasion", "perineural_invasion")]) + margin_row

    tags = []
    if rec.get("organ") == "breast":
        er, pr, her2 = (rec.get(f"breast.{k}") for k in ("er_status", "pr_status", "her2_status"))
        if er == pr == her2 == "negative":
            tags.append(("Triple-negative", "flag"))
        elif her2 == "positive":
            tags.append(("HER2-positive", "pos"))
    if rec.get("organ") == "stomach" and rec.get("stomach.her2_status") == "positive":
        tags.append(("HER2-positive", "pos"))
    if margin == "positive":
        tags.append(("Positive margin", "flag"))
    tags_html = ("<div class=\"pf-tags\">" + "".join(f'<span class="pf-pill {tn}">{esc(tx)}</span>'
                                                    for tx, tn in tags) + "</div>") if tags else ""

    sect = lambda title, body: (f'<div class="pf-fsect"><span class="pf-eyebrow">{title}</span>{body}</div>'
                                if body else "")
    return (f'<div class="pf-find"><div class="pf-fhead"><span class="pf-eyebrow">Diagnosis</span>'
            f'<span class="pf-dx">{esc(_nice(rec.get("histologic_type") or rec.get("histologic_category") or "—"))}'
            f'</span><span class="pf-where">{esc(where)}</span><span class="pf-who">{esc(who)}</span>{tags_html}</div>'
            + (f'<div class="pf-tiles">{tiles_html}</div>' if tiles_html else "")
            + sect("Biomarkers and features", organ_rows) + sect("Invasion and margins", spread_rows)
            + "</div>")


def show_report(rid: str, key: str, compare: bool = False, scroll: bool = True):
    """Report text next to its key findings. compare=True (Criteria page) adds the
    field-by-field extracted vs gold table used for scoring. scroll=True brings the
    report into view when it is opened from a table, which otherwise leaves it
    below the fold."""
    if rid not in records:
        st.warning(f"{rid} has no saved record")
        return
    anchor = f"pf-anchor-{key}"
    st.markdown(f'<div id="{anchor}" class="pf-anchor"></div>', unsafe_allow_html=True)
    if scroll:
        # The script runs when this element mounts, and the report id in it makes each newly
        # opened report a new element, so other reruns (a toggle, say) do not scroll again.
        st.html(f'<script>/* {esc(rid)} */ setTimeout(() => document.getElementById("{anchor}")'
                '?.scrollIntoView({behavior: "smooth", block: "start"}), 150);</script>',
                unsafe_allow_javascript=True)
    with st.container(key=f"pf-report-{key}"):
        left, right = st.columns([1, 1])
    with left:
        st.markdown('<span class="pf-eyebrow">Original report</span>', unsafe_allow_html=True)
        # height="stretch" plus the .st-key-pf-report CSS makes the text box end where the
        # findings panel ends; the text scrolls inside it.
        st.code(texts.get(rid, ""), language="text", wrap_lines=True, height="stretch")
    with right:
        rec = records[rid]
        # Same heading as the left column, outside the box, so both boxes start on one line.
        st.markdown('<span class="pf-eyebrow">Key findings</span>', unsafe_allow_html=True)
        st.markdown(findings_panel(rec), unsafe_allow_html=True)
        if compare and rid in gold:
            g = gold[rid]
            df = pd.DataFrame({"field": list(rec)})
            df["match"] = ["" if f not in FIELDS else ("ok" if _eq(rec.get(f), g.get(f)) else "DIFF")
                           for f in rec]
            df["extracted"] = [_cell(v) for v in rec.values()]
            df["gold"] = [_cell(g.get(f)) for f in rec]
            with st.expander("Extracted vs gold record"):
                st.caption("gold = parsed from train labels by oracle.py; match covers only the scored fields")
                if st.toggle("Only fields that differ", key=f"{key}-diff-{rid}"):
                    df = df[df["match"] == "DIFF"]
                st.dataframe(df, hide_index=True, height=420, width="stretch",
                             column_config={"field": st.column_config.TextColumn(width="medium"),
                                            "match": st.column_config.TextColumn(width="small")})


def id_table(ids: list[str], key: str, extra: dict[str, dict[str, str]] | None = None) -> str | None:
    """Table of ids with summary fields; returns the id of the clicked row.

    Cell selection, not row selection: row selection draws a checkbox column,
    which reads as "pick several", while a click on any cell should just open
    that report.

    The key includes a digest of the ids: Streamlit keeps a keyed table's
    selection when its data changes, so a row picked in an earlier result
    would otherwise point at the wrong report, or past the end of the list.
    """
    if not ids:
        st.markdown('<span class="pf-note">No reports.</span>', unsafe_allow_html=True)
        return None
    digest = hashlib.sha1("\n".join(ids).encode()).hexdigest()[:10]
    # The table is drawn on a canvas, so CSS cannot underline a cell. A combining low line after
    # each character draws the underline instead, and the link colour plus the pointer cursor
    # (CSS on .st-key-pf-ids-*) say the id opens the report. Rows are mapped back by index.
    data = [{"id": "".join(ch + "\u0332" for ch in i), **{h: _cell(records.get(i, {}).get(c)) for c, h in SUMMARY_COLS.items()},
             **{name: col.get(i, "") for name, col in (extra or {}).items()}} for i in ids]
    df = pd.DataFrame(data).style.map(lambda _: "color: #2A5C52", subset=["id"])
    with st.container(key=f"pf-ids-{key}"):
        ev = st.dataframe(df, hide_index=True, width="stretch",
                          on_select="rerun", selection_mode="single-cell", key=f"{key}-{digest}",
                          height=min(320, 38 + 35 * len(ids)),
                          column_config={"id": st.column_config.TextColumn(width="medium")})
    cells = ev.selection.get("cells", []) if ev and ev.selection else []
    row = cells[0][0] if cells else None
    return ids[row] if row is not None and row < len(ids) else None


# ---------- page: ask ----------

def run(question, rule=None, answer_type="ids", by=None, uncovered=None, approved=None):
    """Run a query. The fallback spends only when `approved` (the estimate the user clicked)
    is given and the fresh estimate is no larger; otherwise it returns its estimate unspent."""
    def confirm(est):
        return approved is not None and est.calls <= approved.calls and est.input_tokens <= approved.input_tokens

    try:
        res = run_query(question, records, texts, MODEL, confirm, MAX_FALLBACK, EFFORT,
                        rule=rule, answer_type=answer_type, breakdown_by=by, uncovered=uncovered,
                        catalog=catalog)
    except Exception as e:  # missing key, network, rate limit: show it, keep the app alive
        s.ask_error = f"{type(e).__name__}: {e}"
        return None
    s.ask_error = None
    return res


def set_result(res):
    s.result = res
    s.result_ctx = CTX


def ask(question: str):
    """Queue a question and switch to the thread view.
    The query runs on the next script run, under the working indicator."""
    s.page, s.view, s.asked, s.result, s.ask_error = "ask", "thread", question, None, None
    s.pending = question


def search_bar(form_key: str, bar_key: str, send_key: str, placeholder: str):
    """Rounded input with the round send button. Enter submits; the input clears on send."""
    with st.form(form_key, clear_on_submit=True, border=False):
        with st.container(horizontal=True, vertical_alignment="center", key=bar_key):
            q = st.text_input("Question", placeholder=placeholder, label_visibility="collapsed",
                              key=f"{form_key}-q")
            with st.container(key=send_key, width="content"):
                sent = st.form_submit_button("Search", icon=":material/arrow_upward:")
    if sent and q.strip():
        ask(q.strip())
        st.rerun()


def page_ask():
    if s.result is not None and s.result_ctx != CTX:
        s.result = None
        st.info("Cleared the previous result: the split, model, effort or extraction file changed.")
    if s.view == "home":
        home()
    else:
        thread()


def home():
    with st.container(key="pf-home"):
        with st.container(key="pf-hero"):
            st.markdown('<h1 class="pf-h1">Find the right patients.</h1>', unsafe_allow_html=True)
            search_bar("pf-form-home", "pf-bar", "pf-send", "e.g. LVI-positive breast cancers over 2 cm")
            with st.container(horizontal=True, key="pf-chips", width="content"):
                for i, (label, question) in enumerate(EXAMPLES):
                    st.button(label, key=f"ex-{i}", on_click=ask, args=(question,), help=question)
        figures = "".join(
            f'<figure style="flex:{grow};height:{h}px"><img src="{COMMONS}Special:FilePath/{f}?width=900" '
            f'alt="{esc(f.rsplit(".", 1)[0].replace("_", " "))}" loading="lazy">'
            f'<figcaption><a href="{COMMONS}File:{f}" target="_blank" rel="noopener">{esc(credit)}</a>'
            f'</figcaption></figure>' for f, credit, h, grow in BAND)
        st.markdown(f'<div class="pf-band">{figures}</div>', unsafe_allow_html=True)


def thread():
    with st.container(key="pf-thread"):
        if s.asked:
            st.markdown(f'<div class="pf-bubble">{esc(s.asked)}</div>', unsafe_allow_html=True)
        question = s.pending
        if question is not None:
            s.pending = None  # run once, even if the call fails
            with working("Searching the reports…"):
                set_result(run(question))
        if s.ask_error:
            st.error(s.ask_error)
        if s.result is not None:
            with st.container(key="pf-card"):
                show_result(s.result)
    with st.container(key="pf-dock"):
        search_bar("pf-form-dock", "pf-dockbar", "pf-docksend", "Ask another question")


def show_result(res):
    if res.error:
        st.error(res.error)
        return

    fb = res.fallback
    pending_fb = bool(fb and fb.skipped_reason)
    count = "—" if pending_fb else f"{res.count:,}"
    unit = "awaiting the fallback" if pending_fb else ("report" if res.count == 1 else "reports")
    st.markdown(
        f'<div class="pf-head"><div class="pf-read"><span class="pf-eyebrow">Read as</span>'
        f'<span class="pf-interp">{esc(res.interpretation)}</span></div>'
        f'<div class="pf-count"><b>{count}</b><span>{unit}</span></div></div>', unsafe_allow_html=True)
    with st.container(horizontal=True, key="pf-actions"):
        st.download_button("Download result JSON", json.dumps(to_dict(res), indent=1), type="tertiary",
                           icon=":material/download:", on_click="ignore", file_name="query_result.json", mime="application/json")

    if res.uncovered:
        st.markdown(f'<div class="pf-callout"><b>Outside the schema:</b> {esc(res.uncovered)}<br>'
                    f'The rest of the question matches <b>{len(res.rule_ids)}</b> report(s).</div>',
                    unsafe_allow_html=True)
        if fb and fb.estimate:
            e = fb.estimate
            st.markdown(f'<span class="pf-note">Fallback: {e.reports} candidate(s), {e.cached} cached, '
                        f'{e.calls} needing a call (~{e.input_tokens:,} in / ~{e.output_tokens:,} out tokens '
                        'estimated).</span>', unsafe_allow_html=True)
        if pending_fb:
            e = fb.estimate
            st.warning(f"No final answer yet: checking the uncovered part needs {e.calls} LLM call(s) to "
                       f"{MODEL}. Nothing has been sent.")
            if e.calls > MAX_FALLBACK:
                st.info(f"That is above the cap of {MAX_FALLBACK} calls. Narrow the question to get an "
                        "approve button.")
            elif st.button(f"Spend ~{e.input_tokens + e.output_tokens:,} tokens on {e.calls} call(s)",
                           type="primary"):
                with working("Checking each report…"):
                    new = run(res.question, res.rule, res.answer_type, res.breakdown_by, res.uncovered,
                              approved=e)
                if new is not None:
                    new.translation, new.interpretation = res.translation, res.interpretation
                    set_result(new)
                st.rerun()
            st.markdown('<span class="pf-label">Matching the rest of the question</span>', unsafe_allow_html=True)
            pick = id_table(res.rule_ids, "rule-only")
            if pick:
                show_report(pick, "ask")
            return

    if not res.complete:
        st.warning(f"Lower bound: {len(fb.failed)} report(s) could not be checked.")

    if res.answer_type == "breakdown" and res.breakdown_by and res.breakdown:
        top = max(n for _, n in res.breakdown) or 1
        bars = "".join(f'<div class="pf-bar"><span>{esc(v)}</span><div class="pf-track">'
                       f'<div style="width:{n / top * 100:.1f}%"></div></div><span>{n}</span></div>'
                       for v, n in res.breakdown)
        st.markdown(f'<div class="pf-bars"><span class="pf-label">By {esc(res.breakdown_by)}</span>{bars}</div>',
                    unsafe_allow_html=True)

    extra = {"evidence": fb.evidence} if fb else {}
    st.markdown('<span class="pf-label">Matching reports</span>', unsafe_allow_html=True)
    pick = id_table(res.ids, "result-ids", extra)
    if pick:
        show_report(pick, "ask")
    elif res.ids:
        st.markdown('<span class="pf-note">Click a row to read the report and its key findings.</span>',
                    unsafe_allow_html=True)
    if fb and fb.unverified:
        with st.expander(f"Unverified: {len(fb.unverified)} report(s) said yes, but the quote was not found "
                         "in the text, so they are excluded"):
            pick_u = id_table(fb.unverified, "unverified-ids", {"evidence": fb.evidence})
        if pick_u:
            show_report(pick_u, "unverified")


# ---------- page: criteria ----------

def page_title(title: str, sub: str):
    st.markdown(f'<div><h2 class="pf-title">{esc(title)}</h2><p class="pf-sub">{esc(sub)}</p></div>',
                unsafe_allow_html=True)


def page_criteria():
    page_title("Criteria", "The 30 rules scored against the train answer key.")
    if split != "train":
        st.info("Gold answers exist only for train. Switch the split to train in Settings (top right).")
        return
    rules = load_rules()
    texts_by_id = {c["id"]: c["text"] for c in json.loads((DATA / "criteria.json").read_text())}
    gold_answers = json.loads((DATA / "criteria_train_answers.json").read_text())
    pred = match_all(records, rules)
    scores = score_criteria(pred, gold_answers, set(records))
    with st.container(key="pf-page-card"):
        st.markdown(f'<div class="pf-head"><div class="pf-read"><span class="pf-eyebrow">Mean F1</span>'
                    f'<span class="pf-interp">over {scores["_n_scored"]} criteria with gold ids '
                    f'({len(records):,} train records)</span></div><div class="pf-count">'
                    f'<b>{scores["_mean_f1"]:.4f}</b></div></div>', unsafe_allow_html=True)
        cids = [c for c in scores if not c.startswith("_")]
        table = pd.DataFrame([{"id": c, "F1": round(scores[c]["f1"], 3), "P": round(scores[c]["precision"], 3),
                               "R": round(scores[c]["recall"], 3), "pred": scores[c]["n_pred"],
                               "gold": scores[c]["n_gold"], "criterion": texts_by_id.get(c, "")} for c in cids])
        with st.container(key="pf-crit-list"):
            ev = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                              selection_mode="single-cell", key=f"crit-table-{len(records)}", height=400,
                              column_config={"criterion": st.column_config.TextColumn(width="large")})
        cells = ev.selection.get("cells", []) if ev and ev.selection else []
        sel = [cells[0][0]] if cells else []
        if not sel or sel[0] >= len(cids):
            st.markdown('<span class="pf-note">Click a row to see its rule and its false positives and '
                        'negatives.</span>', unsafe_allow_html=True)
            return
        cid = cids[sel[0]]
        sc = scores[cid]
        lists = {"fp": sc["fp"], "fn": sc["fn"],
                 "tp": sorted(set(pred[cid]) & set(gold_answers[cid]) & set(records))}
        names = {"fp": "False positives", "fn": "False negatives", "tp": "True positives"}
        st.markdown(f'<div class="pf-read" style="max-width:none"><span class="pf-eyebrow">{esc(cid)}</span>'
                    f'<span class="pf-interp">{esc(texts_by_id.get(cid, ""))}</span></div>'
                    f'<div class="pf-rule">{esc(format_rule(rules[cid]["rule"]))}</div>', unsafe_allow_html=True)
        first = next((i for i, k in enumerate(lists) if lists[k]), 0)
        kind = st.radio("Show", list(lists), index=first, horizontal=True, key=f"kind-{cid}",
                        format_func=lambda k: f"{names[k]} ({len(lists[k])})")
        pick = id_table(lists[kind], f"crit-ids-{cid}-{kind}")
        if pick:
            show_report(pick, "crit", compare=True)


# ---------- page: reports ----------

def page_reports():
    page_title("Reports", f"Look up any saved {split} report by id.")
    ids = sorted(records)
    with st.container(key="pf-page-card"):
        q = st.text_input("Report id (or part of it)", placeholder="SYN-a985f38840").strip().lower()
        match = [i for i in ids if q in i.lower()] if q else ids
        if not match:
            st.warning("No saved record has that id.")
            return
        rid = st.selectbox(f"{len(match):,} report(s)", match)
        show_report(rid, "browse", scroll=False)


header()
if s.page == "criteria":
    with st.container(key="pf-page"):
        page_criteria()
elif s.page == "reports":
    with st.container(key="pf-page"):
        page_reports()
else:
    page_ask()
