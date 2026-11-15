# pathfinder
Finding the patients for the right research.

## Report extractor (Task A)

`pathfinder/` turns each synthetic pathology report into a fixed JSON record with an LLM, then answers the 30 criteria with declarative rules over that record.

- `pathfinder/extraction/schema.py`: the fixed schema. Sections: header (sex, age band, year), specimen (organ, site, laterality, procedure), shared tumor fields (histology, size as written, LVI, PNI, margins, nodes, pT/pN) and one block per organ (breast ER/PR/HER2, prostate Grade Group, kidney extent, skin group, and so on).
- `pathfinder/extraction/prompt.py`: the field dictionary the model follows. The final diagnosis block is authoritative; the prose is used only for fields the block omits.
- `pathfinder/extraction/postprocess.py`: flattens a record and computes derived values in code (tumor size in mm, age bounds, stage base), so the model never does arithmetic.
- `pathfinder/criteria/criteria_rules.json`: each criterion as conditions over flat fields, for example c04 is `organ == breast`, `lymphovascular_invasion == present`, `tumor_size_mm > 20`.
- `pathfinder/oracle.py`: builds the same record from train `labels` with no LLM, to test the rules alone and to score the extractor field by field.

Extraction uses LangChain (`ChatOpenAI.with_structured_output`, strict JSON schema) with `gpt-5.4-mini` at low reasoning effort by default. Results are cached per report in `outputs/extractions/`, tagged with a fingerprint of model, effort, prompt and schema, so changing any of them re-extracts.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo 'OPENAI_API_KEY=sk-...' > .env              # git-ignored
.venv/bin/python -m pathfinder.cli extract --split train --limit 210   # organ-stratified sample
.venv/bin/python -m pathfinder.cli evaluate      # criteria P/R/F1 and field accuracy vs train labels
.venv/bin/python -m pathfinder.cli submit        # extract test.jsonl, write outputs/submission.json
```

## Query layer

`pathfinder/query/` answers new cohort questions over the saved extractions. Reports are extracted once; each question becomes a rule.

1. **Translate** (`translate.py`): the LLM gets the question, the field catalog with allowed values (`catalog.py`, read from the schema), and the 30 criteria rules as worked examples, and writes a rule in the engine's JSON format.
2. **Validate** (`validate.py`): unknown fields, values that aren't allowed or never occur, and ops that don't fit a field's type are rejected; the errors go back to the LLM to fix (up to 3 repairs).
3. **Run** (`run.py`): the engine runs the rule over the saved records and returns ids, a count, or a breakdown. No API calls.
4. **Fall back** (`fallback.py`): if part of the question is outside the schema (for example MMR status), the rule covers what it can and each remaining report gets a yes/no LLM check with a verbatim quote. A yes counts only if the quote (3+ words) is found in the report. This step shows a token estimate and asks first, refuses above `--max-fallback` (default 100), never spends from a non-terminal without `--yes`, and caches verdicts.
5. **Show**: the rule is printed with every answer, plus a `--rule ... --uncovered ...` line that re-runs the same query with no translation call.

```sh
.venv/bin/python -m pathfinder.cli query "Kidney cancers larger than 7 cm, broken down by extent of invasion"
.venv/bin/python -m pathfinder.cli query "LVI-positive breast" --rule '{"all":[["organ","eq","breast"],["lymphovascular_invasion","eq","present"]]}' --answer breakdown --by pT_base   # no API calls
.venv/bin/python -m unittest discover -s tests   # offline tests, fake LLMs
```

## Test UI

`streamlit_app.py` is a local Streamlit app for trying the query layer and checking the extractions by eye.

```sh
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run streamlit_app.py
```

- **Find** (home): type a question and press Enter, or pick an example chip. The LLM translates it into a rule behind the scenes, and the answer card shows how the question was read, the count, a breakdown chart, and the matching reports. The rule itself is not shown; use `cli query` to see it or to run a hand-written rule. Click a row to read the report next to its extracted record. When part of the question is outside the schema, the card shows the fallback's cost estimate and sends nothing until you click the button that names the token cost.
- **Criteria** (train only): the 30 rules scored against `criteria_train_answers.json`, with false positives and negatives for each criterion, and each report's extracted fields compared with the gold record from `oracle.py`.
- **Reports**: look up any saved report by id.

The app reads the `gpt-5.4-mini` extractions made at `low` effort (`MODEL` and `EFFORT` in `streamlit_app.py`; it falls back to the first available file if a split has none) and uses the same model and effort for translation. It refuses a fallback that needs more than 20 LLM calls (`MAX_FALLBACK` in `streamlit_app.py`); narrow the question instead.

Click the **pathfinder** logo to return to the empty home view. The settings popover, behind the icon at the top right, picks the split and warns when `OPENAI_API_KEY` is missing.

The look follows the "Pathfinder Redesign" file in Claude Design. Colors and fonts are in `.streamlit/config.toml`; the hero, search bar, and answer card are styled by the CSS at the top of `streamlit_app.py`. The fonts load from Google Fonts and the home-page photos are hotlinked from Wikimedia Commons, with credits shown on hover, so the app needs a network connection to look right.
