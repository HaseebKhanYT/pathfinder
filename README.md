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
