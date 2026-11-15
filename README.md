# Pathfinder

**Team ID:** T20  
**Hackathon:** DNAForge Nov 14th Hackathon  
**Sponsor prizes entered:** Biobase Category: $500

---

## Problem

Biomedical researchers face challenges when it comes to finding correct patient records and samples they need for research. Identifying reports that satisfy all of the criteria requires extensive analysis.

Our goal is to make biomedical research data easier to search and interpret. Researchers should describe the characteristics they are looking for in simple words and then be given reports that match their queries.

## What We Built

Pathfinder extracts structured clinical fields from pathology reports with an LLM extractor (gpt-5.4-mini, fixed JSON schema), then matches research criteria with declarative rules.
On the Biobase train set (9,010 reports), this reached mean F1 **1.00** across all 30 criteria, versus the keyword baseline of ~0.27, with field accuracy at or near 1.0 on most fields.

A query layer answers new plain-English questions: an LLM turns each question into a validated rule that runs over the saved extractions, and any part the schema can't express is checked report by report with a quoted line of evidence.

We built a web application where anyone can upload their files and use our extractor tool to research and find their patients and clinical trials, without being restricted to a Biobase dataset.

## Run or View the Demo

Biobase packs go under `data/` (git-ignored, not shipped in the repo).

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
printf 'NEBIUS_API_KEY=...\nOPENAI_API_KEY=sk-...\n' > .env              # git-ignored
.venv/bin/python -m pathfinder.cli extract --split train --limit 210   # organ-stratified sample
.venv/bin/python -m pathfinder.cli evaluate                            # P/R/F1 and field accuracy vs train labels
.venv/bin/python -m pathfinder.cli query "Kidney cancers larger than 7 cm, broken down by extent of invasion"
.venv/bin/python -m pathfinder.cli submit                              # write outputs/submission.json for Task A
.venv/bin/streamlit run streamlit_app.py                               # web UI
.venv/bin/python -m unittest discover -s tests                         # offline tests
```

The web UI has three pages: **Find** (ask a question, see the count, breakdown and matching reports, and open any report next to its key findings), **Criteria** (the 30 rules scored on train), and **Reports** (look up a report by id). It needs a network connection for its fonts and images, and it asks before any per-report LLM check that costs tokens.

### Run the web UI

The UI is a Streamlit app, and it starts a local server. Run all commands from the repo root.

1. Do the one-time setup:

   ```sh
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```

2. Put both keys in `.env` at the repo root:

   ```
   NEBIUS_API_KEY=...
   OPENAI_API_KEY=sk-...
   ```

   Questions on the **Find** page need `NEBIUS_API_KEY`. Extracting an uploaded dataset with `gpt-5.4-mini` needs `OPENAI_API_KEY`. Without a key, the app still opens: the settings popover warns about a missing Nebius key, and the upload dialog warns about a missing OpenAI key.

3. Make sure there are saved extractions to search, in `outputs/extractions/`. If there are none, the app shows an error and stops. To make a small set:

   ```sh
   .venv/bin/python -m pathfinder.cli extract --split train --limit 210
   ```

   You can also upload reports from inside the app (step 5).

4. Start the server:

   ```sh
   .venv/bin/streamlit run streamlit_app.py --server.headless true
   ```

   Open http://localhost:8501. The `--server.headless true` flag skips Streamlit's first-run email prompt and doesn't open a browser. Leave it out to have Streamlit open a browser tab for you. To use another port, add `--server.port 8502`. To reach the app from other machines on your network, use the "Network URL" it prints.

5. To search reports other than train, click the settings icon at the top right, then **Upload a dataset**.
   - Choose a `.jsonl` file shaped like `test.jsonl`. To search the test set, upload `test.jsonl` itself.
   - The dialog shows how many reports need extracting and roughly how many tokens that will use. Nothing is sent until you click the extract button.
   - After extraction, the app switches to the new dataset.
   - Uploads are saved to `outputs/datasets/`, and their extractions to `outputs/extractions/`.

6. Stop the server with `Ctrl+C`.

   Streamlit reruns `streamlit_app.py` when you save it, but it keeps the `pathfinder/` modules it loaded at startup. After you change code under `pathfinder/`, restart the server, or you will see stale behavior or an `ImportError`.

## Evidence and Limitations

On train, the extractor-plus-rules pipeline scored mean F1 **1.00** on all 30 criteria (9,010 reports); a leave-one-out query-layer check hit mean F1 **0.9999** (29/30 exact). Field accuracy was 1.0 on 23 of 31 fields (lowest pN ≈ 0.985).

## Data and Permissions

Pathfinder uses datasets provided through Biobase and DNAForge, which includes Biobase synthetic pathology reports, Biobase tissue slide metadata, Biobase pathology thumbnails and tiles, and provided image embeddings.

## Next Step

Our next step includes to put our web-based site into an uniformed application, expanding its capabilities and becoming the main assistant for researchers in medicine.

---

**Pathfinder: Finding the patients for the right research.**

Built for the DNAForge × Biobase Hackathon, November 2026.
