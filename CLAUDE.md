# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`pathfinder` ("Finding the patients for the right research") is an entry for the **Biobase challenge, "Find the Right Patients"**, at the DNA Forge AI x Biology Hackathon on November 14, 2026. The answer key is released at 4:00 PM that Saturday, and the challenge page then scores submission files locally in the browser.

The repo contains no application code yet. Everything so far is the challenge data under `data/`.

**Do not propose or build solutions until the user asks.** This file describes the challenges and the data. It does not describe an approach.

## Ground rules

- The global rules in `~/.claude/CLAUDE.md` apply in full: sync `main`, work on a `<type>/<short-kebab-desc>` branch, stage files explicitly, write Conventional Commits with an argued body, open a PR with the Kubernetes-shaped template, and stop. Never merge, and never add Claude attribution to commits or PRs.
- `data/` is git-ignored (about 476 MB). Never commit anything from it, including generated submissions written next to the data. Treat it as read-only input.
- The reports are synthetic and no clinician has reviewed them. Nothing here is for medical decisions.

## The data, at a glance

Everything lives in `data/biobase/`. Each pack has its own README, and those READMEs are the source of truth. Re-read them before working on a task.

| Path | What it holds |
|---|---|
| `pack_reports/` | Synthetic pathology reports and Task A: `train.jsonl` (9,010 labelled), `test.jsonl` (979, text only), `criteria.json` (30 criteria), `criteria_train_answers.json`, `submission_template_criteria.json`, `baseline_keyword.py` |
| `pack_slides/` | Slide metadata and Tasks C1/C2: `slides_manifest.json` (263 slides), `organ_groups.json`, two submission CSV templates, `baseline_slides.py` |
| `tilepack/` | Reduced-resolution tissue tiles (`tiles/<slide id>/<col>_<row>.jpg`, DZI level 12, ~4 µm/px median, range 2 to 16), `tiles.csv`, and `embeddings/` |
| `tilepack/embeddings/` | Precomputed vectors from four models (see below) plus `models.json` |
| `tiles20x/` | A sample of up to 35 full-resolution tiles per slide (~0.5 µm/px, 7,354 in all), `tiles20x.csv`, `blank_tiles.csv` |
| `thumbnails/` | One JPEG per slide, up to 512 px on the long side (~25 µm/px), named `<slide id>.jpg` |

The two halves are **unrelated**: no report matches any slide.

### Embeddings

| Folder | Model | Dim | Files | Note |
|---|---|---|---|---|
| `dinov2/` | DINOv2 ViT-S/14 | 384 | tile + slide | General-purpose, not pathology-specific |
| `hoptimus/` | H-optimus-0 | 1536 | tile + slide | |
| `biomedclip/` | BiomedCLIP | 512 | tile + slide | Only model with a text tower (`"text": true`) |
| `gigapath/` | Prov-GigaPath | 1536 | slide only | Covers 261 of 263 slides |

File formats that are easy to get wrong:

- `tile-vectors.i8` is raw int8, `tiles` rows by `dim` columns. Divide by `scale` (127) for floats.
- `tile-coords.u16` is little-endian `(col, row)` uint16 pairs in the same row order.
- `tile-meta.json` gives each slide's `offset` and `count` into those row-major files. Row `offset + i` is the image `tiles/<slide id>/<col>_<row>.jpg`. `tiles.csv` maps every tile to its row via `index`.
- `slide-vectors.json` is `[{id, v}]`, the L2-normalised mean of the tile vectors.
- `points.json` is a 2-D UMAP of the slides, coordinates in 0 to 1.

## Commands

There is no build, lint, or test tooling yet. The only runnable code is the two baselines (standard library, Python 3; the machine has 3.13):

```sh
# Task A baseline: keyword search, mean F1 0.27, answers 23 of 30 criteria
cd data/biobase/pack_reports
python3 baseline_keyword.py test.jsonl criteria.json my_submission.json

# Tasks C1/C2 baseline: nearest labelled slide by one model's slide vectors
cd data/biobase/pack_slides
python3 baseline_slides.py --manifest slides_manifest.json \
  --vectors ../tilepack/embeddings/gigapath/slide-vectors.json
# writes baseline_organ.csv and baseline_similar.csv
```

`criteria_train_answers.json` makes it possible to score any Task A approach on `train.jsonl` before running it on test.

## The challenges

All tasks are optional. There is no Task B in the provided material.

### Task A: criteria search over reports

For each of the 30 criteria in `criteria.json`, list the `test.jsonl` report ids that match. Scoring is per-criterion precision, recall, and F1; the headline is **mean F1 over all 30**, and a skipped criterion scores 0. A criterion may instead be answered with a count (scored by relative count error), but a count without ids earns no F1. Accepted shapes: a list of ids, `{"ids": [...], "count": n}`, or `{"count": n}`.

**Bar to beat:** mean F1 0.27, with 23 of 30 criteria answered.

What makes it hard:

- **Labels exist only on train.** Test has `id` and `report_text` only, so everything must come from the text.
- **Criteria combine fields, use numbers, and use negation.** For example: "HER2 negative with 1 or more positive lymph nodes" (c01), "within 3 mm of the closest margin" (c05), "5 cm or larger with no perineural invasion" (c13), "positive surgical margin and no extraprostatic extension" (c19). The keyword baseline ignores numbers and negation by design.
- **Criteria use clinical concepts the text may not spell out literally.** For example: "stage III" (c08), pT4 (c12, c23), Grade Group 4 or 5 versus a written Gleason score (c17), triple-negative (c03), "poorly cohesive (diffuse type, not mixed)" (c22), "not well differentiated (grade 2 or 3)" (c28), extension into renal vein or IVC (c26).
- **Demographic and date filters** come from the header line (`Patient: F, age band 60-69 | Report year: 2024`): c06 (under 60), c29 (70 or older), c30 (female, 2023 or 2024). Ages are only given as bands.
- **The final diagnosis block has two layouts**: one field per line, or all fields on one line separated by `; `. Some values (Margins, Biomarkers) contain `;` themselves, so naive splitting breaks.
- **Tumor sizes are scored as written in the report**, so units and phrasing in the text matter.
- **Gross and microscopic prose is LLM-written** from structured fields and can paraphrase or contradict the coded diagnosis block. The title line and final diagnosis block are written by code.
- **Field sets differ by organ** (listed in `pack_reports/README.md`). Skin alone has around 30 possible fields and 12 diagnosis groups.
- **The train mix is skewed toward skin** (2,714 of 9,010), while 28 of 30 criteria target breast, colon, lung, prostate, stomach, or kidney.
- Known gaps: no PD-L1 or Ki-67 anywhere; HER2/ER/PR only in breast, HER2 also in stomach; colon MMR results appear in text but not labels; years run 2021 to 2024 only; staging follows AJCC 8th edition "written from memory".

### Task C1: organ labels for held-out slides

For each of the 67 held-out slides (`"heldout": true`), name the organ. An answer counts if it matches the true organ or organ group, case-insensitive. Score: accuracy. Submit `submission_template_slides_organ.csv` (`slide_id,organ`).

### Task C2: similar slides

For each held-out slide, return up to 10 other slides from the whole set (labelled and held-out). Score: precision at 1, 5, and 10, counted as the share of results in the same organ group as the query. Submit `submission_template_slides_similar.csv` (`query_slide_id,rank,result_slide_id`).

**Bars to beat** (baseline, nearest labelled slide by slide vector; guessing the biggest group gets about 15% on C1):

| Model | C1 accuracy | C2 P@1 | P@5 | P@10 |
|---|---|---|---|---|
| DINOv2 | ~35% | ~0.4 | ~0.3 | ~0.3 |
| H-optimus-0 | ~55% | ~0.6 | ~0.5 | ~0.3 |
| BiomedCLIP | ~60% | ~0.6 | ~0.5 | ~0.4 |
| Prov-GigaPath | ~70% | ~0.8 | ~0.7 | ~0.6 |

What makes C1 and C2 hard:

- **Very small data.** 263 slides; 196 labelled (6 of those have no organ); 67 held out across 15 organ groups. Scores are a rough signal, and some groups have only 2 to 5 labelled examples (Esophagus 2, Bone 4, Bone marrow 4, Skin 5).
- **Supplier imbalance.** SUPP-02 provides 185 slides, SUPP-01 59, SUPP-04 12, SUPP-03 7. The dominant supplier names organs in a cancer-atlas style, and suppliers record different metadata fields. Organ names are raw supplier strings; `organ_groups.json` maps them to the 15 scoring groups.
- **Mixed stains.** 233 H&E, 22 IHC, 8 special stains. Embedding models are mostly trained on H&E.
- **Resolution mismatch.** The tile pack is ~4 µm/px (and varies 2 to 16 across slides), far coarser than the ~0.5 µm/px most pathology models expect. The 0.5 µm/px pack is only a sample of up to 35 tiles per slide. The original whole-slide files are not available.
- **Blank tiles in the 20x pack.** 461 tiles (6%) are black or empty glass and should be removed first using `blank_tiles.csv`. SUPP-04 is worst (70% blank), and `SUPP-04-bb_04_wsi_0009` has no usable 20x tile at all.
- **Coverage gaps.** Prov-GigaPath, the strongest baseline, has slide vectors only (no tile vectors) and is missing 2 slides.
- **Metadata leakage risk.** The manifest carries fields like `report`, `diagnosis`, `stain`, `scanner`, and `supplierId` for labelled slides, and held-out slides still carry `supplierId`, `stain`, `scanner`, `vendor`, `magnification`, `mpp`, `ageBucket`, and `sex`. Whether a shortcut through metadata is acceptable is a judgement call to raise with the user, not to assume.
- HistAI is a separate public slide set the challenge allows as extra training data. Biobase does not host it and it is not in `data/`.
