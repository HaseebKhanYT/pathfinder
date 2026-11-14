# Task A: possible solutions

> Options for beating the keyword baseline on criteria search over the synthetic pathology reports.

This document lists approaches and what the data says about each. Nothing here is built yet. The only score that has been measured is the keyword baseline on `train.jsonl`; every other number is a count taken from the training labels, and the "expected" outcomes are estimates.

## Contents

- [Where the baseline loses points](#where-the-baseline-loses-points)
- [What the data makes possible](#what-the-data-makes-possible)
- [Solution options](#solution-options)
- [Recommendation](#recommendation)
- [What each criterion actually means](#what-each-criterion-actually-means)
- [How to evaluate any approach](#how-to-evaluate-any-approach)
- [Open questions](#open-questions)

## Where the baseline loses points

`baseline_keyword.py` matches a report when it contains every content word of the criterion, after dropping stop words and all numbers. Scored against `criteria_train_answers.json`, it gets a mean F1 of **0.285** on train (the README quotes 0.27 on test). The per-criterion results fall into three failure modes:

| Failure mode | Criteria | What happens |
|---|---|---|
| Returns nothing (F1 0.00) | c03, c05, c08, c12, c22, c23, c28 | Every content word of the criterion is required, and the criterion includes a word that matching reports do not contain, such as "triple" (c03), "III" (c08), or the bare token "pT4" where reports say "pT4a" (c12, c23). These are the 7 unanswered criteria. |
| Returns the whole organ (recall 1.00, precision 0.06 to 0.69) | c01, c02, c04, c06, c07, c10, c11, c13, c14, c16, c17, c18, c19, c20, c21, c24, c25, c30 | Every report for an organ contains the field *names* ("Lymphovascular invasion", "HER2", "Margins"), so the words match whether the value is "present" or "not identified". For example, c02 returns 1,284 of the 1,290 breast reports for 127 true matches. |
| Misses on wording (recall under 0.5) | c26, c27, c29 | The criterion's phrasing differs from the report's: "node-positive" versus "3 of 17 positive", "involved deep margin" versus "carcinoma extends to the deep margin". |

Only c09 (mucinous adenocarcinoma of the colon) scores 1.00, because the answer is a literal phrase.

The root cause is the same in all three: the baseline looks for words anywhere in the text, while the criteria ask about the **value of a specific field**, often compared against a number or negated.

## What the data makes possible

Three measured facts shape the options:

- **The final diagnosis block is the label.** All 86,040 `labels.diagnosis` values in train appear verbatim in their report text. The block is written by code, not by the language model, so the field values can be recovered exactly by parsing.
- **The value vocabularies are small and templated.** For example: stomach `HER2` has 4 shapes, colon `Tumor deposits` has 2, kidney `Extent of invasion` has 8, breast `Margins` has 13. Numbers are the only free part.
- **Both layouts appear in equal measure.** Train is 4,452 one-field-per-line and 4,558 inline (`; `-separated); test is 486 and 493. A parser must handle both, and the inline layout is the hard one because `Margins`, `Biomarkers`, `Lymph nodes`, and stomach `Histologic type` contain `; ` inside their values.

The header line (`Patient: F, age band 60-69 | Report year: 2024`) and the `SPECIMEN:` line are also code-written, which covers organ, sex, age band, and year.

## Solution options

### Option 1: parse the diagnosis block, then apply one rule per criterion

Parse each report into a dictionary of fields (header, specimen, diagnosis block). Write each of the 30 criteria as a small predicate over those fields, for example c04: `organ == breast and LVI starts with "present" and tumor_size_mm > 20`.

- **Inline layout:** split on `; ` only where the next segment starts with a known field name for that organ. The README lists every field name per organ, so the split points are unambiguous.
- **Parser check:** the parser is correct when it reproduces `labels.diagnosis` exactly on all 9,010 train reports. That is a pass/fail test, independent of the criteria.
- **Derived values:** read pT and pN from the `Pathologic stage` field rather than re-deriving stage from depth and node counts. Read Grade Group from the `Gleason score` text ("4+4=8, Grade Group 4").

Strengths: deterministic, runs in seconds, needs only the standard library, and every miss is explainable by reading one report. Train answers give a direct F1 for each rule.

Weaknesses: 30 hand-written rules, and it depends on test following the same templates as train. A template that appears only in test would be silently missed.

Expected: close to 1.00 mean F1 if the templates hold, since the labels come from the same fields.

### Option 2: language model extracts fields, rules decide

Send each report to a language model with a fixed JSON schema (organ, ER, PR, HER2, positive node count, tumor size and unit, margin status and distance, and so on), then apply the same predicates as option 1.

Strengths: tolerant of layout and wording changes; no splitter to write.

Weaknesses: 979 calls for test, plus a labelled sample to validate; extraction errors on numbers and negation are the usual failure; results can vary between runs. It replaces a parser that can be verified exactly with one that can only be sampled.

Worth it mainly as a **fallback** for reports where the option 1 parser fails or finds an unknown field value.

### Option 3: language model judges each report against each criterion

Ask the model directly: "Does this report match this criterion?" To keep the cost down, first filter by organ so each report is judged against only its own organ's criteria (3 to 7 each instead of 30), and ask for all of them in one call.

Strengths: least code; handles clinical paraphrase without any rule writing.

Weaknesses: the criteria carry boundary conventions that a model has to guess (see [the table below](#what-each-criterion-actually-means)): is "within 3 mm" inclusive, do isolated tumor cells count as a positive node, is an equivocal HER2 positive? Train answers settle these, so they would have to go into the prompt as explicit rules, at which point most of option 1 has been written anyway. The prose sections can also contradict the coded block, which gives the model two sources to reconcile.

### Option 4: train a classifier per criterion

Use the train answers as labels and fit 30 binary classifiers on TF-IDF n-grams or sentence embeddings.

Strengths: no rule writing; uses the 9,010 labelled reports directly.

Weaknesses: a poor fit for this task. Bag-of-words features cannot represent "tumor size greater than 20" or "margin distance at most 3", and several criteria have few positives (c25 has 32, c15 has 42, c10 has 52). It would likely land well above the baseline but below options 1 to 3 on the numeric and combined criteria.

### Option 5: field-aware keyword search

Keep the baseline's shape but fix its three failure modes: search within a field's value rather than the whole text, add a synonym table (triple-negative, node-positive, stage III, pT4), and add number comparisons.

Strengths: smallest step from the baseline; a useful story if the write-up needs to show incremental gains.

Weaknesses: once it reads field values and compares numbers, it is option 1 with a less explicit design.

### Option 6: rules first, language model as auditor

Run option 1, then use a language model in two narrow places:

- **Fallback:** reports where the parser finds a missing or unrecognised field value.
- **Audit:** a sample of matches and non-matches per criterion on test, to catch a test-only template that train never showed.

This keeps the deterministic path as the source of the submission and spends model calls only where the rules cannot vouch for themselves.

## Recommendation

Start with **option 1**, and add the option 6 audit if time allows.

The deciding fact is that the answer key is generated from the same structured fields that are printed verbatim in the diagnosis block. A parser can be proved correct against 86,040 known values, and each rule can be scored against train answers before test is touched. Options 2 to 4 trade that certainty for flexibility the data does not appear to need.

Options 2 and 3 become the better choice only if the parser check fails on a meaningful share of reports, or if test turns out to contain templates absent from train.

## What each criterion actually means

The criteria are plain language, but the train answers apply specific conventions. These were read off `criteria_train_answers.json` and hold for every train report in the relevant organ.

| Criterion | Convention in the train answers |
|---|---|
| c01 | Breast HER2 "negative" includes score 2+ with in situ hybridization not amplified. Positive nodes means pN1 or higher: `pN0(i+)` (isolated tumor cells only) does **not** count. |
| c04 | "Larger than 20 mm" is strict. A tumor of exactly 20 mm with LVI present is excluded (6 such reports). |
| c05 | "Within 3 mm" is inclusive: 2 mm and 3 mm match, 4 mm does not. Requires `Margins` to start with "negative". |
| c06 | "Under 60" means age bands up to `50-59`. |
| c08 | Stage III includes `pN1c`: 3 reports with 0 positive nodes but tumor deposits present are matches. Use pN, not the node count. |
| c12, c23 | pT4a and pT4b both match. |
| c13 | "5 cm or larger" is inclusive. Colon sizes are in cm, breast sizes in mm. |
| c15 | Squamous histology and "bronchial margin involved by carcinoma". The 7 adenocarcinomas with an involved bronchial margin are excluded. |
| c17 | Grade Group is written out in the `Gleason score` field, so no Gleason arithmetic is needed. |
| c19 | "No extraprostatic extension" means "not identified (organ-confined)". Focal extension is excluded. |
| c21 | Stomach HER2 "equivocal (score 2+), in situ hybridization not amplified" is **not** positive. |
| c22 | "Poorly cohesive carcinoma, diffuse type" only. "Mixed adenocarcinoma" is excluded even though it has a poorly cohesive component. |
| c24 | ISUP grade 3 or 4. All 855 kidney reports in train are clear cell, so the histology condition never excludes anything there; keep it in the rule in case test differs. |
| c26 | Includes segmental branches of the renal vein, the main renal vein, and the inferior vena cava (27 + 36 + 46 = 109 matches). |
| c27 | Six different phrasings state an involved deep margin, including "carcinoma extends to the peripheral and deep margin". "Cannot be evaluated" is excluded. |
| c28 | Grade is inside the `Diagnosis` text: "moderately differentiated (G2)" or "poorly differentiated (G3)". |

Conventions not yet checked against train: c02, c03, c07, c09 to c11, c14, c16, c18, c20, c25, c29, c30. Several of these reuse a convention above: c11, c16, and c29 all depend on what "positive lymph nodes" means, and c29 also on whether `pN1c` counts as node-positive.

## How to evaluate any approach

- **Score on train first.** `criteria_train_answers.json` gives per-criterion precision, recall, and F1. Report the mean over all 30, because a skipped criterion scores 0.
- **Read the errors, not only the score.** With templated data, each false positive or false negative usually points at one missed phrasing or one wrong boundary.
- **Hold out part of train** when a method is fitted to it (options 4, and prompt tuning in options 2 and 3). Rules written by reading train are also fitted to it, so keep a slice unread for a final check.
- **Compare test against train before submitting.** Test has no labels, but the share of reports per organ and the share matching each criterion should be close to train's. A criterion whose match rate on test is far off its train rate is the signal for a test-only template.
- **Check that every submitted id is in `test.jsonl`.** Ids outside it are invalid.
- **Submit ids, not only counts.** A count without ids earns no F1. `{"ids": [...], "count": n}` earns both scores.

## Open questions

- **Does test use only templates seen in train?** Train is nine times larger, so it probably covers them, but this is the main risk for option 1 and the reason for the option 6 audit.
- **Which source wins when prose and the diagnosis block disagree?** The train answers agree with the coded block wherever this was checked, so the block should win. This has not been tested on a report with a known contradiction.
- **Is a language model API available and allowed on the day?** Options 2, 3, and 6 depend on it. Option 1 does not.
- **Do the unchecked criteria hide another boundary case?** The remaining 13 criteria need the same read-off against train answers before their rules are trusted.
