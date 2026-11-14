# Pathfinder

**Team ID:** T20  
**Hackathon:** DNAForge Nov 14th Hackathon  
**Sponsor prizes entered:** Biobase Category: $500

---

## Problem

Biomedical researchers face a major challenge when it comes to finding the correct patient records and tissue samples needed for their research.

Pathology reports contain valuable information, such as cancer diagnoses, tissue types, biomarkers, and disease characteristics. However, identifying reports that satisfy all of the research criteria requires extensive manual review.

At the same time, digital pathology slides have valuable visual information that is difficult to organize, search, and efficiently compare.

**Our goal is to make biomedical research data easier to search, interpret, and explore using AI.**

A successful solution would allow researchers to describe the clinical characteristics they are looking for in simple words, retrieve relevant pathology reports, and explore associated tissue samples through an interface.

## What We Built

**Pathfinder** is a web-based research platform that simplifies the discovery of pathology reports and tissue samples.

Our platform combines clinical search, report retrieval, and an intuitive research dashboard.

### 1. Natural-Language Patient Search

Researchers can enter clinical criteria using everyday medical language.

For example:

"Find patients with HER2-negative breast cancer and lymph node involvement."

Pathfinder is designed to identify pathology reports that satisfy the requested criteria and present the results.

### 2. Pathology Report Dashboard

Our dashboard is designed to help researchers:

- Browse matching pathology reports.
- View report IDs and relevant clinical information.
- Inspect full report descriptions.
- Identify evidence supporting a potential match.
- Filter and explore search results.

### 3. AI-Assisted Tissue Slide Exploration

Pathfinder also aims to support digital pathology analysis through:

- Tissue slide visualization.
- Organ classification.
- Similar-slide retrieval.
- Side-by-side comparison of histopathology samples.

These features support the broader goal of making biomedical data searchable.

## Run or View the Demo

**Live demo:** Not deployed yet  
**GitHub repository:** https://github.com/HaseebKhanYT/pathfinder  
**Demo video:** Not recorded yet

### Running Locally

Prerequisites:

- Python 3.11+ (stdlib HTTP server; no Node.js required for the website)
- Optional: `streamlit>=1.39.0` for the earlier Streamlit prototype UI (`requirements.txt`)
- Biobase challenge data under `data/biobase/` when available (git-ignored; not redistributed in this repo)

```sh
# Website (stdlib HTTP server on http://127.0.0.1:8000)
python3 -m server.main

# Optional Streamlit prototype
pip install -r requirements.txt
streamlit run app.py
```

### Demo Walkthrough

1. Open Pathfinder.
2. Enter a clinical search query.
3. Submit the query using the search button.
4. Review the returned pathology reports.
5. Select a report to inspect its details.

## Evidence and Limitations

### Dataset

The Biobase challenge provides:

| Dataset | Description |
|---|---|
| Synthetic reports | 9,010 labeled reports |
| Test reports | 979 reports |
| Search criteria | 30 clinical criteria |
| Pathology slides | 263 slides |
| Held-out slides | 67 slides |
| Slide tiles | 12,276 tiles with embeddings |
| High-resolution tiles | 7,354 tiles at approximately 20× magnification |

### Evaluation Approach

We aim to evaluate Pathfinder using the provided challenge datasets.

**Clinical report retrieval:** Compare the reports selected by our system against the provided labels or baseline, using appropriate measures such as precision, recall, and F1 score.

**Organ classification:** Evaluate whether predicted tissue organ groups match known slide labels.

**Similar-slide retrieval:** Assess whether retrieved slides share relevant known characteristics with the query slide.

### Limitations

Pathfinder is a research prototype, not a validated clinical decision-support system.

Current limitations may include:

- Synthetic reports may not reflect the complexity of real clinical documentation.
- Model predictions may contain incorrect matches or classifications.
- Similarity between tissue images does not guarantee clinical or diagnostic equivalence.
- Available pathology information may not be sufficient to establish actual clinical-trial eligibility.
- Additional validation would be required before applying the system to real patient data.

## Data and Permissions

Pathfinder uses datasets provided through the Biobase DNAForge hackathon.

Data sources include:

- Biobase synthetic pathology reports.
- Biobase tissue slide metadata.
- Biobase pathology thumbnails and tiles.
- Provided image embeddings.

## Next Step

Our next priority is to validate Pathfinder's retrieval performance quantitatively against the provided baseline.

We would run the system on held-out reports, compare predicted matches against the challenge's evaluation labels, and analyze false positives and false negatives.

This experiment would help determine whether Pathfinder can reliably identify relevant pathology reports and which types of clinical criteria require improvement.

---

**Pathfinder: Finding the patients for the right research.**

Built for the DNAForge × Biobase Hackathon, November 2026.
