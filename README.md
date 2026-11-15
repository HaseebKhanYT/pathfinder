# Pathfinder

**Team ID:** T20  
**Hackathon:** DNAForge Nov 14th Hackathon  
**Sponsor prizes entered:** Biobase Category: $500

---

## Problem

Biomedical researchers face challenges when it comes to finding correct patient records and samples they need for research. Identifying reports that satisfy all of the criteria requires extensive analysis. 

Our gaol is to make biomedical research data easier to search and interpret. Researchers should describe the characeristics they are looking for in simple words and then be given reports that match their queries. terface.

## What We Built

Pathfinder extracts structured clinical fields from pathology reports with an LLM extractor (gpt-5.4-mini, fixed JSON schema), then matches research criteria with declarative rules.
On the Biobase train set (9,010 reports), this reached mean F1 **1.00** across all 30 criteria, versus the keyword baseline of ~0.27, with field accuracy at or near 1.0 on most fields.

We built a web application where anyone can upload their files and use our extractor tool to research and find their patients and clinical trials, without being restricted to a Biobase dataset.  

## Run or View the Demo

Run the CLI locally with an OpenAI key in `.env`: extract reports (`python -m pathfinder.cli extract`), score against train labels (`evaluate`), ask plain-English cohort questions (`query`), or write a Task A submission (`submit`).
A Streamlit UI exercises the same query layer and criteria matching for interactive demos; Biobase packs stay under `data/` and are not shipped in the repo.

## Evidence and Limitations

On train, the extractor-plus-rules pipeline scored mean F1 **1.00** on all 30 criteria (9,010 reports); a leave-one-out query-layer check hit mean F1 **0.9999** (29/30 exact). Field accuracy was 1.0 on 23 of 31 fields (lowest pN ≈ 0.985).

## Data and Permissions

Pathfinder uses datasets provided through Biobase and DNAForge, which includes Biobase synthetic pathology reports, Biobase tissue slide metadata, Biobase pathology thumbnails and tiles, and provided image embeddings.

## Next Step

Our next step includes to put our web-based site into an uniformed application, expanding its capabilities and becoming the main assistant for researchers in medicine. 

---

**Pathfinder: Finding the patients for the right research.**

Built for the DNAForge × Biobase Hackathon, November 2026.
