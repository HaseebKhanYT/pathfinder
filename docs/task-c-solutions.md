# Tasks C1 and C2: possible solutions

> Options for beating the nearest-slide baseline on organ labelling (C1) and similar-slide retrieval (C2).

This document lists approaches and what the data says about each. No submission has been produced. The measured numbers below come from **leave-one-out checks on the labelled slides**, using only the precomputed slide vectors; they are not scores on the 67 held-out slides, and the caveats in [How far to trust these numbers](#how-far-to-trust-these-numbers) apply to all of them. Options marked "not measured" are untested.

## Contents

- [What the baseline does and where it loses points](#what-the-baseline-does-and-where-it-loses-points)
- [What the data makes possible](#what-the-data-makes-possible)
- [Options for C1](#options-for-c1)
- [Options for C2](#options-for-c2)
- [The metadata question](#the-metadata-question)
- [Recommendation](#recommendation)
- [How to evaluate any approach](#how-to-evaluate-any-approach)
- [How far to trust these numbers](#how-far-to-trust-these-numbers)
- [Open questions](#open-questions)

## What the baseline does and where it loses points

`baseline_slides.py` takes one model's slide vectors. For C1 it copies the organ group of the single most similar labelled slide. For C2 it lists the 10 most similar other slides by cosine similarity.

Repeating that rule leave-one-out on the 190 labelled slides that have an organ group (each slide is scored against the other 189):

| Slide vectors | C1 accuracy | C2 P@1 | P@5 | P@10 |
|---|---|---|---|---|
| DINOv2 | 0.56 | 0.56 | 0.34 | 0.27 |
| H-optimus-0 | 0.58 | 0.58 | 0.40 | 0.31 |
| BiomedCLIP | 0.58 | 0.58 | 0.43 | 0.34 |
| Prov-GigaPath | 0.78 | 0.78 | 0.56 | 0.45 |

Compared with the README's held-out figures, the model ranking is the same and Prov-GigaPath leads by a wide margin in both, so leave-one-out is a usable stand-in for comparing methods. The absolute values differ: C1 is higher here (Prov-GigaPath 0.78 against about 70%, DINOv2 0.56 against about 35%), and P@5 and P@10 are lower (0.56 and 0.45 against about 0.7 and 0.6), partly because a held-out query can draw results from all 262 other slides instead of 189.

Where the points go:

- **One neighbour decides everything.** C1 uses a single nearest slide, so one odd neighbour flips the answer. The other 188 labels are unused.
- **Precision falls with depth.** Prov-GigaPath drops from 0.78 at rank 1 to 0.45 at rank 10. Raw similarity finds one same-organ slide and then drifts to slides that look alike for other reasons.
- **Non-H&E slides are much weaker.** Prov-GigaPath gets 0.81 on the 165 H&E slides and 0.60 on the 25 IHC and special-stain slides.
- **Some organ groups are hard for the vectors.** Prov-GigaPath per group: Kidney, Prostate, Skin, Bone, and Esophagus 1.00; Colon and rectum 0.83; Uterus 0.81; Stomach 0.80; Thyroid 0.79; Liver 0.77; Bone marrow 0.75; Lung 0.71; Breast 0.67; Other or mixed 0.55; Bladder 0.50.
- **The vectors come from coarse tiles.** They are means over tiles at about 4 µm/px, far from the roughly 0.5 µm/px these models expect.

## What the data makes possible

- **The held-out set is narrower than the labelled set.** 57 of the 67 held-out slides are from SUPP-02, 8 from SUPP-01, 2 from SUPP-03, and none from SUPP-04. 62 are H&E and 5 are IHC.
- **SUPP-02 is uniform.** All 128 labelled SUPP-02 slides are H&E, and they cover only 10 of the 15 groups: Colon and rectum 20, Kidney 13, Liver 13, Thyroid 13, Uterus 13, Bladder 12, Prostate 12, Breast 11, Lung 11, Stomach 10. Performance on SUPP-02 H&E therefore decides most of the score.
- **Small groups cap C2.** Esophagus has 2 labelled slides, Bone 4, Bone marrow 4, Skin 5. A query from those groups cannot fill 10 correct results from labelled slides. With the true group known for every query, the labelled pool supports at most 0.97 at P@5 and 0.95 at P@10.
- **Tile coverage is uneven.** The tile pack has 12,276 tiles, from 1 to 150 per slide (median 40); 9 slides have fewer than 10. After removing blanks, the 20x pack has 0 to 35 usable tiles per slide (median 31); 26 slides have fewer than 10, including 6 held-out slides.
- **Prov-GigaPath is missing two slides**, `SUPP-03-bb_03_wsi_0004` and `SUPP-03-bb_03_wsi_0005`. Both are labelled ("Other or mixed"), so every held-out slide has a Prov-GigaPath vector.
- **Six labelled slides have no organ group** (all SUPP-04). They cannot be used as training labels, and returning one in C2 can never score.

## Options for C1

### Option 1: train a classifier on the slide vectors

Replace the single nearest neighbour with a classifier that uses all labelled slides: ridge or logistic regression, linear discriminant analysis, or a class-centroid rule.

Measured, leave-one-out, ridge one-versus-rest on Prov-GigaPath vectors (centred, L2-normalised): **0.88** against 0.78 for nearest neighbour. On SUPP-02 slides it reaches 0.90, and on non-H&E slides 0.72. The regularisation strength was picked on the same leave-one-out run, so 0.88 is optimistic; the neighbouring settings gave 0.84 and 0.87.

Strengths: needs only the vectors already in the pack and a few lines of linear algebra; the largest measured gain for the least work.

Weaknesses: with 2 to 5 examples in the smallest groups, the classifier has little to learn from there.

### Option 2: combine the four models

Each model sees different things, so a combination can beat the best single one.

Measured: the simple version does **not** help. Adding the cosine similarities of Prov-GigaPath, H-optimus-0, and BiomedCLIP gives 0.76, below Prov-GigaPath alone (0.80 centred), because the two weaker models outvote the strong one. A ridge classifier on the three concatenated matches Prov-GigaPath alone overall (0.88) but is better on non-H&E slides (0.76 against 0.72) and on the three smaller suppliers (0.86 against 0.84).

So a combination needs weights (learned, or by each model's own leave-one-out accuracy), and its likely value is on the slides where Prov-GigaPath is weakest.

### Option 3: use the tile vectors instead of the slide mean

A slide vector is a plain average, so fat, stroma, and blood count as much as the informative tissue. With tile vectors (available for H-optimus-0, BiomedCLIP, and DINOv2) each tile can vote: classify tiles, then pool the votes per slide, or weight tiles by how organ-specific their neighbours are. Not measured.

Strengths: uses 12,276 training points instead of 190; can ignore uninformative tiles.

Weaknesses: Prov-GigaPath, the strongest model, has no tile vectors, so this starts from a weaker model. Tiles from one slide are not independent, so validation must hold out whole slides.

### Option 4: re-embed the 0.5 µm/px tiles

Run a pathology model on the `tiles20x/` sample (6,893 tiles once the 461 blanks are removed), which is the resolution those models were trained at, and pool per slide. Not measured.

Strengths: addresses the resolution mismatch, which the pack's README names as a way to raise the scores.

Weaknesses: the most setup. The strong pathology models (H-optimus-0, Prov-GigaPath, UNI, CONCH) are gated on Hugging Face and need access approval, a download, and enough compute to embed about 7,000 tiles. 26 slides have fewer than 10 usable tiles and one has none, so the tile-pack vectors are still needed as a fallback.

### Option 5: zero-shot text prompts with BiomedCLIP

BiomedCLIP has a text tower. Embed a prompt per organ group ("H&E stained section of thyroid tissue") and compare it with the image vectors. Not measured.

Strengths: needs no labelled examples, which is the one thing the smallest groups (Esophagus, Bone, Bone marrow, Skin) lack.

Weaknesses: the text tower is not in the pack, so the model must be downloaded. BiomedCLIP is the third-best image model here, so on its own this probably trails option 1. Most useful as an extra vote for rare groups.

### Option 6: ask a vision language model

Show a general vision model the thumbnail and a few 0.5 µm/px tiles, and ask for the organ from the list of 15 groups. Not measured.

Strengths: independent of the embeddings, so its errors differ from theirs; can read stain type and gross tissue shape from the thumbnail; 67 slides is a small job.

Weaknesses: needs an API on the day; accuracy on organ-of-origin from histology is unknown until tested on labelled slides; answers can vary between runs.

### Option 7: handle non-H&E slides separately

The 5 held-out IHC slides are the weakest spot for every embedding model. Options: compare IHC slides only with other IHC slides; rely more on the combined-model classifier from option 2; or use the thumbnail, which still shows tissue shape. Not measured, and it affects at most 5 of 67 answers.

### Option 8: add HISTAI slides as training data

More labelled slides per organ would help the small groups most. The cost is high: gated access that may not arrive in time, multi-gigabyte whole-slide files, tiling and embedding from scratch, and a mapping from HISTAI's `specialization` and `icd10` to the 15 groups. The new embeddings would also have to be comparable with the pack's (same model, similar resolution and pooling), and how Biobase produced its Prov-GigaPath slide vectors is not documented. Likely out of reach within a hackathon day.

## Options for C2

### Option A: answer C1 first, then rank within the predicted group

C2 is scored only on organ group, so once a query's group is predicted, the best results are labelled slides known to be in that group, ordered by similarity. Slides from other groups follow only if the group runs out.

Measured, leave-one-out, using the option 1 classifier:

| Ranking | P@1 | P@5 | P@10 |
|---|---|---|---|
| Similarity only (baseline rule, centred) | 0.80 | 0.58 | 0.46 |
| Predicted group first | 0.88 | 0.85 | 0.83 |
| True group first (ceiling) | 1.00 | 0.97 | 0.95 |

This makes C2 precision roughly equal to C1 accuracy at every depth, so any C1 improvement carries straight over.

Weakness: it is all-or-nothing per query. When the predicted group is wrong, all 10 results are wrong. A softer version orders candidates by the classifier's probability for each candidate's group, multiplied by similarity.

### Option B: learn a better similarity

Project the vectors into a space where same-organ slides sit closer (linear discriminant analysis, or the classifier's output scores), then rank by similarity there. Not measured. It degrades more gently than option A when the top prediction is wrong, and it can rank held-out slides too.

### Option C: re-rank using the neighbourhood

Standard retrieval refinements: average a query with its top few neighbours and search again, or prefer pairs that are in each other's top results. Not measured. These need no labels, so they can use all 263 slides, but their gain is usually modest next to option A.

### A choice both options face: labelled or held-out results

Results may come from the whole set, including other held-out slides. A labelled slide's group is known; a held-out slide's group is only predicted. Preferring labelled slides is safer. Held-out slides are worth returning only when a group has too few labelled members to fill the list.

## The metadata question

Several manifest fields carry signal about the organ without looking at the tissue. Whether to use them is a decision for the team, and possibly for the organisers. What was found:

- **Supplier narrows the label set.** Restricting a SUPP-02 slide to the 10 groups SUPP-02 is known to supply removes five wrong answers at no cost. A supplier-only guess scores 0.20 leave-one-out, so it is a prior, not an answer.
- **Sex rules out groups.** Where sex is recorded, it excludes Prostate or Uterus. It is recorded for only 8 of the 67 held-out slides.
- **SUPP-01 demographics can identify a case.** SUPP-01 appears to supply several slides per patient: 5 labelled IHC slides are "F, 60 to 69, Breast", and 5 held-out IHC slides are also "F, 60 to 69". If those are the same case, the held-out answer is readable from the labels. This is the strongest shortcut found and the one most worth asking about.
- **SUPP-02 id order is weakly informative.** Organs cluster in runs of consecutive ids (for example, ids 177 to 189 are mostly Liver). Copying the group of the nearest labelled id scores 0.27 leave-one-out within SUPP-02, against 0.16 for always guessing its largest group.

A middle position is to use metadata only to restrict or re-weight an image-based prediction (the supplier and sex priors), and to leave out the case-matching and id-order shortcuts, which say nothing about the tissue.

## Recommendation

1. **C1: option 1**, a regularised linear classifier on Prov-GigaPath slide vectors. It is the largest measured gain and needs nothing outside the pack.
2. **C2: option A**, ranking labelled slides of the predicted group first, in the softer probability-weighted form.
3. **Then, if time allows, option 2** (weighted combination) aimed at the non-H&E and non-SUPP-02 slides, followed by option 6 or option 5 as an independent vote for the small groups.

Option 4 has the highest potential ceiling but the highest setup cost and depends on gated model access; it is worth starting the access request early and treating the result as a bonus. Option 8 is not realistic in the time available.

The metadata priors would add to any of these, subject to the decision above.

## How to evaluate any approach

- **Leave-one-out on the 190 labelled slides with an organ group.** There is no held-out answer key until 4:00 PM, so this is the only score available beforehand.
- **Hold out whole slides.** Any tile-level method must keep all tiles of a slide on the same side of the split.
- **Consider holding out whole SUPP-01 cases.** If SUPP-01 supplies several slides per patient, leave-one-out lets a slide match its own sibling. Grouping SUPP-01 slides by sex, age band, and organ approximates a case.
- **Report SUPP-02 H&E separately.** It is 85% of the held-out set, so it predicts the final score better than the overall figure.
- **Choose settings on one split and report on another**, or accept that a number tuned on leave-one-out is optimistic.
- **Check the submission files.** C1 answers must be a group name or a raw organ name from `organ_groups.json`. C2 results must be ids from the manifest, must not include the query itself, and are capped at 10 per query.

## How far to trust these numbers

- **The sample is small.** On 190 slides, one slide is half a percentage point, and a difference of two or three points is within noise. The gap between nearest neighbour (0.78) and ridge (0.88) is 19 slides, which is large enough to act on; the gap between centred and uncentred vectors (3 slides) is not.
- **Held-out will differ from leave-one-out.** The held-out mix is 85% SUPP-02 against 67% in the labelled set, which should help slightly, since SUPP-02 scores a little higher.
- **Possible same-case slides inflate leave-one-out** for SUPP-01, as described above.
- **Only slide vectors were tested.** Options 3 to 8 and B to C have no measurements here.

## Open questions

- **Which metadata shortcuts are acceptable?** See [The metadata question](#the-metadata-question).
- **How is C2 precision computed when fewer than 10 results are returned?** If the denominator is the number returned, a short list is the right answer for a 2-slide group like Esophagus. If it is always 10, the list should be padded with the next most likely group.
- **Is a held-out slide in "Other or mixed" answerable?** That group has no coherent appearance (0.55 leave-one-out), and it is unknown how many held-out slides fall in it.
- **Is gated model access obtainable today?** Option 4 depends on it.
- **Is a vision model API available and allowed?** Option 6 depends on it.
