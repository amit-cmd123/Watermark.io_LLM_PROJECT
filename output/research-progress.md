# Research progress snapshot

Generated: 2026-10-22T17:55:50.357899+00:00
Run: research-20261021-115335-998767

Saved progress: 161/210 cells; 806 scored rows.

**Provisional: the experiment is still in progress. This is a dated snapshot; the live dashboard continues updating.**

## Matched main-stage cases

53 topic/seed pairs have all five conditions measured for every method. The table uses only these common pairs, at the unchanged cutoff of 4.

| Method | Original detected | Rewrite 1 | Rewrite 3 | Control triggers | Originals at cap |
|---|---:|---:|---:|---:|---:|
| Previous-token, strength 2 | 53/53 (100.0%) | 2/53 (3.8%) | 0/53 (0.0%) | 0/53 (0.0%) | 9/53 |
| Previous-token, strength 3 | 53/53 (100.0%) | 5/53 (9.4%) | 0/53 (0.0%) | 0/53 (0.0%) | 23/53 |
| Fixed vocabulary, strength 3 | 53/53 (100.0%) | 2/53 (3.8%) | 1/53 (1.9%) | 0/53 (0.0%) | 29/53 |

## Interpretation

All three methods strongly mark the original passages. The measured rewrite survival remains low; increasing the original detector score has not established robust detection after rewriting.
These pooled results are exploratory, combine development/calibration/held-out seeds, and share topics. They do not establish a validated winner. Use the dashboard for the separate held-out view.
Output-cap flags show a potential writing-quality cost. They are not semantic quality ratings. No human ratings have been invented.

## What happens next

1. Finish the remaining main comparisons and the separate 128/512-token length sweep.
2. Review meaning, fluency and claims in the saved original/rewrite pairs; report detection alongside quality and incomplete output.
3. Evaluate false positives on a declared human-authored corpus; AI controls here do not establish human false-positive rates.
4. Integrate and evaluate a faithful semantic-watermark reference method with its required encoder/assets. The fixed-vocabulary variant is not a semantic method or exact paper reproduction.
5. Add translation models, independent paraphrasers, additional keys/models, and a new independent confirmation set before claiming improvement.

## Live views

[Research dashboard](http://127.0.0.1:8766/) · [Single-passage lab](http://127.0.0.1:8765/)

Protocol, limitations and paper links: RESEARCH.md. Private keys are excluded from this report.
