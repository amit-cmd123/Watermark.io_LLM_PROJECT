# Watermark Lab: progress report

**Date:** 14 November 2026  
**Scope:** everything done after the first working prototype (the green-list watermark, its detector and its first experiments, described in `PROJECT_TECHNICAL_DOCUMENT.md`).  
**Rule used throughout:** a claim is only stated as fact if it was measured in this project. Everything else is marked as a hypothesis or as not done.

## 1. Short version

- **The central finding is negative for token-level schemes.** On real text, the original prototype detects its own watermark 60/60 times but loses it almost completely after one LLM paraphrase (2 to 5 of 60 still detected). A better detector does not fix this. The words the detector needs are mostly gone after that rewrite.
- **The evaluation attack was unfaithful.** The paraphrase keeps 32% of the original's content words, the same share as an unrelated answer to the same prompt (31%). It is close to rewriting from scratch, so results against it say little about faithful paraphrase.
- **Built since the first prototype:** a faster hash engine, min-hash keying, a semantic (G4) watermark, three benchmark harnesses, a detector lab, a rebuilt dashboard, and tests for all of it.
- **Not established:** that any new scheme survives rewriting better than the original. The full benchmark that would show this was killed at 190 of 200 generations (memory) and lost. No end-to-end speed or accuracy result exists yet for the new schemes.

## 2. What was built

| Piece | File | Status |
|---|---|---|
| Generation-2 engine: stateless 32-bit hash partitions, GPU mask, context window h, span-scan detector, empirical-null helper | `engine.py`, `test_engine.py` | Built. 8 tests pass. Not yet run on real text. |
| Min-hash keying (key comes from the context token with the smallest keyed hash, so an edit breaks the key only if it hits that token) | `engine.py` (`agg="min"`) | Built, tested for numpy/torch agreement and edit behaviour. Real benefit not yet measured. |
| G4 semantic watermark: sentence embedding, key-derived hyperplanes, margin-aware rejection sampling, centring on reference text, z-test on valid sentences | `semantic.py`, `test_semantic.py` | Built. 5 tests pass. Only a 2-text smoke run on a real model. |
| Head-to-head benchmark (speed, perplexity, detection under edits and sentence-wise translation, AUC, false alarms) | `bench_eval.py` | Built. Now checkpoints each finished arm and can resume (`--resume`). Checkpointing is compiled but not run end to end. |
| G4 evaluation vs token schemes under translation, per-sentence paraphrase and whole-text paraphrase, with rewrite fidelity | `bench_semantic.py` | Built. 2-text smoke run only. |
| Clean speed benchmark (interleaved rounds, medians) | `bench_speed.py` | Written. Never run. |
| Detector lab: re-score saved texts with different detectors, no regeneration | `detector_lab.py` | Built and run on real saved texts (section 3.4). |
| Dashboard rebuild: scheme comparison with strip plot, highlighted-text results, scheme selector, preview banner | `ui/`, `ui_server.py`, `ui_worker.py` | Built. See section 5 for what was and was not verified. |
| Processor microbenchmark | `bench.py` | Run. See section 3.1. |
| Roadmap and literature notes | `ROADMAP.md` | Written. Superseded in places by this report. |

## 3. What was measured

### 3.1 Processor speed (microbenchmark only)
Cost of the logits step alone at a 128,256-token vocabulary, worst case for v1 (every context new), RTX 4070:

| Scheme | ms per step |
|---|---:|
| v1 permutation | 4.34 |
| v2 hash, context 1 | 1.47 |
| v2 hash, context 4 | 1.81 |

This is not end-to-end generation speed. An end-to-end speed result does not exist yet (section 6).

### 3.2 The original scheme on real text
Source: `results/research-20261021-115335-998767` (Qwen2.5-0.5B-Instruct, 20 prompts, self-paraphrase, 60 texts per cell).

| Scheme | Original detected | After paraphrase 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| Previous-token, strength 2 | 60/60 | 2/60 | 0/60 | 0/60 |
| Previous-token, strength 3 | 60/60 | 5/60 | 0/60 | 0/60 |
| Fixed vocabulary, strength 3 | 60/60 | 2/60 | 0/60 | 1/60 |

Controls (unwatermarked model text): 0/60 detected in each variant.

### 3.3 Why: the attack removes the words
| Measure (original vs paraphrase 1) | Paraphrase | Unrelated answer, same prompt |
|---|---:|---:|
| Content words of the original kept | 0.32 | 0.31 |
| Bigrams kept | 0.13 | 0.10 |
| Whole-text embedding cosine | 0.89 | 0.83 |
| Mean best-sentence cosine | 0.75 | 0.67 |

The 1-prompt pipeline run (`results/20261114-124027-712120`) also found a silent failure: Marian back-translation of a 270-word text returned 124 words with the second half dropped, yet the run recorded it as a normal result (z = 6.6). The pipeline has no length or fidelity gate.

### 3.4 Detector-side fixes on real paraphrases (detector lab)
Same saved texts, controls as negatives (60), paraphrase 1, 60 texts per cell. "TPR at 2% FPR" means the share detected when the threshold is set so at most 2% of controls pass.

| Variant | Detector | AUC | TPR at 2% FPR |
|---|---|---:|---:|
| Strength 2 | standard (current) | 0.65 | 0.15 |
| | content words only | 0.63 | 0.10 |
| | best 64-token window | 0.57 | 0.12 |
| | content words + window | 0.56 | 0.13 |
| Strength 3 | standard | 0.66 | 0.20 |
| | content words only | 0.67 | 0.20 |
| | best 64-token window | 0.61 | 0.25 |
| | content words + window | 0.65 | 0.20 |

Originals scored AUC 1.00 and TPR 1.00 for every detector.

What this shows:
- **A fixed false-alarm threshold helps,** lifting detection after paraphrase 1 from 3% (z ≥ 4) to about 15%. It is still low.
- **Content-word scoring and window scanning did not help here.** The differences from the standard detector are within the noise at 60 texts.
- **The detector cannot recover what the rewrite removed.**

Caveats: negatives are AI text, not human writing. The recomputed z-scores differ from the stored ones by up to 0.58 on originals (typical z about 10 to 15). The likely cause is that the stored scores used re-tokenized text and mine used raw token IDs, but this was not verified. The conclusion does not depend on it.

### 3.5 G4 in simulation
With synthetic embeddings, margin-aware selection kept valid sentences valid under drift better than choosing any valid candidate:

| Drift (noise norm) | Margin-picked kept | Random valid kept |
|---:|---:|---:|
| 0.3 | 1.00 | 0.86 |
| 0.5 | 0.98 | 0.78 |
| 0.8 | 0.85 | 0.61 |

This is synthetic. Real sentence embeddings are less uniform, so it needs a real test.

### 3.6 Smoke runs (plumbing only, 2 texts, not evidence)
Generation speed in the smoke benchmark: no watermark 27.4 tokens/s, v1 22.8, v2 hash context 1 27.0, v2 context 4 25.6, v2 min-hash 26.1. G4 semantic ran at 7.6 tokens/s against 28.9 for plain generation in its own smoke run (about 3.8 times slower, as expected from sampling several candidates per sentence). With 2 texts, none of these numbers support a conclusion.

## 4. Things that went wrong
- **The full benchmark was lost.** It reached 190 of 200 generations and was stopped by the harness for low system memory. It only wrote its results at the end, so nothing was saved. Roughly 35 minutes of GPU work. Fix applied: per-arm checkpoints and `--resume`. Not yet run.
- **The speed numbers from that run were invalid.** The no-watermark arm measured 26.4 tokens/s while v1 measured 30.9 and v2 context 1 measured 38.6, which is impossible because the watermark only adds work. Other work (tests, screenshots) was running on the same machine, and generation is CPU-bound. `bench_speed.py` was written to measure speed alone with interleaved rounds. Not yet run.
- **The dashboard server was stopped for low memory twice.** Free memory was about 2.6 GB of 15.7 GB, mostly used by the browsers (Edge WebView and Zen).
- **Environment traps found:** a global `HF_HOME` points to another project's cache (overrode with a forced value in the scripts); `run.ps1` uses `ErrorActionPreference='Stop'`, so piping it with `2>&1` turns a harmless sacremoses warning into a fatal error; the `!` shell is bash and drops backslashes, so `.\run.ps1` fails there; Hugging Face downloads often get connection-reset (Qwen, both translation models and the embedding model are cached locally).
- **Mistakes of mine, corrected:** I first said no real run existed (one did, I had read an out-of-date document); I claimed a 3x speed gain from a processor microbenchmark that said nothing about generation speed; a unit test used unrealistically large drift and failed for the wrong reason (fixed by setting realistic drift, assertion unchanged); a UI bug printed "+-2%" (fixed).

## 5. Dashboard: what was and wasn't verified
- **Verified:** the UI tests pass (3 of 3); the JavaScript passes syntax checks; the new screens were rendered in headless Chrome against a stubbed API fed with real saved data (smoke benchmark summaries and a real worker output), with no page errors. The sign-formatting bug was checked from the DOM text.
- **Not verified after the redesign:** the live server path (starting jobs, polling, cancelling). It worked with the earlier version of the UI.
- **Screens:** Compare (verdict chips, scoreboard, strip plot with tabs, rewrite fidelity, text viewer, limits), Generate & compare, Check text, Rewrite test, Runs. A preview banner appears when a benchmark has fewer than 20 texts.

## 6. What is not established
- That min-hash keying, a longer context, or G4 survives rewriting better than the original scheme.
- Real generation speed for any new scheme (only the microbenchmark and 2-text smoke runs exist).
- Text quality (perplexity) for any new scheme beyond 2-text smoke values.
- A false-positive rate on human-written text. All negatives so far are unwatermarked model text.
- Resistance to an adaptive attacker, and any security claim.
- That checkpointing and `--resume` work end to end (they compile; they have not run).

## 7. Running things

PowerShell, from `D:\codings\llm-watermark-prototype` (use the project interpreter recorded in `.runtime/python-path.txt`):

```powershell
.\run.ps1 ui                         # dashboard at http://127.0.0.1:8765
.\run.ps1 test                       # v1 tests
python -m unittest test_ui test_engine test_watermark test_research test_model_loading test_semantic
python detector_lab.py               # CPU only, about 4 minutes
python bench_speed.py                # run with nothing else running
python bench_eval.py --limit 20 --seeds 2 --tokens 200    # about 35 min; add --resume <folder> after an interruption
python bench_semantic.py --limit 15 --seeds 2 --tokens 250  # about 45 min
```

Set `HF_HUB_OFFLINE=1` to avoid download retries. Do not run benchmarks while low on memory. `npm run dev` does not apply: there is no `package.json` and no Node build step.

Test status at the last full run: 29 tests passed across the ui, engine, watermark, research and model_loading suites; `test_semantic` (5) passed separately; `test_ui` (3) passed after the redesign. The suites were not all rerun together after the final benchmark and UI edits.

## 8. Next steps, in order
1. **Free memory, then run the full benchmark** (with checkpointing) and `bench_speed.py` alone. This gives the first real speed, quality and detection numbers for the new schemes.
2. **Add the fixed-false-alarm metric to the dashboard.** The benchmark now records `tpr_fpr5` and `tpr_fpr1`, but the screen does not show them yet.
3. **Build a human-text set (200+ documents)** and calibrate thresholds on it, so false-alarm rates are real.
4. **Run the G4 evaluation** and read it against rewrite fidelity. G4 can only help if the rewrite keeps sentence meaning.
5. **Add a fidelity gate** and a sentence-by-sentence paraphraser to the main experiment, and report detection against rewrite fidelity.
6. **Fuse the token and semantic scores** only if step 4 shows G4 adds anything.
7. **Test against an adaptive attacker** before any robustness claim.

## 9. File map
- Core: `watermark.py` (v1), `detect.py`, `engine.py` (v2), `semantic.py` (G4), `generate.py`, `attacks.py`, `model_loading.py`
- Experiments: `evaluate.py`, `research.py`, `bench_eval.py`, `bench_semantic.py`, `bench_speed.py`, `bench.py`, `detector_lab.py`
- Dashboard: `ui_server.py`, `ui_worker.py`, `ui/index.html`, `ui/app.js`, `ui/style.css` (the earlier UI is kept in `ui/legacy/`)
- Docs: `PROJECT_TECHNICAL_DOCUMENT.md` (first prototype), `ROADMAP.md`, `VALIDATION.md`, this file
- Results: `results/research-20261021-115335-998767` (real), `results/detector-lab-*`, `results/benchmark-*` (smoke runs, partial or empty), `results/20261114-124027-712120` (1-prompt run)

## 10. Sources consulted
Only abstracts or search summaries were read for most papers. Check details before implementing from them. One paper (arXiv 2607.16010) could not be fetched, so its claim of 100% removal of KGW and Unigram watermarks after paraphrase is unconfirmed.

- Rastogi and Pruthi, [Revisiting the Robustness of Watermarking to Paraphrasing Attacks](https://arxiv.org/html/2411.05277v1) (read in full text via a fetch)
- [k-SemStamp](https://arxiv.org/pdf/2402.11399) and [SemStamp](https://arxiv.org/pdf/2310.03991)
- [Semantic Invariant Robust Watermark](https://arxiv.org/pdf/2310.06356)
- [Robust Text Watermarking via Dual Semantic Embeddings](https://arxiv.org/abs/2606.31602)
- [Can Watermarks Survive Translation?](https://arxiv.org/pdf/2402.14007) and [Cross-Lingual Summarization as a Watermark Removal Attack](https://arxiv.org/pdf/2510.24789)
- [Scalable watermarking (SynthID-Text), Nature 2024](https://ideas.repec.org/a/nat/nature/v634y2024i8035d10.1038_s41586-024-08025-4.html)
- [Publicly-Detectable Watermarking for Language Models](https://arxiv.org/pdf/2310.18491)
- [Unified Attacks: Spoofing and Scrubbing (CDG-KD)](https://arxiv.org/abs/2504.17480v2) and [DITTO](https://arxiv.org/html/2510.10987v3)
- [WaterPark](https://arxiv.org/html/2411.13425v1)
- [A Reinforcement Learning Framework for Robust LLM Watermarking](https://arxiv.org/pdf/2510.21053) and [Semantic Behavioral Watermarking](https://arxiv.org/abs/2610.08668v1)
- [AI Watermark Evidence Fails Forensic Readiness](https://arxiv.org/html/2607.16010v1) (not read)
