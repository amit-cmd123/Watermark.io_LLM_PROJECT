# Next prototype handoff (Prototype 2)

This file is the starting point for any future AI model or developer continuing
Watermark Lab. Read it before changing code, and update it when the state changes.

Last verified: 9 November 2026.

## 1. Read this first

1. Read [README.md](README.md) for setup and the current results.
2. Read section 3 below. Those rules protect the credibility of the project.
3. Run the tests: `.\.venv\Scripts\python.exe -m pytest -q` (expected: 21 passed).
4. Pick the first unfinished work package in section 5.
5. When you finish something, update sections 2 and 5 here, with the run name that
   backs any new number.

## 2. Current state

### Done and verified

| Item | Evidence |
| --- | --- |
| 21 automated tests pass | `pytest`, 9 November 2026 |
| Three-method benchmark complete: 210/210 cells, 1,050 rows | `results/research-20261021-115335-998767` |
| Saved z-scores reproduce exactly (180 of 180 `baseline` passages) under newer transformers 4.57.6 and numpy 2.5.4 | Re-scored 9 November 2026 |
| Generation, paraphrase and back-translation work end to end | `results/20261023-093154-313869` (4 prompts) |
| Qwen and both Helsinki-NLP translation models are cached | `.runtime/huggingface/hub` |

### Key results

- Originals detected: 60/60 for all three methods. Controls flagged: 0/60.
- Detected after one rewrite: `baseline` 2/60, `stronger` 5/60, `fixed` 2/60.
- Held-out seed after one rewrite: 2/20, 2/20, 0/20.
- Raising delta from 2 to 3 lifted the mean original z from 9.55 to 15.55, but mean
  z after one rewrite only went from 0.73 to 1.06, and originals hit the token cap
  more often (9/60 to 25/60).

**Conclusion:** token-level green lists do not survive this rewriter. Tuning `delta`
is not a route to robustness, so Prototype 2 must change what the signal is tied to.

### Not done

- Human ratings of meaning, fluency and facts: 0 entered.
- Human-authored false-positive measurement: no corpus chosen.
- Full 20-prompt back-translation run: only a 4-prompt pilot exists.
- Semantic watermark method: not implemented.
- Independent paraphraser, more than one key, and more than one generator model.
- Forgery, key-extraction and adaptive-attack tests.

### Environment notes

- The benchmark ran on CUDA with torch 2.14.1+cu126, transformers 4.46.1 and
  numpy 2.5.3.
- The current `.venv` has **CPU-only** torch 2.14.1, transformers 4.57.6 and
  numpy 2.5.4. Install a CUDA build before starting long runs.
- `results/` is git-ignored. A fresh clone has none of the run data, so back it up
  separately, without publishing `watermark.key` files.
- Two runs started on 9 November (`research-20261109-230243-059143` and
  `research-20261109-230730-967268`) are interrupted CPU runs with almost no data.
  Their workers have exited. Do not count them as results.
- `output/research-progress.md` is a dated snapshot at 161/210 cells. It is superseded.

## 3. Rules that must not be broken

- Never invent, estimate or backfill measurements or human ratings. Empty stays empty.
- Never present offline smoke-test or synthetic-substitution results as research.
- Never describe the `fixed` method as semantic or as an exact paper reproduction.
- Never declare a winner from data used to choose it. A selected method needs a new
  independent confirmation set.
- Never claim a false-positive rate for human text without a declared human corpus.
- Report failed attacks and failed runs. Do not delete them to improve a table.
- Never commit, print or export a real `watermark.key`.
- Keep the method, prompt, seed, model revision, sampling settings and token cap
  identical across every method being compared.
- Do not break existing run folders. Bump `SCHEME_VERSION` when a scheme changes.

## 4. Prototype 2 goal

**Question:** can a watermark tied to meaning, not to exact tokens, stay detectable
after paraphrasing, while keeping false positives and writing quality acceptable?

**Hypothesis:** paraphrases change the tokens but largely keep the meaning, so a
signal derived from a semantic embedding should lose less strength than a green list
seeded by the previous token.

### Proposed acceptance criteria

The team should confirm these targets **before** seeing any Prototype 2 data.

| Measure | Target |
| --- | --- |
| Robustness | True-positive rate after one rewrite, at a fixed false-positive rate of 1%, beats `baseline` with non-overlapping 95% intervals |
| False positives | Measured on at least 500 human-written passages, because 0/300 is the minimum needed to bound the rate near 1% |
| Quality | Human meaning and fluency ratings no worse than `baseline` by more than a margin agreed in advance |
| Confirmation | Result repeats on a new set of topics and keys not used for development |

A negative result is still a valid outcome if the experiment is sound.

## 5. Work packages, in order

Each package should be its own commit or pull request.

### WP0: Fix the foundations (small)

| Change | File | Why |
| --- | --- | --- |
| Record `repetition_penalty` and the full resolved generation config in manifests, or set it explicitly | `generate.py`, `research.py`, `evaluate.py` | It is silently inherited as 1.1 from the Qwen config |
| Record that delta is applied before temperature | `RESEARCH.md`, manifests | Custom processors run before the temperature warper, so the effective boost is `delta / temperature` (2.5 and 3.75 in the study) |
| Make `.\run.ps1 test` run all tests | `run.ps1` | It runs 7 of 21 today |
| Use one default model everywhere | `run.ps1`, `evaluate.py`, `detect.py` | CLI defaults to gated Llama, while the UI and benchmark use Qwen |
| Pin versions in a lock file | `requirements.txt` | Lower bounds only today; `numpy>=1.26` is too loose for this code |

Done when the tests pass and a new manifest contains the full generation settings.

### WP1: Make green-list derivation independent of NumPy (small)

NumPy does not promise that `Generator` streams stay identical across versions (NEP
19). They matched between 2.5.3 and 2.5.4, but a future change could silently produce
z near 0 for real watermarked text. Replace the permutation with a rule defined by
the project, for example rank each token by `HMAC(key, context || token_id)` and take
the lowest `floor(gamma * V)`. This also removes the PCG64 caveat. Add a golden test
with fixed expected IDs, and bump `SCHEME_VERSION`. Keep the v1 path so old runs can
still be scored.

### WP2: Add a scheme interface (medium)

Define one interface with `process_logits`, `score` and a config dictionary, and move
`baseline` and `fixed` behind it. Today `FixedScheme` and `fixed_score` live inside
`research.py`, and `detect.py` cannot score the `fixed` method. Reformat `research.py`,
`research_server.py` and `test_research.py` to normal style at the same time, because
their dense one-line form makes them difficult to change safely.

Done when all existing tests pass and `detect.py --scheme fixed` matches saved scores.

### WP3: Get cheap evidence from existing data (small, no GPU)

- Add partial-match scoring such as WinMax (Kirchenbauer et al., ICLR 2024) and
  re-score the 1,050 saved passages.
- Fill in human ratings through the dashboard, starting with held-out seed 2.
- Report mean z against lexical overlap and length change.
- Run the full 20-prompt back-translation experiment.

### WP4: Build a stronger attack and evaluation set (medium)

- Add an independent paraphraser, for example DIPPER (Krishna et al.), so the
  generator is not attacking itself. Include a light paraphrase setting as well.
- Declare a human-written negative corpus, with its source and license.
- Use several keys and at least one more generator model.
- Move from 20 hand-written topics to a documented dataset sample.

### WP5: Implement a semantic candidate (large, the core of Prototype 2)

| Candidate | Idea | Note |
| --- | --- | --- |
| SIR (Liu et al.) | Embedding of prior text drives the logit bias | Needs an encoder and trained mapping network from the authors' release |
| SemStamp (Hou et al.) | Rejection-samples sentences into keyed semantic regions | Sentence-level; generation is slower |

Recommended start: SemStamp, because its sentence-level signal directly targets
paraphrasing and needs no project-trained mapping network. Verify this against the
paper and code first. MarkLLM can serve as a cross-check of a faithful implementation.
Check the license of any reused code or weights.

Done when the method runs behind the WP2 interface and its behaviour on a small
sample has been compared with the reference implementation.

### WP6: Run the confirmation study (large)

Pre-register the plan, compare the candidate against `baseline` and `stronger` with
the WP4 evaluation set, and report detection, false positives, quality, and cost per
token. Keep development, calibration and confirmation data separate.

### Later, only if time allows

- Non-distorting sampling, such as SynthID-Text (Dathathri et al.).
- Forgery and key-extraction tests. The `fixed` list is likely to be the easiest to
  learn from observed text, because it never changes for a given key.
- Host and Origin checks on both local servers.

## 6. References for Prototype 2

- Liu et al., [SIR: A Semantic Invariant Robust Watermark](https://arxiv.org/abs/2310.06356)
- Hou et al., [SemStamp](https://arxiv.org/abs/2310.03991)
- Krishna et al., [Paraphrasing evades detectors of AI-generated text](https://arxiv.org/abs/2303.13408) (DIPPER)
- Kirchenbauer et al., [On the Reliability of Watermarks](https://arxiv.org/abs/2306.04634) (WinMax)
- Pan et al., [MarkLLM: An Open-Source Toolkit for LLM Watermarking](https://arxiv.org/abs/2405.10051)
- Dathathri et al., [Scalable watermarking for identifying large language model outputs](https://www.nature.com/articles/s41586-024-08025-4), Nature 2024
- [NEP 19: RNG policy](https://numpy.org/neps/nep-0019-rng-policy.html)

## 7. Change log

| Date | Change |
| --- | --- |
| 9 November 2026 | File created from the verified state of the code and saved runs |
