# Watermark Lab: LLM text watermarking prototype

Watermark Lab hides a keyed statistical signal in text while a language model
writes it, then checks for that signal later using only the key and the tokenizer.
It also measures how well the signal survives rewriting and translation.

**Main finding so far:** the watermark is detected reliably in untouched text, but
a single paraphrase almost always removes it. Making the watermark stronger did not
fix this. See [Results](#results-so-far) and [Prototype 2 plan](NEXT_PROTOTYPE.md).

This is a research prototype. It is not a production AI-text detector, and its
results must not be used to accuse anyone of using AI.

## What is included

| Part | What it does | Start it with |
| --- | --- | --- |
| Watermark Lab | Generate a control and a watermarked passage side by side, check text, try attacks, run batch jobs | `.\run.ps1 ui`, then open http://127.0.0.1:8765 |
| Research dashboard | Run or resume the three-method benchmark, read saved passages, enter human ratings, export results | `.\run.ps1 research`, then open http://127.0.0.1:8766 |
| Batch experiment | Control, watermarked, three paraphrases and back-translation for each prompt | `.\run.ps1 experiment -Model Qwen/Qwen2.5-0.5B-Instruct` |
| Standalone detector | Score a text file without loading any model weights | `python detect.py` (see [Detect a saved passage](#detect-a-saved-passage)) |
| Offline smoke test | Exercise generation and detection on an untrained tiny model, with no downloads | `.\run.ps1 smoke` |

Both dashboards listen on loopback only and need no extra UI dependencies.

## Quick start (Windows PowerShell)

Requires Python 3.13 (the tested version) and enough disk space for the models.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\download_model.ps1
.\run.ps1 ui
```

- `.\download_model.ps1` caches `Qwen/Qwen2.5-0.5B-Instruct` in `.runtime\huggingface`.
  Use `-Model` and `-Revision` to fetch another model. The two Helsinki-NLP translation
  models are downloaded the first time back-translation runs.
- A plain `pip install` on Windows gives CPU-only PyTorch, which is slow for full
  benchmarks. For a GPU, install a CUDA build from
  <https://pytorch.org/get-started/locally/>.
- `run.ps1` picks its interpreter from `-PythonExe`, then `.runtime\python-path.txt`,
  then `.venv`, then `python` on PATH.
- `.\run.ps1 experiment` without `-Model` uses the gated
  `meta-llama/Llama-3.2-1B-Instruct`, which requires Hugging Face access. Pass the
  Qwen model as shown above unless you have that access.
- `.\run.ps1 test` runs only the 7 core watermark tests. Use the `pytest` command
  above to run all 21.

## How it works

1. **Partition.** For each previous token, HMAC-SHA256 with the private key seeds a
   PCG64 permutation of the vocabulary. The first `floor(gamma * V)` token IDs form
   the green list.
2. **Bias.** During generation, `delta` is added to the logits of green tokens, so
   the model leans toward them without being forced.
3. **Detect.** The detector re-tokenizes the text, counts green tokens and computes:

   ```text
   z = (G - gamma * T) / sqrt(T * gamma * (1 - gamma))
   ```

   `G` is the green count and `T` the number of scored tokens. Text with `z >= 4` is
   flagged. The first token, pairs containing special IDs and repeated bigrams are
   not counted unless you pass `--count-repeats`.

Detection needs the same key, tokenizer and revision, and `gamma` used at generation.

### Methods compared

| ID | Green list depends on | gamma | delta | Scored unit |
| --- | --- | ---: | ---: | --- |
| `baseline` | Previous token | 0.25 | 2 | Distinct token pairs |
| `stronger` | Previous token | 0.25 | 3 | Distinct token pairs |
| `fixed` | Nothing (one list for the whole key) | 0.25 | 3 | Distinct token IDs |

`fixed` is inspired by Zhao et al. It is not an exact reproduction and it is not a
semantic watermark. `detect.py` currently supports the previous-token scheme only.

## Results so far

All numbers come from local runs on `Qwen/Qwen2.5-0.5B-Instruct` with CUDA.
The `results/` folder is git-ignored, so these runs exist only on the machine that
produced them.

### Three-method benchmark

Run `research-20261021-115335-998767`, complete: 210 of 210 cells and 1,050 scored
passages. The main stage below has 20 topics and 3 seeds per method, at cutoff 4.

| Method | Control flagged | Original detected | Rewrite 1 | Rewrite 2 | Rewrite 3 | Mean z, original to rewrite 1 | Originals hitting token cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0/60 | 60/60 | 2/60 | 0/60 | 0/60 | 9.55 to 0.73 | 9/60 |
| `stronger` | 0/60 | 60/60 | 5/60 | 0/60 | 0/60 | 15.55 to 1.06 | 25/60 |
| `fixed` | 0/60 | 60/60 | 2/60 | 0/60 | 1/60 | 12.24 to 0.94 | 35/60 |

Ordinary unwatermarked controls hit the cap 11 times out of 60, so the stronger
variants appear to run longer, which may be a writing-quality cost.

In the separate length sweep (128 and 512 tokens, 5 topics), every original was
detected and none of the 90 rewrites was.

### Back-translation pilot

Run `20261023-093154-313869`: 4 prompts, delta 2. English to French to English kept
2 of 4 passages detectable (mean z 3.76, from 8.86), while all 12 paraphrases
failed. Four prompts are too few to conclude anything, and the full 20-prompt
experiment has not been run.

### How to read these numbers

- Rewrites are from the same small Qwen model. They keep about 85% of the original
  length and share about 20% of its vocabulary, so this is a heavy rewrite. Meaning
  preservation has not been checked: **no human ratings have been entered yet.**
- 0/60 flagged controls is compatible with a true false-positive rate up to about
  6% (95% Wilson interval). Controls are AI text, so the human false-positive rate
  is **unmeasured**.
- Topics repeat across seeds, one key was used, and the z-scores for `fixed` use a
  different unit from the other two. No method has been validated as a winner.

The full protocol is in [RESEARCH.md](RESEARCH.md).

## Detect a saved passage

Take the model and resolved commit from the run's `manifest.json`.

```powershell
python detect.py results\RUN\01_original.txt --key-file results\RUN\watermark.key --model Qwen/Qwen2.5-0.5B-Instruct --revision COMMIT
```

Always pass `--model`, because the detector's default is the gated Llama model. Also
pass `--gamma` if the run did not use 0.25.

## Output files

| Location | Contents | In git? |
| --- | --- | --- |
| `results/<timestamp>/` | Batch run: texts, token IDs, `results.csv`, `results.jsonl`, `summary.json`, `manifest.json`, `status.json`, two PNG plots | No |
| `results/research-<timestamp>/` | Benchmark: `measurements.jsonl`, `results.csv`, `samples/*.json`, `completed.json`, optional `reviews.json` | No |
| `results/synthetic-smoke-<timestamp>/` | Offline smoke test, not research evidence | No |
| `.runtime/` | Model cache, UI jobs, local interpreter path | No |
| `output/` | Presentations, screenshots and a dated research snapshot | Yes |

Runs save after every passage, and benchmark runs can be stopped and resumed
without regenerating saved text.

**Keep `watermark.key` private.** Anyone holding it can detect, and potentially
forge, the watermark. Keys are git-ignored and excluded from HTTP exports. Keys typed
into Watermark Lab reset on page reload, so copy yours if you need it later.

## Project layout

| File | Role |
| --- | --- |
| `watermark.py` | Keyed green lists and the generation logits processor |
| `detect.py` | Detector and command-line interface |
| `generate.py` | Model loading and generation |
| `attacks.py` | Paraphrase prompt and Marian back-translation |
| `evaluate.py`, `plot_results.py` | Batch experiment and plots |
| `research.py`, `research_server.py` | Resumable benchmark and its dashboard server |
| `ui_server.py`, `ui_worker.py`, `ui/` | Watermark Lab server, isolated job worker and front end |
| `smoke.py`, `check_detector.py` | Offline integration checks |
| `test_*.py` | Unit, statistics, API and resume tests |
| `install_dependencies.py` | Install workaround for restricted Windows sandboxes only |

## Known limits and planned changes

- **Paraphrasing removes the signal.** This is the main problem for Prototype 2.
- **Not calibrated.** The cutoff of 4 is a declared choice, and p-values are
  approximate because tokens are not independent.
- **Not cryptographic.** PCG64 is not a cryptographic generator. Forgery and key
  extraction have not been tested.
- **Small sample.** One small model, one key per run and 20 hand-written topics,
  which are not C4 or RealSumm samples.
- **Not yet recorded.** Run manifests omit `repetition_penalty`, which is inherited
  as 1.1 from the Qwen generation config.

The ordered work list, acceptance criteria and handoff rules are in
[NEXT_PROTOTYPE.md](NEXT_PROTOTYPE.md).

## Further reading

| Document | Use it for |
| --- | --- |
| [NEXT_PROTOTYPE.md](NEXT_PROTOTYPE.md) | Current state and the plan for the next prototype |
| [RESEARCH.md](RESEARCH.md) | Benchmark protocol and statistical limits |
| [WATERMARK_LAB_COMPLETE_GUIDE.md](WATERMARK_LAB_COMPLETE_GUIDE.md) | Full explanation of the app, scores and presentation |
| [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md) | Beginner walkthrough of HMAC, tokens, logits and z-scores |
| [PROJECT_TECHNICAL_DOCUMENT.md](PROJECT_TECHNICAL_DOCUMENT.md) | Design rationale for the first milestone |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Implementation and environment history |
| [VALIDATION.md](VALIDATION.md) | Integration checks from 8 October 2026 |

`RESEARCH.md`, `VALIDATION.md` and the first-milestone documents were written while
work was in progress, and some of their status notes are out of date. Where they
disagree with this README, trust this README and the saved run data.

## References

- Kirchenbauer et al., [A Watermark for Large Language Models](https://proceedings.mlr.press/v202/kirchenbauer23a.html), ICML 2023
- Kirchenbauer et al., [On the Reliability of Watermarks for Large Language Models](https://arxiv.org/abs/2306.04634), ICLR 2024
- Zhao et al., [Provable Robust Watermarking for AI-Generated Text](https://proceedings.iclr.cc/paper_files/paper/2024/hash/beae9ed5316bcc48e616754c06c11875-Abstract-Conference.html), ICLR 2024
- Liu et al., [A Semantic Invariant Robust Watermark for Large Language Models](https://arxiv.org/abs/2310.06356)
- Sadasivan et al., [Can AI-Generated Text be Reliably Detected?](https://arxiv.org/abs/2303.11156)
- Kuditipudi et al., [Robust Distortion-free Watermarks for Language Models](https://arxiv.org/abs/2307.15593)
