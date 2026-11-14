# Statistical watermarking prototype

For the full beginner-to-technical explanation of the app, workflow, scores,
completed local benchmark, research papers and presentation script, read
[Watermark Lab: complete guide](WATERMARK_LAB_COMPLETE_GUIDE.md).

The 30% implementation: keyed green-list generation, a model-independent detector,
twenty paired prompts, three recursive paraphrases, actual English–French–English
translation, CSV measurements, and plots, with a local browser dashboard.

## Interactive dashboard

For the live comparison benchmark, run `.\run.ps1 research` and open
http://127.0.0.1:8766. It shows saved passages, rewrite survival, controls,
length comparisons, checkpoints and exports. See [RESEARCH.md](RESEARCH.md)
for the protocol and the research stages still pending.

Run `.\run.ps1 ui`, then open http://127.0.0.1:8765.
No additional UI dependencies are needed. The dashboard provides paired generation,
token highlighting, independent detection, three recursive paraphrases,
English–French–English back-translation, cancellable experiment jobs, and an archive
with measurements, charts, saved passages, and CSV downloads.

The default **Generate readable paragraphs** mode displays the full decoded passages
side by side, with their paragraph breaks preserved. Watermark token highlights are
available in a separate disclosure under each passage; statistics appear below the text.
It uses a trained language model, so model files must be downloaded first.
If this session cannot download them, run `.\download_model.ps1` in a normal
PowerShell window with internet access, then generate again in the dashboard.
The script caches the default Qwen model where the running UI expects it;
`-Model` and `-Revision` select another model and version.

Choose **Numeric mechanism test** for immediate testing without downloads.
It uses actual untrained tiny GPT-2 generation and numeric token IDs; the prompt
does not condition this fixture. For meaningful prose and language attacks, select
**Real language model**. Pretrained files must be cached or downloaded, and gated
models require access. The UI cannot remove those model requirements.

Keep the same private key, model/tokenizer revision, and green fraction when
detecting a generated passage. Keys reset when the page reloads; copy your key
privately if you need to reuse it. Full experiment keys are retained under
`.runtime/ui-jobs/JOB/watermark.key`; the experiment manifest records that path.
Comparison exports exclude keys. The server listens only on loopback.

See [VALIDATION.md](VALIDATION.md) for earlier integration checks and
[RESEARCH.md](RESEARCH.md) for the current real-text benchmark. Qwen generation and
paraphrasing now run locally; the separate translation models are still pending.

For the complete implementation, experiment, environment, and milestone history,
read [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

For step-by-step explanations of HMAC, hashing, tokens, logits, z-scores, and the
experiment, read [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md).

For an in-depth account of the problem, solution, design choices, implementation,
completion status, evidence, and remaining work, read
[PROJECT_TECHNICAL_DOCUMENT.md](PROJECT_TECHNICAL_DOCUMENT.md).

## Run on Windows

```powershell
cd D:\codings\llm-watermark-prototype
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.ps1 test
.\run.ps1 experiment
```

Select a CUDA-enabled PyTorch build from https://pytorch.org/get-started/locally/
for GPU generation. The default model requires Hugging Face access to
`meta-llama/Llama-3.2-1B-Instruct`. Authenticate using the Hugging Face CLI before
the experiment if necessary. An ungated small alternative for pipeline validation:

```powershell
.\run.ps1 experiment -Model Qwen/Qwen2.5-0.5B-Instruct
```

`run.ps1` accepts `-PythonExe` to use an existing environment without changing it.
A machine-local `.runtime/python-path.txt` can record a working interpreter.

## Outputs and independent detection

Every experiment creates a timestamped results directory. It contains original
and attacked texts, raw generation token IDs, incremental `results.jsonl`,
`results.csv`, `summary.json`, `manifest.json`, `status.json`, and two PNG plots.
An interrupted run retains its scored samples and explicitly reports failure.
The default run scores 120 texts: 20 controls, 20 originals, 60 paraphrases,
and 20 back-translations. The French intermediate passages are also saved.

A random 32-byte key is saved in the experiment directory when no key is supplied.
Keep that key private; do not include it in publicly shared experiment artifacts.
For independent detection, use the model and resolved revision from the manifest:

```powershell
python detect.py results\RUN\01_original.txt --key-file results\RUN\watermark.key --model MODEL --revision COMMIT
```

The detector loads only the tokenizer. It excludes the first completion token
because its prompt context is unavailable, ignores pairs containing special IDs,
and deduplicates repeated bigrams by default. `--count-repeats` disables that filter.
Both raw-ID and decoded/re-tokenized z-scores are recorded for original outputs.

## Method and limits

For each prior token, HMAC-SHA256 derives a seed for a deterministic PCG64 vocabulary
permutation. The first floor(gamma * V) IDs form the green list; delta is added to
their logits. PCG64 is not a cryptographic PRNG, so this is a keyed research baseline,
not a formal cryptographic security construction. Scheme version, vocabulary hash,
NumPy version, model commit, seeds, and decoding settings are recorded.

Detection uses z = (G - gamma*T) / sqrt(T*gamma*(1-gamma)), with gamma equal to the
actual partition fraction. The one-sided normal p-value is approximate; token
dependence and repeated contexts limit calibration. A threshold of 4 is a declared
experimental choice, not a validated rare false-positive guarantee.

The twenty default prompts are specified topics, not C4/RealSumm samples. Twenty
controls are insufficient to establish a small false-positive rate. Manually check
attack outputs for semantic fidelity and record length changes. A short output
can have a lower z-score even at the same green ratio. Translation input exceeding
model context is rejected instead of silently truncated.

## Offline smoke test

```powershell
.\run.ps1 smoke
```

This invokes actual Transformers generation using randomly initialized tiny GPT-2
weights and measures synthetic token substitutions over twenty seeds. Its graphs
are clearly marked **OFFLINE SMOKE TEST**. It does not demonstrate language quality,
paraphrase resistance, or back-translation robustness. It needs no pretrained weights.

To check the independent detector against a saved smoke run:

```powershell
python check_detector.py results\synthetic-smoke-RUN
```

Use the same working interpreter as `run.ps1`, or pass its absolute path instead of
`python`. This creates a local fixture tokenizer and compares CLI results with the
raw-ID measurements without downloading a model.

## Sandbox dependency installation

If Python's mode-0700 temporary directories fail under the Windows fallback sandbox,
`install_dependencies.py` can install into this project's venv using inherited
permissions for temporary folders under `.runtime/tmp` only. It does not modify
system permissions. It still obeys pip's offline configuration: downloads require
an environment with network access. Ordinary external PowerShell installation is
preferred when the sandbox deliberately blocks downloads.

## References

- Kirchenbauer et al., A Watermark for Large Language Models, ICML 2023:
  https://proceedings.mlr.press/v202/kirchenbauer23a.html
- Kirchenbauer et al., On the Reliability of Watermarks for Large Language Models,
  ICLR 2024: https://arxiv.org/abs/2306.04634
- Sadasivan et al., Can AI-Generated Text be Reliably Detected?:
  https://arxiv.org/abs/2303.11156
- Kuditipudi et al., Robust Distortion-free Watermarks for Language Models:
  https://arxiv.org/abs/2307.15593
