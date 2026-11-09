# Validation — 8 October 2026

The prototype files were created and executed in this workspace after terminal
access recovered. The actual paraphrase/back-translation experiment is not complete.

## Completed checks

- Seven unit tests passed, with no skipped tests, using the recorded CUDA-enabled
  interpreter. Tests cover deterministic keyed partitions, key sensitivity, invalid
  parameters, ineligible-token boundaries, repeated-bigram filtering, a known
  z-score, batched logit modification, and synthetic detectability.
- Actual Transformers generation with a randomly initialized tiny GPT-2 completed
  on CUDA across twenty seeds. One hundred sequences/perturbations were scored.
- The independent `detect.py` CLI loaded a local fixture tokenizer without language
  model weights. Its green counts and z-scores exactly matched saved raw-token
  measurements for a control and watermarked sample.
- The generated PNG was visually inspected and clearly identifies its synthetic scope.

## Synthetic smoke results

These are implementation checks, not language-quality or paraphrase-resistance results.

| Condition | Mean z-score | Detections at z >= 4 |
| --- | ---: | ---: |
| Unwatermarked control | -0.324 | 0/20 |
| Watermarked | 14.859 | 20/20 |
| Synthetic replacement of 10% of tokens | 11.861 | 20/20 |
| Synthetic replacement of 30% of tokens | 7.075 | 20/20 |
| Synthetic replacement of 60% of tokens | 2.421 | 2/20 |

Artifacts: `results/synthetic-smoke-20261008-085721/` contains the CSV, summary,
token IDs, graph, local fixture tokenizer, and standalone detector comparison.
The fixture key is deliberately public and only for synthetic validation.

## Remaining experiment

A one-prompt pretrained-model attempt used `Qwen/Qwen2.5-0.5B-Instruct`, an ungated
small model for pipeline validation. It failed before generating any text because
Hugging Face requests were routed through an unavailable proxy at `127.0.0.1:9`.
Pip's offline setting also prevented new dependency downloads. Existing workspace
dependencies were reused without modifying their environment.

The failed attempt is explicitly recorded in
`results/20261008-085541-448449/status.json` with zero scored texts.
No pretrained-model completion, semantic paraphrase, or translated attack result
has been produced. No rare false-positive rate has been established.

When network access and the remaining translation dependencies are available:

```powershell
.\run.ps1 experiment -Model Qwen/Qwen2.5-0.5B-Instruct
```

For the original project model after Hugging Face access is configured:

```powershell
.\run.ps1 experiment -Model meta-llama/Llama-3.2-1B-Instruct
```

Both commands default to twenty prompts. The real run will produce 120 scored texts
and separate plots. Synthetic graphs must not be presented as those research results.
