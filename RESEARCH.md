# Research bench

Open http://127.0.0.1:8766. The original lab links to this page.
To restart its server, run `.\run.ps1 research`. The UI can resume a saved run.
The benchmark runs as a separate process, so closing a browser tab does not stop it.

## Active experiment

The initial run is `research-20261021-115335-998767` under `results/`.
Its exact state, measured counts and estimated remaining time are on the dashboard.

The main study uses 20 topics, three seed indices, and three variants:

- Previous-token green list, gamma 0.25 and delta 2 (current baseline).
- Previous-token green list, gamma 0.25 and delta 3 (stronger bias).
- Fixed-vocabulary green list, gamma 0.25 and delta 3, with distinct-token scoring.

Each cell scores its ordinary control, marked original and three recursive rewrites.
That is 180 main cells and 900 scored rows. Controls are generated once per
(topic, seed, length) and scored separately for each method. They are not independent
extra negative samples just because several rows refer to the same control.

The separate length sweep uses five topics, seed index 0, token caps 128 and 512,
and all three variants. It adds 30 cells and 150 rows. Total: 210 cells / 1,050 scored
rows when complete. A row is saved as soon as that passage finishes. The run may
contain a partially completed cell in addition to its completed-cell count.

## Fairness and statistical limits

Every method receives the same prompt, random seed, model revision, sampling
settings, cutoff and token cap. Detection primarily uses re-tokenized displayed
text. Complete decoded text and raw generation IDs are retained. The fixed method
counts distinct eligible token IDs, while previous-token detection counts distinct
eligible token pairs. Neither normal score is a guaranteed calibrated null test.

The fixed method is an experimental variant inspired by Zhao et al.,
[Provable Robust Watermarking for AI-Generated Text](https://proceedings.iclr.cc/paper_files/paper/2024/hash/beae9ed5316bcc48e616754c06c11875-Abstract-Conference.html).
It changes the partitioning strategy and uses unique-token scoring. It is not an
exact reproduction, a new semantic method, or an implementation of the paper's proofs.

The primary exploratory view keeps the cutoff at 4. The separate held-out view uses
seed index 1 controls to select an exploratory cutoff of max(4, max control score +
0.000001), then evaluates seed index 2. Seed index 0 is development. Twenty calibration
controls per method are insufficient for a rare false-positive claim. All methods
use the same control texts, but method-specific scores. Selecting a winner using
these held-out results requires a new independent confirmation set.

Displayed Wilson intervals are descriptive and do not model the correlation caused
by repeating topics. Controls are unwatermarked AI prose. A human false-positive
rate requires a separate, declared human-authored negative corpus. None is invented.

## Quality and honest outcomes

The UI shows original and rewritten passages side by side. Reaching the generation
cap is flagged, including cases that end with punctuation. Lexical overlap is a
descriptive statistic, not a semantic-similarity claim. Human ratings of preserved
meaning, fluency and claims remain empty until supplied through the review form.
An improved original z-score alone never establishes an improved watermark.

The first measured rewrites failed detection for all variants. These failures remain
in the results. The dashboard has no automatic validated-winner badge.

## Pending research stages

Back-translation requires cached `Helsinki-NLP/opus-mt-en-fr` and
`Helsinki-NLP/opus-mt-fr-en` models. Only the Qwen generation model is currently
cached. Translation is explicitly not run in this study; random substitutions are
not a stand-in.

A semantic candidate requires a dedicated encoder, its appropriate trained/reference
assets, faithful integration and a controlled comparison. The reference is
[Liu et al., A Semantic Invariant Robust Watermark for Large Language Models](https://arxiv.org/abs/2310.06356).
The current fixed-vocabulary candidate must not be described as semantic watermarking.

Human false-positive calibration, independent paraphrasers, additional keys and
models, and independent confirmation are subsequent evaluations. Their status is
visible rather than counted as completed.

## Persistence and exports

Each run stores `manifest.json`, `status.json`, `completed.json`,
`measurements.jsonl`, `results.csv`, `samples/*.json`, and optional `reviews.json`.
The secret key lives in `watermark.key` and is excluded from HTTP exports.
The full report includes actual saved passages, metrics, settings and reviews.

Stop requests take effect between generations and preserve intermediate rows. Resume
loads saved controls and passages instead of regenerating or duplicating them.
Checkpoints survive server restarts. A failed process records its error in status;
an exited worker is shown as stopped and resumable even if its final status write
failed. Atomic JSON and measurement writes retry transient Windows reader locks.
Tests cover paired plans, calibration separation, deduplication, resume idempotency,
API path boundaries, authorization and review validation.
