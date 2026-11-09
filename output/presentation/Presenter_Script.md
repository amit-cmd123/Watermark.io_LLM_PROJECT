# Presentation script

Statistical Watermarking for LLM Text

## 1. Statistical watermarking for LLM text

Our project asks how long a deliberate watermark remains detectable when someone rewrites generated text. We have a working baseline, readable generation, independent detection, and an interactive rewriting test. I will distinguish implemented features from measured evidence, explain the observed paraphrasing failure, and outline the evaluation needed before making robustness claims.

Sources: 

## 2. The problem statement

AI-generated text can be copied and rewritten before it is shared. Our specific research problem is whether a signal intentionally embedded during generation remains detectable after these transformations. We assume access to the generator at insertion time and possession of the correct secret key at detection time. We do not assume that every model adds our watermark, or that a failed detection proves human authorship.

Sources: R1: Kirchenbauer et al. (ICML 2023). A Watermark for Large Language Models. https://proceedings.mlr.press/v202/kirchenbauer23a.html
R3: Sadasivan et al. (TMLR, revised 2025). Can AI-Generated Text be Reliably Detected?. https://arxiv.org/abs/2303.11156

## 3. Why this baseline?

We chose a keyed green-list method because it works through sampling without retraining. It provides a shared insertion and detection mechanism with a clear strength-versus-quality tradeoff. The detector needs the tokenizer and key but does not run the generation model. These properties make it a useful research baseline. They do not guarantee survival under substantial rewriting.

Sources: R1: Kirchenbauer et al. (ICML 2023). A Watermark for Large Language Models. https://proceedings.mlr.press/v202/kirchenbauer23a.html

## 4. How our watermark works

A token is a small unit of text, often part of a word. At each generation position, the secret key and previous token determine a vocabulary partition. About a quarter of eligible choices enter the green list. A positive bias promotes those choices without forcing every token to be green. The detector rebuilds the same lists and counts favored choices. Our HMAC-SHA256 and PCG64 implementation is a research baseline, not a formal cryptographic security construction.

Sources: R1: Kirchenbauer et al. (ICML 2023). A Watermark for Large Language Models. https://proceedings.mlr.press/v202/kirchenbauer23a.html
watermark.py, generate.py, detect.py. Default gamma=0.25, delta=2, previous-token context.

## 5. What the signal score means

G is the count of green choices among T scored token pairs. Gamma is the actual favored vocabulary fraction. Zero is around the null expectation. Larger positive scores give stronger evidence of this pattern. We choose four as an experimental cutoff. We exclude the first completion token, special tokens, and repeated bigrams. Dependence limits calibration of the approximate normal p-value. The score is not a probability that AI wrote the passage.

Sources: detect.py detect_ids and watermark.py actual partition fraction.

## 6. Research foundations

Kirchenbauer and colleagues introduced soft promotion of green tokens and statistical detection. Their follow-up found that residual n-grams can support detection after rewriting with enough observed text. Sadasivan and colleagues stress-test detection using recursive paraphrasing and spoofing. These conditional findings are not a universal contradiction. Methods, attacks, lengths and operating points differ.

Sources: R1: Kirchenbauer et al. (ICML 2023). A Watermark for Large Language Models. https://proceedings.mlr.press/v202/kirchenbauer23a.html
R2: Kirchenbauer et al. (ICLR 2024). On the Reliability of Watermarks for Large Language Models. https://proceedings.iclr.cc/paper_files/paper/2024/hash/d78e9e4316e1714fbb0f20be66f8044c-Abstract-Conference.html
R3: Sadasivan et al. (TMLR, revised 2025). Can AI-Generated Text be Reliably Detected?. https://arxiv.org/abs/2303.11156

## 7. Research directions beyond the baseline

Kuditipudi and colleagues couple generation to keyed random sequences and align text during detection. Liu and colleagues derive watermark behavior from semantic representations. SynthID-Text studies production feasibility and efficiency. WaterBench motivates jointly evaluating detection and generation quality across tasks. These are future comparisons, not implemented or benchmarked alternatives in our prototype.

Sources: R4: Kuditipudi et al. (TMLR 2024). Robust Distortion-free Watermarks for Language Models. https://arxiv.org/abs/2307.15593
R5: Liu et al. (ICLR 2024). A Semantic Invariant Robust Watermark for Large Language Models. https://arxiv.org/abs/2310.06356
R6: Dathathri et al. (Nature 2024). Scalable watermarking for identifying large language model outputs. https://doi.org/10.1038/s41586-024-08025-4
R7: Tu et al. (ACL 2024). WaterBench: Towards Holistic Evaluation of Watermarks for Large Language Models. https://aclanthology.org/2024.acl-long.83/

## 8. What we have built

The prototype generates a paired control and watermarked passage using Qwen2.5-0.5B-Instruct. It includes independent detection, recursive paraphrasing, and a local interface with full paragraphs and optional highlights. The batch experiment runner and French back-translation path also exist, but implementation alone does not mean the full study is complete. The observed natural-language evidence remains a local example.

Sources: watermark.py, detect.py, generate.py, ui_worker.py, attacks.py, evaluate.py, ui/. Current saved real outputs: b0221f45c23f7596 and d4bb09fa5a361a1d.

## 9. Synthetic integration evidence

The integration test uses an untrained tiny GPT-2 across twenty seeds. Controls had zero detections out of twenty, and marked samples had twenty out of twenty. More synthetic replacements weakened the signal. At sixty percent replacement, two of twenty remained detectable. This validates implementation mechanics. It does not establish prose quality, semantic robustness or a rare false-positive rate.

Sources: results/synthetic-smoke-20261008-085721/summary.json and results.csv. Untrained random tiny GPT-2.

## 10. Real text: insertion and detection work

This paired example uses the same prompt and generation seed, with watermark bias applied only to the marked sample. The control scores about 0.72, below the cutoff, while the marked sample scores 11.12. This demonstrates insertion and detection on this example, not a general detection rate. Raw-ID scores may differ from scores after decoding and re-tokenizing the visible text. The full study should retain both.

Sources: .runtime/ui-jobs/b0221f45c23f7596/output.json. Qwen/Qwen2.5-0.5B-Instruct. One paired example, raw generation-token scores.

## 11. The writing changes during paraphrasing

These exact excerpts come from the saved original and first paraphrase. Both discuss urban gardens, but the wording and some claims change. A valid attack evaluation should preserve meaning rather than generate unrelated text. We have observed topical similarity without a systematic fidelity assessment. The full passages also appear truncated at the configured token limit, which the next experiment must address.

Sources: Exact excerpts: .runtime/ui-jobs/d4bb09fa5a361a1d/output.json. Semantic fidelity has not been formally evaluated.

## 12. One rewrite removed detectable evidence

The original scores 11.122. The three recursive paraphrases score about -0.083, -1.078 and 0.082. Every rewrite is below four. The first rewrite has nearly the same checked-token count, 194 versus 195, but favored choices fall from 116 to 48. This is a concrete evasion example for the baseline. It is neither an aggregate failure rate nor evidence that every watermark can be removed.

Sources: .runtime/ui-jobs/d4bb09fa5a361a1d/output.json. One passage and recursive paraphrase chain, Qwen/Qwen2.5-0.5B-Instruct.

## 13. What this result actually establishes

The detector finds the original signal and misses it after this rewrite. The paraphrase is still AI-generated, so a failed watermark detection cannot establish human authorship. This result does not disprove all watermarking. Published reliability depends on residual fragments, text length and attack conditions. The next study must specify the attack model, quality requirements and false-positive operating point.

Sources: R2: Kirchenbauer et al. (ICLR 2024). On the Reliability of Watermarks for Large Language Models. https://proceedings.iclr.cc/paper_files/paper/2024/hash/d78e9e4316e1714fbb0f20be66f8044c-Abstract-Conference.html
R3: Sadasivan et al. (TMLR, revised 2025). Can AI-Generated Text be Reliably Detected?. https://arxiv.org/abs/2303.11156
.runtime/ui-jobs/d4bb09fa5a361a1d/output.json.

## 14. Limits of the current evidence

One small model and a few examples validate a workflow without characterizing performance across domains and adversaries. The synthetic controls are insufficient for rare false-positive claims. Truncation, semantic drift, and differences between raw and re-tokenized scoring need attention. Translation code exists but this presentation does not claim a completed translation benchmark.

Sources: VALIDATION.md historical synthetic results, current API outputs, evaluate.py, ui_worker.py.

## 15. Next controlled experiment

The proposed pilot uses twenty prompts and three seeds per prompt. Each run scores six conditions: ordinary control, original watermark, three recursive paraphrases, and back-translation. That yields 360 scored texts. Save complete passages, token IDs, exact revisions, private keys, lengths and results. Fix completion quality and attack instructions before scaling. This is planned work.

Sources: Proposed extension of prompts.json and evaluate.py: 20 x 3 x 6 = 360 scored texts.

## 16. Further research and comparisons

Measure the baseline at several strengths and lengths, then compare alternatives at matched false-positive rates and comparable quality. Semantic-context and distortion-free alignment methods are candidates, not guaranteed fixes. Test independent paraphrasers, multiple domains and languages, wrong keys and mixed documents. Longer text and windowed detection deserve attention because of the reliability paper.

Sources: R2: Kirchenbauer et al. (ICLR 2024). On the Reliability of Watermarks for Large Language Models. https://proceedings.iclr.cc/paper_files/paper/2024/hash/d78e9e4316e1714fbb0f20be66f8044c-Abstract-Conference.html
R4: Kuditipudi et al. (TMLR 2024). Robust Distortion-free Watermarks for Language Models. https://arxiv.org/abs/2307.15593
R5: Liu et al. (ICLR 2024). A Semantic Invariant Robust Watermark for Large Language Models. https://arxiv.org/abs/2310.06356
R6: Dathathri et al. (Nature 2024). Scalable watermarking for identifying large language model outputs. https://doi.org/10.1038/s41586-024-08025-4
R7: Tu et al. (ACL 2024). WaterBench: Towards Holistic Evaluation of Watermarks for Large Language Models. https://aclanthology.org/2024.acl-long.83/

## 17. Expected outcomes and deliverables

The research should produce a reproducible benchmark and a clear assessment of strengths and weaknesses. Report detection at declared empirical false-positive operating points, signal trajectories, uncertainty, output quality and computational costs. A narrow useful operating range is an honest possible outcome. A deployment or paraphrase-resistance claim should require held-out evidence under the stated threat model.

Sources: R7: Tu et al. (ACL 2024). WaterBench: Towards Holistic Evaluation of Watermarks for Large Language Models. https://aclanthology.org/2024.acl-long.83/
Proposed deliverables. No target performance is claimed as achieved.

## 18. Current conclusion

We have a functioning prototype with readable generation, keyed insertion, independent detection and recursive rewriting. Our observed paraphrase example removes detectable evidence from the baseline. The next milestone measures how consistently that occurs and which tradeoffs improve it. The research contribution should be a reproducible assessment with explicit failure conditions, potentially followed by an improved method if the evidence supports one.

Sources: 

## 19. References: foundations

Full source URLs appear below and in the relevant slide notes. Published results are distinct from our local measurements.

Sources: R1: Kirchenbauer et al. (ICML 2023). A Watermark for Large Language Models. https://proceedings.mlr.press/v202/kirchenbauer23a.html
R2: Kirchenbauer et al. (ICLR 2024). On the Reliability of Watermarks for Large Language Models. https://proceedings.iclr.cc/paper_files/paper/2024/hash/d78e9e4316e1714fbb0f20be66f8044c-Abstract-Conference.html
R3: Sadasivan et al. (TMLR, revised 2025). Can AI-Generated Text be Reliably Detected?. https://arxiv.org/abs/2303.11156

## 20. References: further research

Full source URLs appear below and in the relevant slide notes. Published results are distinct from our local measurements.

Sources: R4: Kuditipudi et al. (TMLR 2024). Robust Distortion-free Watermarks for Language Models. https://arxiv.org/abs/2307.15593
R5: Liu et al. (ICLR 2024). A Semantic Invariant Robust Watermark for Large Language Models. https://arxiv.org/abs/2310.06356
R6: Dathathri et al. (Nature 2024). Scalable watermarking for identifying large language model outputs. https://doi.org/10.1038/s41586-024-08025-4
R7: Tu et al. (ACL 2024). WaterBench: Towards Holistic Evaluation of Watermarks for Large Language Models. https://aclanthology.org/2024.acl-long.83/
