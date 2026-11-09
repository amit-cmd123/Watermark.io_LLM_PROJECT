# Watermark Lab: ten-slide presenter script

Completed local benchmark. Detailed explanations and primary citations for each slide.

## 1. Watermark Lab

Watermark Lab studies whether a signal inserted into the token choices of a language model remains detectable when its text is rewritten. We control generation and test with the same secret key. We have completed a real local comparison of three method settings across twenty explanatory topics, three seeds and a separate length sweep. The saved run contains 210 cells and 1,050 measurements. Our strongest engineering outcome is the complete insertion, inspection and recovery workflow. The main research finding is that original detection works in this study while recursive paraphrasing leaves very little recoverable signal. The following slides distinguish implementation, measured evidence and proposed work. Full depth appears in the speaker notes, and the accompanying guide contains the longer technical reference.

### Sources

- Local run research-20261021-115335-998767, completed 22 October 2026. Measurement snapshot: output/documentation/research-evidence-20261022.json. Full guide: WATERMARK_LAB_COMPLETE_GUIDE.md.

## 2. The problem and the reason for this study

Text often leaves the system that generated it. Copying removes surrounding metadata, and rewriting changes wording. Our question is whether a deliberately embedded keyed pattern can supply evidence of origin after those transformations. This differs from trying to classify every unknown paragraph as human or AI. We chose a green-list baseline because it changes sampling without retraining, provides an explicit detector and exposes a strength-versus-quality tradeoff. This was an engineering starting point, not evidence that it was the most robust available construction. The current threat model gives an attacker the marked passage and a paraphraser without our insertion processor. The detector then searches the transformed text for the original pattern. Useful success means quality-preserving transformed detection at a declared accidental-trigger rate. A negative result can still establish a reproducible failure condition and motivate a better construction.

### Sources

- Kirchenbauer et al. A Watermark for Large Language Models. ICML 2023. https://proceedings.mlr.press/v202/kirchenbauer23a.html
- Sadasivan et al. Can AI-Generated Text be Reliably Detected? TMLR, preprint revised 2025. https://arxiv.org/abs/2303.11156

## 3. The watermark changes next-token sampling

A token is a vocabulary unit rather than necessarily a whole word. The model produces logits for possible next tokens. Our scheme combines the secret key with the preceding token ID using HMAC-SHA256, then seeds PCG64 to create a deterministic permutation of the eligible vocabulary. Roughly one quarter is green. The actual fraction accounts for rounding the partition size. The logits processor adds delta to every green choice and leaves other candidates available. The marked sample uses delta 2 in the baseline and delta 3 in the stronger setting. The ordinary control receives no processor. The fixed condition uses one constant partition and a different distinct-token scoring unit. Generation uses the same paired seeds, but differing early choices change subsequent model contexts. HMAC use does not prove security of the whole construction: PCG64 is not a cryptographic PRNG. All diagram values illustrate the mechanism rather than a measured next-token probability distribution.

### Sources

- Kirchenbauer et al. A Watermark for Large Language Models. ICML 2023. https://proceedings.mlr.press/v202/kirchenbauer23a.html
- Local run research-20261021-115335-998767, completed 22 October 2026. Measurement snapshot: output/documentation/research-evidence-20261022.json. Full guide: WATERMARK_LAB_COMPLETE_GUIDE.md.
- Source: watermark.py, generate.py and research.py.

## 4. The score measures excess favored choices

The baseline detector counts distinct eligible adjacent token pairs. T is that evidence count, G is the count whose current token falls in the keyed green set, and gamma is the actual green fraction. It computes z=(G-gamma*T)/sqrt(T*gamma*(1-gamma)). The first completion token, special-token boundaries and repeated pairs do not contribute. The screenshot example has 116 green pairs out of 195, compared with an expectation around 48.75. That gives z approximately 11.12. The first rewrite has 48 green pairs out of 194, almost exactly the expectation of 48.5, producing z approximately -0.083. Its evidence count barely changes, so length alone does not explain the drop. Four is an experimental cutoff. A z-score is not an AI probability, and the normal tail is only an approximate null calculation. Dependence and domain effects require empirical calibration. The fixed method instead counts distinct eligible token IDs and intentionally exports no approximate p-value.

### Sources

- detect.py, research.py fixed_score. Historical saved chain: .runtime/ui-jobs/d4bb09fa5a361a1d/output.json.

## 5. The app connects writing to the evidence

The app runs locally with a plain HTML/CSS/JavaScript frontend and Python HTTP servers. The single-passage lab on port 8765 provides Generate and compare, Check a passage, Try rewriting, Batch experiment and Saved experiments. Generation displays the complete decoded prose before optional token highlights. Handoff buttons carry the key and configuration to the next task. Checking needs the tokenizer and key, not the generation weights. Recursive rewriting transforms the previous output three times without watermark bias. The research bench on port 8766 adds live progress, three-method comparisons, full saved passages, length diagnostics, reviews, checkpoints and CSV/JSON exports. Main-UI generated samples primarily show raw-ID scores with a separate text score, while the research benchmark scores visible text after re-tokenization. A numeric untrained fixture exists for mechanics, but cannot produce paragraphs or validate semantic robustness. The screenshot shows actual saved app layout from an earlier verification session, not the final results table. The servers are running for the presentation.

### Sources

- ui/index.html, ui/app.js, ui_worker.py, ui/research.html, ui/research.js and research_server.py. Screenshot: output/research-passages.jpg.

## 6. The controlled local experiment

The completed run uses Qwen2.5-0.5B-Instruct, temperature 0.8, top-p 0.95, base seed 1234 and main cutoff 4. Main generation prompts ask for two complete paragraphs around 120 words with a 256-token ceiling. The main design contains twenty manual explanatory topics, three seed indices and three method settings. Each cell has an ordinary control, marked original and three recursive paraphrases, giving 180 cells and 900 rows. A separate sweep adds caps 128 and 512 on the first five topics at seed index zero, giving thirty cells and 150 rows. Total: 210 cells and 1,050 scored rows. Controls are generated once for each topic/seed/cap and scored across methods, so those rows are not independent negative texts. The full plan represents 910 generation calls absent discarded attempts. Seed zero is development, seed one controls set a provisional cutoff of at least four, and seed two provides an exploratory held-out view. One key, one model and the same model for paraphrasing limit generalization. Translation and semantic methods are outside this completed run.

### Sources

- Local run research-20261021-115335-998767, completed 22 October 2026. Measurement snapshot: output/documentation/research-evidence-20261022.json. Full guide: WATERMARK_LAB_COMPLETE_GUIDE.md.
- Protocol: research.py, RESEARCH.md and prompts.json.

## 7. Rewriting removes most detectable evidence

Every main-stage original crosses the cutoff: sixty out of sixty per method. After the first rewrite, baseline detection is two of sixty (3.3%), stronger bias five of sixty (8.3%), and fixed vocabulary two of sixty (3.3%). After the second rewrite, all three have zero detections. After the third rewrite, baseline and stronger bias remain zero, while the fixed variant detects one of sixty (1.7%). Ordinary controls trigger zero of sixty for each detector, but the text is shared and these are AI-generated negatives, not a human corpus. The chart shows detection proportions rather than average z-scores. A non-monotonic individual chain or aggregate rate can occur because a later rewrite can retain or accidentally recover favored evidence. All three methods detect original insertion reliably in this measured setup. Stronger first-round survival is a small exploratory difference, not a validated robust winner, and requires paired analysis, quality review and independent confirmation. Topic correlation and limited sample size constrain uncertainty.

### Sources

- Local run research-20261021-115335-998767, completed 22 October 2026. Measurement snapshot: output/documentation/research-evidence-20261022.json. Full guide: WATERMARK_LAB_COMPLETE_GUIDE.md.

## 8. Research foundations and alternative methods

Kirchenbauer et al. ICML 2023 supply the green-list insertion and independent statistical-detection foundation. Their reliability follow-up studies retained n-grams after human/model edits and the value of sufficient observed text and span-sensitive detection. Sadasivan et al. stress-test detector families using recursive paraphrasing and spoofing. Different methods, lengths and attack conditions explain why studies can report useful retention or strong failures. Zhao et al. propose Unigram-Watermark with a fixed grouping strategy and conditional theoretical guarantees. Our fixed variant uses project-specific keying and unique-token scoring, so it is not an exact reproduction and does not inherit the theorem. Liu et al. use semantic context representations and a trained watermark transformation in SIR. It needs dedicated assets and is pending here. Kuditipudi et al. couple sampling to a keyed random sequence and align text for detection, with distribution-preservation properties under their construction. SynthID-Text studies tournament sampling, quality and serving efficiency at scale. WaterBench motivates joint quality and detection evaluation under fairer comparisons. None of those alternative papers has been fully reproduced in this app. The full titles and primary source links follow.

### Sources

- Kirchenbauer et al. A Watermark for Large Language Models. ICML 2023. https://proceedings.mlr.press/v202/kirchenbauer23a.html
- Kirchenbauer et al. On the Reliability of Watermarks for Large Language Models. ICLR 2024. https://arxiv.org/abs/2306.04634
- Sadasivan et al. Can AI-Generated Text be Reliably Detected? TMLR, preprint revised 2025. https://arxiv.org/abs/2303.11156
- Zhao et al. Provable Robust Watermarking for AI-Generated Text. ICLR 2024. https://arxiv.org/abs/2306.17439
- Liu et al. A Semantic Invariant Robust Watermark for Large Language Models. ICLR 2024. https://arxiv.org/abs/2310.06356
- Kuditipudi et al. Robust Distortion-free Watermarks for Language Models. TMLR 2024. https://arxiv.org/abs/2307.15593
- Dathathri et al. Scalable watermarking for identifying large language model outputs. Nature 2024. https://doi.org/10.1038/s41586-024-08025-4
- Tu et al. WaterBench: Towards Holistic Evaluation of Watermarks for Large Language Models. ACL 2024. https://aclanthology.org/2024.acl-long.83/

## 9. The quality and validation gaps

Original detector scores rise with stronger bias, but completeness diagnostics show a cost that needs investigation. Original cap flags in the main stage are 9/60 for the baseline, 25/60 for stronger bias and 35/60 for fixed vocabulary. A cap flag means the generation reached its configured maximum without a final EOS token. It is a completion diagnostic, not a fluent-writing grade. All current human-review counts are zero. Meaning, claims and fluency have therefore not received a systematic review. Zero observed ordinary-control triggers does not establish a zero error rate, and these controls do not estimate human false-positive rates. The same small model generates originals and paraphrases, and the current run uses one key. Exact-token dependence makes the baseline fragile to replacing a token and its next-token context. Fixed partitioning removes that context dependence but a paraphraser can still replace the token itself. We need fair comparison at controlled empirical errors and acceptable quality before selecting a method. Translation code exists, but no translation or semantic-watermark comparison is included in the completed run.

### Sources

- Local run research-20261021-115335-998767, completed 22 October 2026. Measurement snapshot: output/documentation/research-evidence-20261022.json. Full guide: WATERMARK_LAB_COMPLETE_GUIDE.md.
- Tu et al. WaterBench: Towards Holistic Evaluation of Watermarks for Large Language Models. ACL 2024. https://aclanthology.org/2024.acl-long.83/

## 10. The next milestone is a fair robustness gain

The end goal is a watermark that retains useful detection after meaning-preserving transformations while keeping accidental detections and writing degradation low. Step one is to freeze this completed baseline, review passage fidelity and completeness, and calibrate against a documented human-authored negative corpus with a separate confirmation split. Step two is to reproduce a faithful reference candidate such as SIR, including its semantic encoder and trained transformation assets. An embedding similarity score alone would only assess fidelity and would not implement semantic watermarking. Step three is to compare the candidate on independent paraphrasers, new keys and models, and a fresh test set at matched empirical false-positive rates and comparable quality. Actual translation is another distinct attack to add when models are available. Report raw denominators, paired successes/failures, uncertainty, cap limits, quality ratings and runtime. The desired outcome is a reproducible useful operating range, with an improvement claimed only when independent measurements support it. The current testbed and completed baseline make that next comparison possible.

### Sources

- Local run research-20261021-115335-998767, completed 22 October 2026. Measurement snapshot: output/documentation/research-evidence-20261022.json. Full guide: WATERMARK_LAB_COMPLETE_GUIDE.md.
- Liu et al. A Semantic Invariant Robust Watermark for Large Language Models. ICLR 2024. https://arxiv.org/abs/2310.06356
- Tu et al. WaterBench: Towards Holistic Evaluation of Watermarks for Large Language Models. ACL 2024. https://aclanthology.org/2024.acl-long.83/
