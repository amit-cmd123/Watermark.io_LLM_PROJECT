# Roadmap: seven generations

Status key: DONE = built and tested here. NEXT = designed, not built. Claims about
robustness stay unproven until the real-model experiment (G1 run) completes.

| Gen | Goal | Technique | Exit evidence |
|---|---|---|---|
| G1 | Baseline | KGW green list, 1-token context, z-test (`watermark.py`, `detect.py`) | 7 unit tests DONE; real 20-prompt attack run NOT done |
| G2 | Fast, correct engine | Stateless 32-bit hash partitions, GPU mask, context window h, span-scan detector, empirical null (`engine.py`) | 13 tests pass; 4.3 -> 1.5 ms/step at 128k vocab DONE |
| G3 | Quality-preserving sampling | SynthID-style tournament sampling (g-values, m layers), Gumbel-max variant, speculative-decoding hook | Perplexity/win-rate vs unwatermarked at equal detection power |
| G4 | Semantic + cross-lingual robustness | Sentence-embedding LSH keys for the partition (SIR/SemStamp/PMarK/DEW family) layered on G2 token signal | Detection after PEGASUS/DIPPER-style paraphrase and back-translation, matched length |
| G5 | Security | Pseudorandom-code embedding, signature-based public detectability, key rotation + multi-key attribution, spoof tests (DITTO-style learned imitation) | Spoof success rate vs query budget; false-positive on human text at 1e-4 |
| G6 | Learned / adaptive | Entropy-adaptive delta, learned embedder+detector trained against a paraphraser (RL framing), radioactivity probes for distillation tracing | Pareto of quality / detectability / robustness beats G3-G4 |
| G7 | Provenance fabric | Watermark + signed metadata + output-retrieval index, agent-action watermarking, audit service | End-to-end red-team report |

## Order of work
1. Finish the G1 real run (`run.ps1 experiment -Model Qwen/Qwen2.5-0.5B-Instruct -Limit 1`, then 20). HF is reachable; interpreter in `.runtime/python-path.txt` has torch 2.14 + CUDA on an RTX 4070 (8 GB).
2. Wire `engine.py` into `evaluate.py` as `--scheme hash32` and rerun to compare G1 vs G2 on real text.
3. Build a human-text null set (e.g. 1,000+ documents) and use `engine.empirical_null` to replace the normal-tail threshold.
4. Build G3, then G4, each as a separate scheme version with its own agreement tests.

## Literature used
- Kirchenbauer et al., A Watermark for LLMs (ICML 2023); On the Reliability of Watermarks (ICLR 2024)
- Dathathri et al., Scalable watermarking, SynthID-Text (Nature 2024): tournament sampling, ~0.6% latency overhead, 20M-response live study
- Liu et al., Semantic Invariant Robust Watermark (ICLR 2024); SemStamp; PMarK (ICLR 2026); Dual Semantic Embeddings, arXiv 2606.31602
- Fairoze et al., Publicly-Detectable Watermarking; Christ & Gunn, pseudorandom codes
- Sadasivan et al., Can AI-Generated Text be Reliably Detected? (recursive paraphrase)
- DITTO (arXiv 2510.10987) and CDG-KD (arXiv 2504.17480): spoofing/scrubbing via distillation
- RL framework for robust watermarking, arXiv 2510.21053; SBW agent watermarking, arXiv 2610.08668

## Revision 1 (14 Nov 2026): what the real data says

Source: `results/research-20261021-115335-998767` (Qwen2.5-0.5B-Instruct, 20 prompts, self-paraphrase).

| Finding | Number |
|---|---|
| Detection of originals / after paraphrase 1 / 2 / 3 (previous-token, delta 2) | 60/60, 2/60, 0/60, 0/60 |
| Same, delta 3 | 60/60, 5/60, 0/60, 0/60 |
| Same, fixed vocabulary (Unigram-style), delta 3 | 60/60, 2/60, 0/60, 1/60 |
| Content words of the original kept in paraphrase 1 | 0.32 (an unrelated sibling answer to the same prompt: 0.31) |
| Bigrams kept | 0.13 (sibling: 0.10) |
| Whole-text embedding cosine, original vs paraphrase 1 / sibling / other topic | 0.89 / 0.83 / 0.09 |
| Mean best sentence match cosine, paraphrase 1 / sibling | 0.75 / 0.67 |

Reading: the 0.5B self-paraphrase keeps the topic and almost nothing else. It is close to
resampling the answer. Consequences:
1. No token- or word-level scheme (including any better G2/G3 variant) can survive this attack: the
   words are not there. Tuning delta/context will not fix it.
2. Sentence-level semantic keys have a thin margin (0.75 vs 0.67 against chance), so a semantic
   watermark will be near chance on this attacker.
3. The attack itself is under-specified: it is not meaning-preserving enough to be called a paraphrase.
   Any survival number reported against it is not a statement about faithful paraphrase.

## Revised plan
0. Measurement first. Add a fidelity gate (best-sentence-match + entailment) and a stronger,
   sentence-by-sentence paraphraser (Qwen2.5-1.5B/3B). Report detection as a curve over attack
   fidelity, not one number. Calibrate thresholds on human text at fixed FPR (`engine.empirical_null`).
1. Token layer = G2 engine, h=1, entropy-weighted detector (EWD-style). Covers edits, partial
   rewrites, mixing, copy-paste, light paraphrase.
2. Semantic layer = k-SemStamp-style sentence partitions via embedding LSH + rejection sampling.
   Only worth keeping if it beats the token layer at matched fidelity.
3. Fusion detector (Fisher on token + semantic p-values, empirical null).
4. Backstop = provenance retrieval (embedding index of issued outputs). Note sibling answers are
   hard negatives (0.83 vs 0.89), so test it on same-prompt siblings.
5. Adaptive attackers before any robustness claim: green-list reverse-engineering (Rastogi & Pruthi
   2024: Unigram 62.8 -> 0.2 and SIR 24.0 -> 3.8 TPR@1%FPR under their attack with DIPPER L60),
   cross-lingual removal (CWRA; X-SIR defence), key rotation.

Not read in full: arXiv 2607.16010 (snippet claims 100% removal of KGW/Unigram after paraphrase).
