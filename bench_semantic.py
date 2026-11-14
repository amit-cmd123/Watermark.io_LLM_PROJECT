"""G4 evaluation: semantic watermark vs token schemes under rewriting, on a real model.

Same prompts and seeds for every arm. Attacks are text-level and rewrite the text:
  * sentence-by-sentence English->French->English
  * sentence-by-sentence LLM paraphrase (aligned, so semantic retention is measurable)
  * whole-text LLM paraphrase (the attack that removed the token watermark earlier)
Each attack records its fidelity (embedding similarity to the original) so a result
can be read against how faithful the rewrite was. Writes results/benchmark-g4-<time>/summary.json
in the same format the dashboard's Scheme comparison screen reads."""
import argparse
import glob
import json
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch

import bench_eval as be
import engine
import semantic
from attacks import paraphrase_prompt
from generate import Generator

ROOT = be.ROOT
KEY = be.KEY


def paraphrase_batch(gen, sentences, seed):
    tok = gen.tokenizer
    prompts = [tok.apply_chat_template([{"role": "user", "content":
               "Rewrite the sentence below using different wording. Keep the meaning. "
               "Return only the rewritten sentence.\n\nSentence: " + s}], tokenize=False,
               add_generation_prompt=True) for s in sentences]
    out_text = []
    side, tok.padding_side = tok.padding_side, "left"
    try:
        for i in range(0, len(prompts), 24):
            enc = tok(prompts[i:i + 24], return_tensors="pt", padding=True, add_special_tokens=False).to(gen.device)
            torch.manual_seed(seed + i)
            with torch.inference_mode():
                out = gen.model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95, top_k=0,
                                         max_new_tokens=96, pad_token_id=tok.pad_token_id)
            for row in out[:, enc["input_ids"].shape[1]:]:
                text = tok.decode(row, skip_special_tokens=True).strip()
                out_text.append(text.split("\n")[0].strip() or " ")
    finally:
        tok.padding_side = side
    return out_text


def best_match(embedder, a_units, b_units):
    if not a_units or not b_units:
        return 0.0
    ea, eb = embedder.embed(a_units), embedder.embed(b_units)
    return float((ea @ eb.T).max(1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=250)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--bits", type=int, default=3)
    args = ap.parse_args()

    out = ROOT / "results" / ("benchmark-g4-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True)

    def status(stage, done=0, total=0, state="running"):
        (out / "status.json").write_text(json.dumps(
            {"status": state, "stage": stage, "completed": done, "total": total}), encoding="utf-8")

    status("loading models")
    gen = Generator(args.model)
    embedder = semantic.Embedder(gen.device)
    translator = be.SentenceTranslator(gen.device)
    vocab = be.v1.vocabulary_ids(gen.tokenizer)
    token_arms = {
        "v2 hash, context 1": engine.HashScheme(KEY, vocab, 0.25, context=1),
        "v2 min-hash, context 4": engine.HashScheme(KEY, vocab, 0.25, context=4, agg="min"),
    }

    # calibrate the semantic null on earlier unwatermarked samples (same domain, different run)
    ref_units = []
    for path in sorted(glob.glob(str(ROOT / "results" / "research-*" / "samples" / "control-*.json")))[:60]:
        ref_units += semantic.segment(json.loads(open(path, encoding="utf-8").read())["text"])
    scheme = semantic.SemanticScheme(KEY, bits=args.bits).fit(embedder.embed(ref_units))
    print(f"semantic scheme: {scheme.version}, calibrated on {len(ref_units)} sentences, null rate {scheme.null_rate:.3f}", flush=True)

    prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))[:args.limit]
    items = [(p, 1234 + 1000 * s + i) for s in range(args.seeds) for i, p in enumerate(prompts)]
    settings = SimpleNamespace(temperature=0.8, top_p=0.95, max_new_tokens=args.tokens)
    gen.generate("Warm up.", 0, SimpleNamespace(temperature=0.8, top_p=0.95, max_new_tokens=16))

    arms = ["control (no watermark)", *token_arms, "G4 semantic"]
    total, done, data = len(arms) * len(items), 0, {}
    for arm in arms:
        rows = []
        for prompt, seed in items:
            done += 1
            status(f"generating: {arm}", done, total)
            t0 = time.perf_counter()
            if arm == "G4 semantic":
                text, info = semantic.generate(gen, embedder, scheme, prompt, seed, settings,
                                               candidates=args.candidates, max_units=max(6, args.tokens // 24))
                n_tok, extra = info["tokens"], info
            else:
                proc = engine.make_processor(token_arms[arm], args.delta) if arm in token_arms else None
                text, ids = gen.generate(prompt, seed, settings, proc)
                n_tok, extra = len(ids), {}
            dt = time.perf_counter() - t0
            rows.append(dict(prompt=prompt, seed=seed, text=text, tps=n_tok / dt, ppl=be.perplexity(gen, text), **extra))
        data[arm] = rows
        print(arm, "tokens/s", round(float(np.mean([r["tps"] for r in rows])), 1), flush=True)

    # attacks (text -> text), per watermarked arm
    conds = ["none", "translate to French and back (per sentence)", "paraphrase each sentence", "paraphrase whole text"]
    attacked, fidelity, retention = {}, {c: [] for c in conds[1:]}, {c: [] for c in conds[1:]}
    wm_arms = [a for a in arms if a != "control (no watermark)"]
    for k, arm in enumerate(wm_arms):
        status(f"attacking: {arm}", k, len(wm_arms))
        texts = [r["text"] for r in data[arm]]
        unit_lists = [semantic.segment(t) for t in texts]
        flat = [u for us in unit_lists for u in us]
        cuts = np.cumsum([0] + [len(us) for us in unit_lists])
        regroup = lambda xs: [xs[cuts[i]:cuts[i + 1]] for i in range(len(texts))]
        bt = regroup(translator.back_translate(flat))  # one translation per unit, batched
        pp = regroup(paraphrase_batch(gen, flat, 99))
        whole = []
        for i, (prompt, seed) in enumerate(items):
            status(f"paraphrasing whole text: {arm}", i, len(items))
            whole.append(gen.generate(paraphrase_prompt(texts[i]), seed + 5000, settings)[0])
        attacked[arm] = {conds[0]: texts, conds[1]: [" ".join(x) for x in bt], conds[2]: [" ".join(x) for x in pp], conds[3]: whole}
        aligned = {conds[1]: bt, conds[2]: pp}
        for c in conds[1:3]:
            sims = []
            for us, vs in zip(unit_lists, aligned[c]):
                if us:
                    ea, eb = embedder.embed(us), embedder.embed(vs)
                    sims.append(float((ea * eb).sum(1).mean()))
            fidelity[c].append(float(np.mean(sims)))
        fidelity[conds[3]].append(float(np.mean([best_match(embedder, u, semantic.segment(w)) for u, w in zip(unit_lists, whole)])))

    def tok_z(arm, text):
        z = engine.detect(gen.tokenizer.encode(text, add_special_tokens=False), token_arms[arm])["z_score"]
        return 0.0 if z is None else float(z)

    def sem_z(text):
        z = semantic.detect_embeddings(embedder.embed(semantic.segment(text)), scheme)["z_score"]
        return 0.0 if z is None else float(z)

    base_tps = float(np.mean([r["tps"] for r in data["control (no watermark)"]]))
    base_ppl = float(np.mean([r["ppl"] for r in data["control (no watermark)"] if r["ppl"]]))
    control_texts = [r["text"] for r in data["control (no watermark)"]]
    summary = {"kind": "benchmark", "model": args.model, "delta": args.delta, "gamma": 0.25, "threshold": 4.0,
               "samples": len(items), "max_new_tokens": args.tokens, "conditions": conds, "schemes": {},
               "created_at": datetime.now().isoformat(timespec="seconds"),
               "fidelity": {c: float(np.mean(v)) for c, v in fidelity.items()},
               "semantic_null_rate": scheme.null_rate, "semantic_calibration_sentences": len(ref_units),
               "control": {"tokens_per_second": base_tps, "perplexity": base_ppl, "sample": control_texts[0]}}
    for arm in wm_arms:
        zf = (lambda t: sem_z(t)) if arm == "G4 semantic" else (lambda t, a=arm: tok_z(a, t))
        neg = [zf(t) for t in control_texts]
        result = {}
        for c in conds:
            texts = attacked[arm][c]
            zs = [zf(t) for t in texts]
            kept = float(np.mean([len(t) / max(1, len(o["text"])) for t, o in zip(texts, data[arm])]))
            result[c] = {"mean_z": float(np.mean(zs)), "tpr": float(np.mean([z >= 4.0 for z in zs])), "auc": be.auc(zs, neg),
                         "length_kept": kept, "z": [round(z, 3) for z in zs]}
        rows = data[arm]
        tps = float(np.mean([r["tps"] for r in rows]))
        ppl = float(np.mean([r["ppl"] for r in rows if r["ppl"]]))
        entry = {"tokens_per_second": tps, "overhead_pct": (base_tps / tps - 1) * 100, "perplexity": ppl,
                 "perplexity_vs_control_pct": (ppl / base_ppl - 1) * 100, "fpr": float(np.mean([z >= 4.0 for z in neg])),
                 "control_z": [round(z, 3) for z in neg], "conditions": result, "sample": rows[0]["text"],
                 "back_translation_sample": attacked[arm][conds[1]][0]}
        if arm == "G4 semantic":
            entry["units_valid_share"] = float(np.mean([r["valid_units"] / max(1, r["units"]) for r in rows]))
        summary["schemes"][arm] = entry
        print(arm, {c: (round(result[c]["tpr"], 2), round(result[c]["auc"], 3)) for c in conds}, flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    status("complete", total, total, "complete")
    print("Results:", out, flush=True)


if __name__ == "__main__":
    main()
