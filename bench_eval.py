"""Head-to-head benchmark of watermark schemes on a real model.

Measures, per scheme and on identical prompts/seeds:
  * generation throughput (tokens/s, includes the logits-processor cost)
  * text quality (perplexity of the completion under the unwatermarked base model)
  * detection accuracy (mean z, TPR at the nominal threshold, AUC against controls,
    FPR on controls) under token edits and sentence-wise English-French-English translation
Writes results/benchmark-<time>/summary.json (+status.json while running)."""
import argparse
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / ".runtime" / "huggingface")  # a global HF_HOME points at another project
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

import detect as v1_detect
import engine
import watermark as v1
from generate import Generator

KEY = b"benchmark-key-0123456789abcdef"
THRESHOLD = 4.0


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def tpr_at_fpr(pos, neg, fpr):
    """Share of `pos` above the threshold that lets at most `fpr` of `neg` through."""
    neg = np.sort(np.asarray(neg, float))[::-1]
    thr = neg[min(int(np.floor(fpr * len(neg))), len(neg) - 1)]
    return float((np.asarray(pos, float) > thr).mean())


def build_schemes(vocab, delta):
    def v2(name, **kw):
        s = engine.HashScheme(KEY, vocab, gamma=0.25, context=kw["context"], agg=kw.get("agg", "chain"))
        return dict(name=name, scheme=s, processor=lambda s=s: engine.make_processor(s, delta),
                    z=lambda ids, s=s: engine.detect(ids, s)["z_score"])
    s1 = v1.WatermarkScheme(vocab, KEY, 0.25)
    return [
        dict(name="v1 previous-token", scheme=s1, processor=lambda: v1.make_processor(s1, delta),
             z=lambda ids: v1_detect.detect_ids(ids, s1)["z_score"]),
        v2("v2 hash, context 1", context=1),
        v2("v2 hash, context 4", context=4),
        v2("v2 min-hash, context 4", context=4, agg="min"),
    ]


def sentences(text):
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return parts or [text]


class SentenceTranslator:
    """Sentence-by-sentence en->fr->en. Per-sentence avoids Marian silently dropping long inputs."""

    def __init__(self, device):
        from transformers import MarianMTModel, MarianTokenizer
        self.device = device
        self.models = {}
        for name in ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"):
            tok = MarianTokenizer.from_pretrained(name, local_files_only=True)
            mod = MarianMTModel.from_pretrained(name, local_files_only=True).to(device).eval()
            self.models[name] = (tok, mod)

    def _run(self, name, batch):
        tok, mod = self.models[name]
        out = []
        for i in range(0, len(batch), 32):
            chunk = batch[i:i + 32]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            with torch.inference_mode():
                gen = mod.generate(**enc, num_beams=4, max_new_tokens=256)
            out += tok.batch_decode(gen, skip_special_tokens=True)
        return out

    def back_translate(self, texts):
        flat, spans = [], []
        for t in texts:
            s = sentences(t)
            spans.append((len(flat), len(flat) + len(s)))
            flat += s
        french = self._run("Helsinki-NLP/opus-mt-en-fr", flat)
        english = self._run("Helsinki-NLP/opus-mt-fr-en", french)
        return [" ".join(english[a:b]) for a, b in spans]


def perplexity(gen, text):
    ids = gen.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(gen.device)
    if ids.shape[1] < 4:
        return None
    with torch.inference_mode():
        loss = gen.model(ids, labels=ids).loss
    return float(math.exp(loss.item()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--no-translate", action="store_true")
    ap.add_argument("--resume", help="results folder of an interrupted run; finished arms are reused")
    args = ap.parse_args()

    out = Path(args.resume) if args.resume else ROOT / "results" / ("benchmark-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    def status(stage, done=0, total=0, state="running"):
        (out / "status.json").write_text(json.dumps(
            {"status": state, "stage": stage, "completed": done, "total": total}), encoding="utf-8")

    status("loading model")
    gen = Generator(args.model)
    vocab = v1.vocabulary_ids(gen.tokenizer)
    vocab_arr = np.array(vocab)
    schemes = build_schemes(vocab, args.delta)
    prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))[:args.limit]
    items = [(p, 1234 + 1000 * s + i) for s in range(args.seeds) for i, p in enumerate(prompts)]
    settings = SimpleNamespace(temperature=0.8, top_p=0.95, max_new_tokens=args.tokens)
    gen.generate("Warm up.", 0, SimpleNamespace(temperature=0.8, top_p=0.95, max_new_tokens=16))

    arms = [dict(name="control (no watermark)", processor=lambda: None)] + schemes
    total = len(arms) * len(items)
    data, done = {}, 0
    for arm in arms:
        ckpt = out / ("arm-" + re.sub(r"\W+", "-", arm["name"]).strip("-") + ".json")
        if ckpt.exists():  # finished before an interruption: reuse it
            data[arm["name"]] = json.loads(ckpt.read_text(encoding="utf-8"))
            done += len(items)
            print(arm["name"], "reused from checkpoint", flush=True)
            continue
        rows = []
        for prompt, seed in items:
            done += 1
            status(f"generating: {arm['name']}", done, total)
            t0 = time.perf_counter()
            text, ids = gen.generate(prompt, seed, settings, arm["processor"]())
            dt = time.perf_counter() - t0
            rows.append(dict(text=text, ids=ids, tps=len(ids) / dt, ppl=perplexity(gen, text)))
        data[arm["name"]] = rows
        tmp = ckpt.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows), encoding="utf-8")
        tmp.replace(ckpt)  # atomic: a kill never leaves a half-written arm
        print(arm["name"], "tokens/s", round(float(np.mean([r["tps"] for r in rows])), 1), flush=True)

    translator = None
    if not args.no_translate:
        status("loading translators")
        translator = SentenceTranslator(gen.device)
    rng = np.random.default_rng(7)

    def sub(ids, p):
        ids = list(ids)
        for i in np.flatnonzero(rng.random(len(ids)) < p):
            ids[i] = int(rng.choice(vocab_arr))
        return ids

    def delete(ids, p):
        return [t for t in ids if rng.random() >= p]

    conditions = ["none", "10% words replaced", "20% words replaced", "30% words replaced",
                  "20% words deleted"] + ([] if translator is None else ["translate to French and back"])
    control_ids = [r["ids"] for r in data["control (no watermark)"]]
    summary = {"kind": "benchmark", "model": args.model, "delta": args.delta, "gamma": 0.25,
               "threshold": THRESHOLD, "samples": len(items), "max_new_tokens": args.tokens,
               "conditions": conditions, "schemes": {}, "created_at": datetime.now().isoformat(timespec="seconds")}
    base_tps = float(np.mean([r["tps"] for r in data["control (no watermark)"]]))
    base_ppl = float(np.mean([r["ppl"] for r in data["control (no watermark)"] if r["ppl"]]))
    summary["control"] = {"tokens_per_second": base_tps, "perplexity": base_ppl,
                          "sample": data["control (no watermark)"][0]["text"]}

    for k, sch in enumerate(schemes):
        status(f"scoring: {sch['name']}", k, len(schemes))
        rows = data[sch["name"]]
        neg = [sch["z"](ids) or 0.0 for ids in control_ids]
        bt_ids = None
        if translator is not None:
            bt = translator.back_translate([r["text"] for r in rows])
            bt_ids = [gen.tokenizer.encode(t, add_special_tokens=False) for t in bt]
            bt_text = bt[0]
        result = {}
        for cond in conditions:
            if cond == "none":
                attacked = [r["ids"] for r in rows]
            elif cond.startswith("translate"):
                attacked = bt_ids
            elif "replaced" in cond:
                attacked = [sub(r["ids"], int(cond[:2]) / 100) for r in rows]
            else:
                attacked = [delete(r["ids"], 0.2) for r in rows]
            zs = [sch["z"](ids) or 0.0 for ids in attacked]
            kept = float(np.mean([len(a) / max(1, len(r["ids"])) for a, r in zip(attacked, rows)]))
            result[cond] = {"mean_z": float(np.mean(zs)), "tpr": float(np.mean([z >= THRESHOLD for z in zs])),
                            "auc": auc(zs, neg), "tpr_fpr5": tpr_at_fpr(zs, neg, 0.05), "tpr_fpr1": tpr_at_fpr(zs, neg, 0.01), "length_kept": kept, "z": [round(float(z), 3) for z in zs]}
        tps = float(np.mean([r["tps"] for r in rows]))
        ppl = float(np.mean([r["ppl"] for r in rows if r["ppl"]]))
        summary["schemes"][sch["name"]] = {
            "tokens_per_second": tps, "overhead_pct": (base_tps / tps - 1) * 100,
            "perplexity": ppl, "perplexity_vs_control_pct": (ppl / base_ppl - 1) * 100,
            "fpr": float(np.mean([z >= THRESHOLD for z in neg])), "control_z": [round(float(z), 3) for z in neg],
            "conditions": result, "sample": rows[0]["text"],
            "back_translation_sample": bt_text if translator is not None else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    status("complete", total, total, "complete")
    print("Results:", out, flush=True)


if __name__ == "__main__":
    main()
