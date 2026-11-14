"""Detector lab: how much of the loss after rewriting can the detector alone recover?

Re-scores saved texts (token IDs from an earlier research run) with different detectors and
no regeneration, so it is CPU-only and quick. Negatives are the run's unwatermarked controls;
every detector is judged by AUC and by detection at a fixed false-alarm rate measured on them,
not by a fixed z cutoff.

Detectors (all on the same previous-token scheme the texts were generated with):
  standard      z over every scored word transition (the current detector)
  content       z over transitions that land on a content word only
  window        best z over sliding windows of transitions (finds partly rewritten text)
  content+window  both"""
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / ".runtime" / "huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from transformers import AutoTokenizer  # noqa: E402

import watermark  # noqa: E402

GAMMA = 0.25
STOP = set("""the a an and or of to in on for with is are was were be been it its this that these those as by at from which who
their they them can also not but such more most may will would could should has have had than then into out up down over
under about between both each other some any many much very so if when while where how what why their there here only just
like make made makes use used using help helps often well even through during across within without against among""".split())


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def tpr_at_fpr(pos, neg, fpr):
    neg = np.sort(np.asarray(neg, float))[::-1]
    k = int(np.floor(fpr * len(neg)))
    thr = neg[min(k, len(neg) - 1)]
    return float((np.asarray(pos) > thr).mean()), float((neg > thr).mean())


def zscore(g, n):
    return (g - GAMMA * n) / np.sqrt(n * GAMMA * (1 - GAMMA)) if n else 0.0


def best_window(flags, size):
    flags = np.asarray(flags, np.int64)
    if len(flags) <= size:
        return zscore(flags.sum(), len(flags))
    sums = np.convolve(flags, np.ones(size, np.int64), "valid")
    return float(((sums - GAMMA * size) / np.sqrt(size * GAMMA * (1 - GAMMA))).max())


def main():
    run = sorted(ROOT.glob("results/research-*"))[-1]
    samples = run / "samples"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    scheme = watermark.WatermarkScheme(watermark.vocabulary_ids(tok), (run / "watermark.key").read_bytes(), GAMMA)
    content_cache = {}

    def is_content(i):
        if i not in content_cache:
            t = tok.convert_ids_to_tokens(int(i)) or ""
            word = t[1:] if t.startswith("Ġ") else ""
            content_cache[i] = bool(re.fullmatch(r"[a-z]{4,}", word.lower())) and word.lower() not in STOP
        return content_cache[i]

    def flags(ids):
        seen, out = set(), []
        for prev, cur in zip(ids, ids[1:]):
            if prev not in scheme.eligible or cur not in scheme.eligible or (prev, cur) in seen:
                continue
            seen.add((prev, cur))
            out.append((cur, bool((scheme.green_ids(prev) == cur).any())))
        return out

    def detectors(ids):
        f = flags(ids)
        g = np.array([x[1] for x in f], np.int64)
        c = np.array([is_content(x[0]) for x in f], bool)
        gc = g[c]
        return {"standard": zscore(g.sum(), len(g)), "content": zscore(gc.sum(), len(gc)),
                "window": best_window(g, 64), "content+window": best_window(gc, 40)}, len(g)

    def load(path):
        d = json.loads(path.read_text(encoding="utf-8"))
        ids = d.get("raw_ids", d.get("ids"))
        ids = ids if isinstance(ids, list) else json.loads(ids)
        return ids, float(d["z_score"]) if "z_score" in d else None

    controls = sorted(samples.glob("control-*-n256.json"))
    print(f"run {run.name}: {len(controls)} controls", flush=True)
    neg, parity = {k: [] for k in ("standard", "content", "window", "content+window")}, []
    for p in controls:
        ids, stored = load(p)
        s, _ = detectors(ids)
        for k in neg:
            neg[k].append(s[k])

    result = {"kind": "detector-lab", "run": run.name, "created_at": datetime.now().isoformat(timespec="seconds"),
              "negatives": len(controls), "variants": {}}
    for variant in ("baseline", "stronger"):
        result["variants"][variant] = {}
        for cond in ("original", "paraphrase_1"):
            files = sorted(samples.glob(f"p*-s*-n256-{variant}-{cond}.json"))
            pos = {k: [] for k in neg}
            for p in files:
                ids, stored = load(p)
                s, n = detectors(ids)
                for k in pos:
                    pos[k].append(s[k])
                if cond == "original" and len(parity) < 400:
                    parity.append(abs(s["standard"] - stored))
            row = {}
            for k in pos:
                t2, f2 = tpr_at_fpr(pos[k], neg[k], 0.02)
                row[k] = {"auc": auc(pos[k], neg[k]), "tpr_at_2pct_fpr": t2, "actual_fpr": f2,
                          "mean_z": float(np.mean(pos[k]))}
            row["standard"]["tpr_at_z4"] = float(np.mean(np.array(pos["standard"]) >= 4))
            result["variants"][variant][cond] = {"n": len(files), "detectors": row}
            print(f"{variant:9s} {cond:13s} n={len(files)}  " + "  ".join(
                f"{k}: AUC {row[k]['auc']:.2f} TPR@2%FPR {row[k]['tpr_at_2pct_fpr']:.2f}" for k in row), flush=True)
    result["parity_max_abs_diff_vs_stored_z"] = float(max(parity)) if parity else None
    print("max |recomputed z - stored z| on originals:", result["parity_max_abs_diff_vs_stored_z"], flush=True)
    out = ROOT / "results" / ("detector-lab-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True)
    (out / "summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out / "status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    print("Results:", out, flush=True)


if __name__ == "__main__":
    main()
