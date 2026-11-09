"""Offline integration exercise with random GPT-2 weights, NOT research results.

Tests real Transformers generation hooks and detection across twenty seeds.
Synthetic token substitutions are not paraphrasing or back-translation.
"""
import csv
from datetime import datetime
import json
from pathlib import Path
import secrets

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessorList

from detect import detect_ids
from watermark import WatermarkScheme, make_processor


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPT2Config(vocab_size=128, n_positions=260, n_ctx=260,
                        n_embd=32, n_layer=1, n_head=2,
                        bos_token_id=None, eos_token_id=None, pad_token_id=127)
    model = GPT2LMHeadModel(config).to(device).eval()
    scheme = WatermarkScheme(range(127), b"offline-smoke-test-key", 0.25)
    processor = make_processor(scheme, 2)
    output = Path("results") / ("synthetic-smoke-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for sample in range(20):
        prompt = torch.tensor([[sample + 1, sample + 2]], device=device)
        originals = {}
        for condition, processors in (("control", []), ("watermarked", [processor])):
            torch.manual_seed(100 + sample)
            with torch.inference_mode():
                ids = model.generate(prompt, max_new_tokens=200, do_sample=True,
                                     top_k=0, top_p=1, temperature=1,
                                     logits_processor=LogitsProcessorList(processors),
                                     pad_token_id=127)[0, 2:].tolist()
            originals[condition] = ids
            rows.append({"sample": sample + 1, "condition": condition,
                         **detect_ids(ids, scheme)})
            (output / f"{sample + 1:02d}_{condition}.json").write_text(json.dumps(ids))
        for fraction in (0.1, 0.3, 0.6):
            rng = np.random.default_rng(1000 + sample)
            ids = np.array(originals["watermarked"])
            selected = rng.choice(len(ids), int(len(ids) * fraction), replace=False)
            ids[selected] = rng.integers(0, 127, len(selected))
            rows.append({"sample": sample + 1,
                         "condition": f"synthetic_replace_{int(fraction * 100)}pct",
                         **detect_ids(ids.tolist(), scheme)})
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    summary = {condition: {
        "samples": 20,
        "detected": sum(r["detected"] for r in rows if r["condition"] == condition),
        "mean_z": float(np.mean([r["z_score"] for r in rows if r["condition"] == condition]))
    } for condition in conditions}
    summary["metadata"] = {"model": "random tiny GPT-2; untrained",
                           "device": device, "research_results": False,
                           "attacks": "synthetic token substitutions only"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 5))
    for sample in range(1, 21):
        values = [next(r["z_score"] for r in rows
                       if r["sample"] == sample and r["condition"] == condition)
                  for condition in conditions[1:]]
        ax.plot(range(4), values, marker="o", alpha=0.4)
    ax.axhline(4, color="black", linestyle="--", label="Threshold")
    ax.set_xticks(range(4), ["Watermarked", "Replace 10%", "Replace 30%", "Replace 60%"])
    ax.set_ylabel("Z-score")
    ax.set_title("OFFLINE SMOKE TEST: random model, synthetic substitutions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "synthetic_smoke_z_scores.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
