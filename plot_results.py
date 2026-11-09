"""Plot real measurements saved by evaluate.py; no fabricated values."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(rows, output):
    conditions = ["control", "original", "paraphrase_1", "paraphrase_2",
                  "paraphrase_3", "back_translation"]
    lookup = {(r["prompt_id"], r["condition"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(9, 5))
    for prompt_id in sorted({r["prompt_id"] for r in rows}):
        values = [lookup.get((prompt_id, c), {}).get("z_score") for c in conditions[1:5]]
        ax.plot(range(4), [np.nan if z is None else z for z in values],
                marker="o", alpha=0.4, linewidth=1)
    ax.axhline(rows[0]["threshold"], linestyle="--", color="black", label="Threshold")
    ax.set_xticks(range(4), ["Original", "Paraphrase 1", "Paraphrase 2", "Paraphrase 3"])
    ax.set_ylabel("Z-score")
    ax.set_title("Watermark signal after recursive paraphrasing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "paraphrase_z_scores.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for position, condition in enumerate(conditions):
        values = [r["z_score"] for r in rows
                  if r["condition"] == condition and r["z_score"] is not None]
        if values:
            ax.boxplot(values, positions=[position], widths=0.5)
            ax.scatter(np.full(len(values), position), values, s=15, alpha=0.55)
    ax.axhline(rows[0]["threshold"], linestyle="--", color="black")
    ax.set_xticks(range(6), conditions, rotation=25)
    ax.set_ylabel("Z-score")
    ax.set_title("Detection scores by condition")
    fig.tight_layout()
    fig.savefig(output / "condition_z_scores.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_directory", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in
            (args.results_directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    if not rows:
        parser.error("No scored samples")
    plot_results(rows, args.results_directory)
