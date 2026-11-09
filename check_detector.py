"""Validate the standalone detector against saved synthetic integration samples.

The local tokenizer and key are fixtures for random-model smoke tests only.
No pretrained model or network access is required.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_directory", type=Path)
    args = parser.parse_args()
    output = args.results_directory.resolve()
    vocab = {f"tok_{i}": i for i in range(127)}
    vocab["[PAD]"] = 127
    backend = Tokenizer(WordLevel(vocab, unk_token="[PAD]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
                                       unk_token="[PAD]", pad_token="[PAD]")
    tokenizer_path = output / "fixture_tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    key_path = output / "fixture.key"
    key_path.write_bytes(b"offline-smoke-test-key")
    with (output / "results.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    checks = []
    for condition in ("control", "watermarked"):
        ids = json.loads((output / f"01_{condition}.json").read_text(encoding="utf-8"))
        text_path = output / f"01_{condition}.txt"
        text_path.write_text(" ".join(tokenizer.convert_ids_to_tokens(ids)), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("detect.py")), str(text_path),
             "--model", str(tokenizer_path), "--key-file", str(key_path)],
            check=True, capture_output=True, text=True,
        )
        observed = json.loads(completed.stdout)
        expected = next(r for r in rows if r["sample"] == "1" and r["condition"] == condition)
        if not math.isclose(observed["z_score"], float(expected["z_score"]), abs_tol=1e-12):
            raise AssertionError(f"Standalone z-score mismatch: {condition}")
        if observed["green_tokens"] != int(expected["green_tokens"]):
            raise AssertionError(f"Standalone green-token mismatch: {condition}")
        checks.append({"condition": condition, "passed": True, **observed})
    (output / "standalone_detector_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
