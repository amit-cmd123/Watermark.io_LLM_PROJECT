"""Independent detector. Only CLI text detection loads a tokenizer, never weights."""
import argparse
import json
import math
import os
from pathlib import Path

from scipy.stats import norm
from watermark import WatermarkScheme, vocabulary_ids


def detect_ids(ids, scheme, threshold=4.0, deduplicate=True):
    green = total = 0
    seen = set()
    for previous, current in zip(ids, ids[1:]):
        pair = (int(previous), int(current))
        if pair[0] not in scheme.eligible or pair[1] not in scheme.eligible:
            continue
        if deduplicate and pair in seen:
            continue
        seen.add(pair)
        total += 1
        green += pair[1] in scheme.green_set(pair[0])
    z = ((green - scheme.gamma * total) /
         math.sqrt(total * scheme.gamma * (1 - scheme.gamma))) if total else None
    return {"scored_tokens": total, "green_tokens": green,
            "green_ratio": green / total if total else None,
            "z_score": z, "p_value_approx": float(norm.sf(z)) if z is not None else None,
            "detected": bool(z is not None and z >= threshold),
            "threshold": threshold, "deduplicate": deduplicate}


def detect_text(text, tokenizer, scheme, threshold=4.0, deduplicate=True):
    return detect_ids(tokenizer.encode(text, add_special_tokens=False),
                      scheme, threshold, deduplicate)


def load_key(path):
    if path is not None:
        key = Path(path).read_bytes()
    else:
        key = os.environ.get("WATERMARK_KEY", "").encode()
    if len(key) < 16:
        raise ValueError("Provide --key-file or WATERMARK_KEY with at least 16 bytes")
    return key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text_file", type=Path)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--threshold", type=float, default=4)
    parser.add_argument("--count-repeats", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    scheme = WatermarkScheme(vocabulary_ids(tokenizer), load_key(args.key_file), args.gamma)
    print(json.dumps(detect_text(args.text_file.read_text(encoding="utf-8"), tokenizer,
                                scheme, args.threshold, not args.count_repeats), indent=2))


if __name__ == "__main__":
    main()
