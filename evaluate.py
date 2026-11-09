"""Twenty-prompt paired experiment with checkpoints and real attack outputs."""
import argparse
import csv
from datetime import datetime
import gc
import json
from pathlib import Path
import secrets

from detect import detect_ids, detect_text, load_key
from watermark import SCHEME_VERSION, WatermarkScheme, tokenizer_fingerprint, vocabulary_ids


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def export_results(rows, output):
    if not rows:
        return
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for condition in sorted({r["condition"] for r in rows}):
        group = [r for r in rows if r["condition"] == condition]
        valid = [r["z_score"] for r in group if r["z_score"] is not None]
        summary[condition] = {
            "samples": len(group), "scorable_samples": len(valid),
            "detected": sum(r["detected"] for r in group),
            "detection_rate": sum(r["detected"] for r in group) / len(group),
            "mean_z": sum(valid) / len(valid) if valid else None,
            "mean_scored_tokens": sum(r["scored_tokens"] for r in group) / len(group),
        }
    write_json(output / "summary.json", summary)


def run(args):
    import torch
    import transformers
    import numpy as np
    from attacks import Translator, paraphrase_prompt
    from generate import Generator
    from plot_results import plot_results
    from watermark import make_processor

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts or any(
            not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Prompts file must contain a nonempty JSON list of strings")
    prompts = prompts[:args.limit]
    output = args.output / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    if args.key_file:
        key = load_key(args.key_file)
    else:
        key = secrets.token_bytes(32)
        args.key_file = output / "watermark.key"
        args.key_file.write_bytes(key)
    write_json(output / "status.json", {"status": "loading_model"})
    rows = []

    try:
        generator = Generator(args.model, args.revision)
        tokenizer = generator.tokenizer
        scheme = WatermarkScheme(vocabulary_ids(tokenizer), key, args.gamma)
        processor = make_processor(scheme, args.delta)
        deduplicate = not args.count_repeats
        manifest = {
            "scheme_version": SCHEME_VERSION, "model": args.model,
            "requested_revision": args.revision,
            "resolved_model_commit": getattr(generator.model.config, "_commit_hash", None),
            "vocabulary_sha256": tokenizer_fingerprint(tokenizer),
            "gamma_requested": args.gamma, "gamma_actual": scheme.gamma,
            "delta": args.delta, "threshold": args.threshold,
            "temperature": args.temperature, "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens, "seed": args.seed,
            "deduplicate": deduplicate, "device": generator.device,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "numpy_version": np.__version__, "prompts": prompts,
            "key_file": str(args.key_file.resolve()),
            "dataset_note": "Default prompts are manually specified topics, not C4/RealSumm.",
            "statistics_note": "Normal p-values are approximate; 20 controls cannot calibrate rare FPR.",
            "translation_models": ["Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"],
        }
        write_json(output / "manifest.json", manifest)

        def record(prompt_id, condition, text, ids=None):
            name = f"{prompt_id:02d}_{condition}"
            (output / f"{name}.txt").write_text(text, encoding="utf-8")
            stats = detect_text(text, tokenizer, scheme, args.threshold, deduplicate)
            row = {"prompt_id": prompt_id, "condition": condition,
                   "text_file": f"{name}.txt", "characters": len(text),
                   "retokenized_tokens": len(tokenizer.encode(text, add_special_tokens=False)),
                   **stats, "raw_ids_z_score": None}
            if ids is not None:
                write_json(output / f"{name}.token_ids.json", ids)
                row["raw_ids_z_score"] = detect_ids(ids, scheme, args.threshold,
                                                    deduplicate)["z_score"]
            rows.append(row)
            with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            export_results(rows, output)
            write_json(output / "status.json", {"status": "running", "scored_texts": len(rows)})
            print(f"{prompt_id:02d} {condition:18s} tokens={stats['scored_tokens']:4d} "
                  f"z={stats['z_score']}", flush=True)

        originals = {}
        for prompt_id, topic in enumerate(prompts, start=1):
            prompt = (topic + " Write a coherent passage of about 200 words in prose paragraphs. "
                      "Return only the passage.")
            seed = args.seed + prompt_id
            control, ids = generator.generate(prompt, seed, args)
            record(prompt_id, "control", control, ids)
            original, ids = generator.generate(prompt, seed, args, processor)
            record(prompt_id, "original", original, ids)
            originals[prompt_id] = original
            current = original
            for round_number in range(1, 4):
                current, _ = generator.generate(
                    paraphrase_prompt(current), args.seed + 1000 * round_number + prompt_id, args)
                record(prompt_id, f"paraphrase_{round_number}", current)

        device = generator.device
        del generator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        translator = Translator(manifest["translation_models"][0], device)
        manifest["translation_en_fr_commit"] = translator.commit
        french = {}
        for prompt_id, text in originals.items():
            french[prompt_id] = translator.translate(text)
            (output / f"{prompt_id:02d}_french.txt").write_text(french[prompt_id], encoding="utf-8")
        del translator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        translator = Translator(manifest["translation_models"][1], device)
        manifest["translation_fr_en_commit"] = translator.commit
        write_json(output / "manifest.json", manifest)
        for prompt_id, text in french.items():
            record(prompt_id, "back_translation", translator.translate(text))
        plot_results(rows, output)
        write_json(output / "status.json", {"status": "complete", "scored_texts": len(rows)})
        print(f"Results: {output.resolve()}")
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "scored_texts": len(rows),
                                            "error": str(error)})
        export_results(rows, output)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--prompts", type=Path, default=Path(__file__).with_name("prompts.json"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--delta", type=float, default=2)
    parser.add_argument("--threshold", type=float, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--count-repeats", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.max_new_tokens < 2 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("Invalid generation settings")
    run(args)
