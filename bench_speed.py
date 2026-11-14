"""Clean generation-throughput comparison. Run it alone: generation is CPU-launch-bound, so any
other work on the machine inflates the numbers.

Arms are interleaved round by round (a different random order each round) so GPU clock drift
and background noise hit every arm equally. Reports the median tokens/s per arm and overhead
versus no watermark, with the spread across rounds."""
import argparse
import json
import random
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np

import bench_eval as be
import engine
from generate import Generator

ROOT, KEY = be.ROOT, be.KEY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompts", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--delta", type=float, default=2.0)
    args = ap.parse_args()

    out = ROOT / "results" / ("speed-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True)
    gen = Generator(args.model)
    vocab = be.v1.vocabulary_ids(gen.tokenizer)
    arms = {s["name"]: s["processor"] for s in be.build_schemes(vocab, args.delta)}
    arms = {"control (no watermark)": lambda: None, **arms}
    prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))[:args.prompts]
    settings = SimpleNamespace(temperature=0.8, top_p=0.95, max_new_tokens=args.tokens)
    for _ in range(2):  # warm up kernels and clocks
        gen.generate("Warm up the model.", 0, settings)

    rounds = {name: [] for name in arms}
    rng = random.Random(0)
    for r in range(args.rounds):
        order = list(arms)
        rng.shuffle(order)
        for name in order:
            tokens, seconds = 0, 0.0
            for i, prompt in enumerate(prompts):
                t0 = time.perf_counter()
                _, ids = gen.generate(prompt, 4000 + i, settings, arms[name]())
                seconds += time.perf_counter() - t0
                tokens += len(ids)
            rounds[name].append(tokens / seconds)
        print(f"round {r + 1}/{args.rounds}", {n: round(rounds[n][-1], 1) for n in order}, flush=True)

    base = float(np.median(rounds["control (no watermark)"]))
    result = {"kind": "speed", "model": args.model, "tokens": args.tokens, "prompts": args.prompts,
              "rounds": args.rounds, "created_at": datetime.now().isoformat(timespec="seconds"), "arms": {}}
    for name, values in rounds.items():
        med = float(np.median(values))
        result["arms"][name] = {"median_tps": med, "min_tps": float(min(values)), "max_tps": float(max(values)),
                                "overhead_pct": (base / med - 1) * 100, "rounds": [round(v, 2) for v in values]}
        print(f"{name:26s} median {med:6.1f} tok/s  range {min(values):.1f}-{max(values):.1f}  overhead {(base / med - 1) * 100:+.1f}%", flush=True)
    (out / "speed.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("Results:", out, flush=True)


if __name__ == "__main__":
    main()
