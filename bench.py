"""Processor overhead: v1 (per-context permutation) vs v2 (stateless GPU hash)."""
import time
import numpy as np
import torch
import engine
import watermark

V, B, STEPS = 128256, 1, 200
key = b"benchmark-key-0123456789"
dev = "cuda" if torch.cuda.is_available() else "cpu"
v1 = watermark.make_processor(watermark.WatermarkScheme(range(V), key), 2.0)
procs = {"v1 permutation": v1}
for h in (1, 4):
    procs[f"v2 hash h={h}"] = engine.make_processor(engine.HashScheme(key, range(V), context=h), 2.0)
rng = np.random.default_rng(0)
for name, proc in procs.items():
    ids = torch.tensor(rng.integers(0, V, (B, 8)), device=dev)
    scores = torch.randn(B, V, device=dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(STEPS):
        ids = torch.cat([ids, torch.tensor(rng.integers(0, V, (B, 1)), device=dev)], 1)  # fresh contexts: worst case for v1
        proc(ids, scores)
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"{name:16s} {(time.perf_counter()-t)/STEPS*1000:8.3f} ms/step")
