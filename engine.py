"""Generation-2 engine: stateless hash partitions, context window h, span detection.

Scheme `hash32-h{h}-v2`: token t is green after context c iff
mix32(mix32(ctxhash(c) ^ scramble(t)) ^ k2) < gamma * 2**32.
No per-step vocabulary permutation, so the whole green mask is one vectorised
op on the GPU. The same integer functions run in numpy (detector) and torch
(processor), so agreement is exact. Not a proven-secure construction.
"""
import hashlib
import hmac
import math

import numpy as np

MASK = 0xFFFFFFFF
MUL = 0x45D9F3B
GOLD = 0x9E3779B1


def mix32(x):
    x = x & MASK
    x = x ^ (x >> 16)
    x = (x * MUL) & MASK
    x = x ^ (x >> 16)
    x = (x * MUL) & MASK
    return x ^ (x >> 16)


def context_hash(columns, k1, agg="chain"):
    """columns: iterable of int64 arrays/tensors (oldest first), each shape (...).

    agg="chain": every context token feeds the key (an edit anywhere in the window breaks it).
    agg="min":   the key is the context token with the smallest keyed hash (an edit breaks it
                 only if it removes or beats that token), the min-hash idea from the
                 KGW reliability study. Same cost; more edit-tolerant for h > 1."""
    columns = list(columns)
    if agg == "min":
        hashed = [mix32(c ^ k1) for c in columns]
        best = hashed[0]
        minimum = np.minimum if isinstance(best, np.ndarray) else __import__("torch").minimum
        for other in hashed[1:]:
            best = minimum(best, other)
        return best
    state = k1
    for column in columns:
        state = mix32(((state * 31) + column + 1) ^ k1)
    return state


def green_flags(state, tokens, k2, threshold):
    """state broadcastable against tokens; returns bool array of green membership."""
    return mix32(mix32(state ^ ((tokens * GOLD) & MASK)) ^ k2) < threshold


class HashScheme:
    def __init__(self, key, vocab_ids, gamma=0.25, context=1, agg="chain"):
        if len(key) < 16 or not 0 < gamma < 1 or context < 1 or agg not in ("chain", "min"):
            raise ValueError("Need key >= 16 bytes, 0 < gamma < 1, context >= 1, agg chain|min")
        digest = hmac.new(key, b"llm-watermark-v2\x00", hashlib.sha256).digest()
        self.k1 = int.from_bytes(digest[:4], "big")
        self.k2 = int.from_bytes(digest[4:8], "big")
        self.threshold = int(gamma * (1 << 32))
        self.gamma = self.threshold / (1 << 32)
        self.context = context
        self.agg = agg
        self.vocab = np.array(sorted(set(vocab_ids)), dtype=np.int64)
        self.version = f"hash32-h{context}{'-min' if agg == 'min' else ''}-v2"

    def eligible_mask(self, size):
        mask = np.zeros(size, dtype=bool)
        mask[self.vocab[self.vocab < size]] = True
        return mask

    def flags_for_ids(self, ids):
        """Per-position (is_scorable, is_green) for ids[context:]; numpy, vectorised."""
        ids = np.asarray(ids, dtype=np.int64)
        h = self.context
        if len(ids) <= h:
            return np.zeros(0, bool), np.zeros(0, bool), np.zeros(0, np.int64)
        columns = [ids[j:len(ids) - h + j] for j in range(h)]
        target = ids[h:]
        state = context_hash(columns, self.k1, self.agg)
        green = green_flags(state, target, self.k2, self.threshold)
        eligible = np.isin(target, self.vocab)
        for column in columns:
            eligible &= np.isin(column, self.vocab)
        keys = state * (1 << 20) + target  # dedupe key (collision-tolerant)
        return eligible, green, keys


def make_processor(scheme, delta=2.0):
    import torch
    from transformers import LogitsProcessor

    if not math.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")

    class HashProcessor(LogitsProcessor):
        def __init__(self):
            self._tokens = {}
            self._eligible = {}

        def _cached(self, scores):
            key = (scores.device, scores.shape[-1])
            if key not in self._tokens:
                size = scores.shape[-1]
                self._tokens[key] = torch.arange(size, device=scores.device, dtype=torch.long)
                self._eligible[key] = torch.from_numpy(
                    scheme.eligible_mask(size)).to(scores.device)
            return self._tokens[key], self._eligible[key]

        def __call__(self, input_ids, scores):
            if delta == 0 or input_ids.shape[1] < scheme.context:
                return scores
            tokens, eligible = self._cached(scores)
            window = input_ids[:, -scheme.context:].long()
            state = context_hash([window[:, j] for j in range(scheme.context)], scheme.k1, scheme.agg)
            green = green_flags(state[:, None], tokens[None, :], scheme.k2, scheme.threshold)
            return scores + delta * (green & eligible[None, :]).to(scores.dtype)

    return HashProcessor()


def detect(ids, scheme, threshold=4.0, deduplicate=True, window=None):
    """z-test on green transitions; optional max-over-windows span scan.

    With `window`, reports the best window's z and a Bonferroni-corrected p
    (conservative: overlapping windows are treated as independent tests)."""
    from scipy.stats import norm

    eligible, green, keys = scheme.flags_for_ids(ids)
    keep = eligible.copy()
    if deduplicate:
        _, first = np.unique(keys[eligible], return_index=True)
        mask = np.zeros(eligible.sum(), bool)
        mask[first] = True
        keep[np.flatnonzero(eligible)] = mask
    flags = green[keep]
    total, hits, gamma = len(flags), int(flags.sum()), scheme.gamma
    sd = lambda n: math.sqrt(n * gamma * (1 - gamma))
    z = (hits - gamma * total) / sd(total) if total else None
    result = {"scheme": scheme.version, "scored_tokens": total, "green_tokens": hits,
              "green_ratio": hits / total if total else None, "z_score": z,
              "p_value_approx": float(norm.sf(z)) if z is not None else None,
              "detected": bool(z is not None and z >= threshold), "threshold": threshold}
    if window and total >= window:
        sums = np.convolve(flags.astype(np.int64), np.ones(window, np.int64), "valid")
        zs = (sums - gamma * window) / sd(window)
        best = int(zs.argmax())
        p_adj = min(1.0, float(norm.sf(zs[best])) * len(zs))
        result.update(window=window, best_window_start=best, best_window_z=float(zs[best]),
                      best_window_p_corrected=p_adj,
                      detected=bool(result["detected"] or zs[best] >= threshold + 1.0))
    return result


def empirical_null(null_ids_list, scheme, deduplicate=True):
    """Calibrate on real unwatermarked text: returns z-scores and the 1 - alpha
    quantile you should use instead of the normal-tail threshold."""
    zs = [detect(ids, scheme, deduplicate=deduplicate)["z_score"] for ids in null_ids_list]
    zs = np.array([z for z in zs if z is not None])
    return {"n": len(zs), "mean": float(zs.mean()), "std": float(zs.std()),
            "q99": float(np.quantile(zs, 0.99)), "max": float(zs.max())}
