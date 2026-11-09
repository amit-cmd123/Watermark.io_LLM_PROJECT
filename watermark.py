"""HMAC-seeded green-list research baseline; PCG64 is not a crypto PRNG."""
import hashlib
import hmac
import math
import struct
from functools import lru_cache

import numpy as np

SCHEME_VERSION = "hmac-pcg64-v1"


class WatermarkScheme:
    def __init__(self, vocab_ids, key, gamma=0.25):
        self.vocab = np.array(sorted(set(vocab_ids)), dtype=np.int64)
        if not key or not 0 < gamma < 1:
            raise ValueError("Require a nonempty key and 0 < gamma < 1")
        self.key = key
        self.green_size = math.floor(gamma * len(self.vocab))
        if not 0 < self.green_size < len(self.vocab):
            raise ValueError("Green partition must be nonempty and smaller than vocabulary")
        self.gamma = self.green_size / len(self.vocab)
        self.eligible = set(self.vocab.tolist())

    @lru_cache(maxsize=128)
    def green_ids(self, previous_token):
        message = b"llm-watermark-v1\x00" + struct.pack(">Q", int(previous_token))
        digest = hmac.new(self.key, message, hashlib.sha256).digest()
        seed = int.from_bytes(digest[:16], "big")
        rng = np.random.Generator(np.random.PCG64(seed))
        ids = rng.permutation(self.vocab)[:self.green_size]
        ids.flags.writeable = False
        return ids

    @lru_cache(maxsize=128)
    def green_set(self, previous_token):
        return frozenset(self.green_ids(previous_token).tolist())


def make_processor(scheme, delta=2.0):
    import torch
    from transformers import LogitsProcessor

    if not math.isfinite(delta) or delta < 0:
        raise ValueError("delta must be finite and nonnegative")

    class WatermarkProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if delta == 0:
                return scores
            modified = scores.clone()
            for row in range(input_ids.shape[0]):
                ids = scheme.green_ids(int(input_ids[row, -1].item()))
                indices = torch.tensor(ids.copy(), device=scores.device, dtype=torch.long)
                modified[row, indices] += delta
            return modified

    return WatermarkProcessor()


def vocabulary_ids(tokenizer):
    return sorted(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids))


def tokenizer_fingerprint(tokenizer):
    import json
    payload = json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
