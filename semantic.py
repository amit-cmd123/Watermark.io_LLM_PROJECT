"""G4: sentence-level semantic watermark (k-SemStamp family, margin-aware).

Each sentence is embedded and keyed to a region of embedding space: `bits` key-derived
hyperplanes cut the (centred) space into 2**bits buckets and a key-derived subset of buckets
is "valid". Generation draws several candidate sentences and keeps a valid one, preferring
the candidate farthest from every hyperplane so a paraphrase's embedding drift is less
likely to flip its bucket. Detection embeds the text's sentences and z-tests the valid share.

Not a proven-secure construction; the partition is learnable from enough output.
"""
import hashlib
import hmac
import math
import re

import numpy as np

MIN_CHARS = 20
_PIECE = re.compile(r"\S.*?(?:[.!?](?=\s|$)|$)", re.S)


def units(text, min_chars=MIN_CHARS):
    """(start, end) spans of sentence units: sentences merged until >= min_chars.
    Generation and detection share this, so their unit boundaries agree."""
    spans, start = [], None
    for m in _PIECE.finditer(text):
        if start is None:
            start = m.start()
        if len(text[start:m.end()].strip()) >= min_chars and re.search(r"[.!?]$", m.group()):
            spans.append((start, m.end()))
            start = None
    if start is not None and spans and len(text[start:].strip()) < min_chars:
        pass  # trailing fragment: dropped, not scored
    elif start is not None:
        tail = text[start:].strip()
        if len(tail) >= min_chars:
            spans.append((start, len(text.rstrip())))
    return spans


def segment(text, min_chars=MIN_CHARS):
    return [text[a:b].strip() for a, b in units(text, min_chars)]


class SemanticScheme:
    def __init__(self, key, dim=384, bits=3, gamma=0.25):
        if len(key) < 16 or not 0 < gamma < 1 or not 1 <= bits <= 8:
            raise ValueError("Need key >= 16 bytes, 0 < gamma < 1, 1 <= bits <= 8")
        seed = int.from_bytes(hmac.new(key, b"llm-watermark-g4\x00planes", hashlib.sha256).digest()[:16], "big")
        rng = np.random.Generator(np.random.PCG64(seed))
        planes = rng.standard_normal((bits, dim)).astype(np.float32)
        self.planes = planes / np.linalg.norm(planes, axis=1, keepdims=True)
        self.bits, self.dim = bits, dim
        buckets = 2 ** bits
        count = max(1, min(buckets - 1, round(gamma * buckets)))
        self.valid = np.sort(rng.permutation(buckets)[:count])
        self.gamma = count / buckets      # nominal; `fit` replaces the null rate with a measured one
        self.null_rate = self.gamma
        self.center = np.zeros(dim, np.float32)
        self.version = f"semantic-lsh{bits}-v1"

    def fit(self, reference):
        """Centre on reference sentence embeddings and measure the real valid rate (the null)."""
        reference = np.asarray(reference, np.float32)
        self.center = reference.mean(0)
        rate = float(self.is_valid(reference).mean())
        self.null_rate = min(0.95, max(0.02, rate))
        return self

    def project(self, embeddings):
        e = np.asarray(embeddings, np.float32) - self.center
        e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
        return e @ self.planes.T

    def buckets(self, embeddings):
        return ((self.project(embeddings) > 0).astype(np.int64) << np.arange(self.bits)).sum(1)

    def is_valid(self, embeddings):
        return np.isin(self.buckets(embeddings), self.valid)

    def margin(self, embeddings):
        return np.abs(self.project(embeddings)).min(1)


def detect_embeddings(embeddings, scheme, threshold=4.0):
    n = len(embeddings)
    if n == 0:
        return {"scheme": scheme.version, "sentences": 0, "valid": 0, "valid_ratio": None, "z_score": None,
                "p_value_approx": None, "detected": False, "threshold": threshold}
    from scipy.stats import norm
    hits, g = int(scheme.is_valid(embeddings).sum()), scheme.null_rate
    z = (hits - g * n) / math.sqrt(n * g * (1 - g))
    return {"scheme": scheme.version, "sentences": n, "valid": hits, "valid_ratio": hits / n, "z_score": z,
            "p_value_approx": float(norm.sf(z)), "detected": bool(z >= threshold), "threshold": threshold}


class Embedder:
    """all-MiniLM-L6-v2 mean-pooled sentence vectors (local files only)."""

    def __init__(self, device="cpu", name="sentence-transformers/all-MiniLM-L6-v2"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.model = AutoModel.from_pretrained(name, local_files_only=True).to(device).eval()

    def embed(self, texts, batch=64):
        torch, out = self.torch, []
        for i in range(0, len(texts), batch):
            b = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=128,
                         return_tensors="pt").to(self.device)
            with torch.inference_mode():
                h = self.model(**b).last_hidden_state
            e = (h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
            out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


def detect_text(text, embedder, scheme, threshold=4.0):
    return detect_embeddings(embedder.embed(segment(text)), scheme, threshold)


def generate(gen, embedder, scheme, prompt, seed, settings, candidates=8, max_units=14, unit_tokens=56):
    """Sentence-by-sentence rejection sampling with the chat model in `gen` (a generate.Generator).
    Returns (text, info) where info counts units, how many landed valid, and tokens produced."""
    torch = gen.torch if hasattr(gen, "torch") else __import__("torch")
    tok, model = gen.tokenizer, gen.model
    base = (tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                    add_generation_prompt=True) if tok.chat_template else prompt)
    text, valid_units, tokens = "", 0, 0
    for step in range(max_units):
        inputs = tok(base + text, return_tensors="pt", add_special_tokens=False).to(gen.device)
        torch.manual_seed(seed * 1000 + step)
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=True, temperature=settings.temperature,
                                 top_p=settings.top_p, top_k=0, max_new_tokens=unit_tokens,
                                 num_return_sequences=candidates, pad_token_id=tok.pad_token_id)
        new = out[:, inputs["input_ids"].shape[1]:]
        raws = tok.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        ended = [bool((row == tok.eos_token_id).any()) for row in new]
        cuts = []
        for raw in raws:
            sp = units(raw)
            cuts.append(raw[:sp[0][1]] if sp else raw)
        keep = [i for i, c in enumerate(cuts) if c.strip()]
        if not keep:
            break
        E = embedder.embed([cuts[i].strip() for i in keep])
        ok, margin = scheme.is_valid(E), scheme.margin(E)
        pool = [j for j in range(len(keep)) if ok[j]]
        pick = max(pool, key=lambda j: margin[j]) if pool else int(np.argmax(margin))
        valid_units += int(bool(ok[pick]))
        i = keep[pick]
        text += cuts[i]
        tokens += len(tok(cuts[i], add_special_tokens=False)["input_ids"])
        if ended[i] and cuts[i].strip() == raws[i].strip():
            break
    return text, {"units": step + 1, "valid_units": valid_units, "tokens": tokens}
