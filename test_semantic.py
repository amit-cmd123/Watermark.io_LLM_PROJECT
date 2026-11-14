import unittest

import numpy as np

import semantic

KEY = b"semantic-test-key-0123456789"


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class SemanticTests(unittest.TestCase):
    def test_segmentation_is_stable_and_merges_fragments(self):
        text = "1. Trees cool cities in summer. They also filter air. Short. And they hold soil in place!"
        parts = semantic.segment(text)
        self.assertTrue(all(len(p) >= semantic.MIN_CHARS for p in parts))
        self.assertEqual(semantic.segment(text), parts)
        self.assertGreaterEqual(len(parts), 2)

    def test_scheme_deterministic_and_key_sensitive(self):
        a, b = semantic.SemanticScheme(KEY), semantic.SemanticScheme(KEY)
        c = semantic.SemanticScheme(b"another-key-0123456789")
        x = unit(np.random.default_rng(0).standard_normal((50, 384)))
        self.assertTrue(np.array_equal(a.buckets(x), b.buckets(x)))
        self.assertFalse(np.array_equal(a.buckets(x), c.buckets(x)))
        self.assertAlmostEqual(a.gamma, 0.25)

    def test_fit_calibrates_the_null_on_skewed_embeddings(self):
        rng = np.random.default_rng(1)
        shared = rng.standard_normal(384) * 3          # real embeddings share a big common direction
        ref = unit(shared + rng.standard_normal((800, 384)))
        s = semantic.SemanticScheme(KEY).fit(ref)
        fresh = unit(shared + rng.standard_normal((400, 384)))
        r = semantic.detect_embeddings(fresh, s)
        self.assertLess(abs(r["z_score"]), 3.5)
        self.assertAlmostEqual(r["valid_ratio"], s.null_rate, delta=0.08)

    def test_planted_valid_sentences_are_detected(self):
        rng = np.random.default_rng(2)
        ref = unit(rng.standard_normal((800, 384)))
        s = semantic.SemanticScheme(KEY).fit(ref)
        pool = unit(rng.standard_normal((4000, 384)))
        planted = pool[s.is_valid(pool)][:30]
        self.assertGreater(semantic.detect_embeddings(planted, s)["z_score"], 6)
        self.assertIsNone(semantic.detect_embeddings(np.zeros((0, 384)), s)["z_score"])

    def test_margin_choice_resists_small_drift(self):
        rng = np.random.default_rng(3)
        ref = unit(rng.standard_normal((800, 384)))
        s = semantic.SemanticScheme(KEY).fit(ref)
        pool = unit(rng.standard_normal((6000, 384)))
        valid = pool[s.is_valid(pool)]
        high = valid[np.argsort(-s.margin(valid))[:200]]
        rand = valid[:200]
        # noise of norm ~0.5 on a unit vector: cosine ~0.9 to the original, like a close paraphrase
        drift = lambda v: unit(v + (0.5 / np.sqrt(v.shape[1])) * rng.standard_normal(v.shape))
        kept_high = s.is_valid(drift(high)).mean()
        kept_rand = s.is_valid(drift(rand)).mean()
        self.assertGreater(kept_high, kept_rand + 0.05)


if __name__ == "__main__":
    unittest.main()
