import importlib.util
import unittest

import numpy as np

import engine

KEY = b"engine-test-key-0123456789"


class EngineTests(unittest.TestCase):
    def test_deterministic_and_key_sensitive(self):
        a = engine.HashScheme(KEY, range(200), context=2)
        b = engine.HashScheme(KEY, range(200), context=2)
        c = engine.HashScheme(b"another-key-0123456789", range(200), context=2)
        ids = list(np.random.default_rng(0).integers(0, 200, 300))
        self.assertEqual(engine.detect(ids, a), engine.detect(ids, b))
        self.assertNotEqual(engine.detect(ids, a)["green_tokens"],
                            engine.detect(ids, c)["green_tokens"])

    def test_green_rate_matches_gamma_on_random_text(self):
        scheme = engine.HashScheme(KEY, range(5000), gamma=0.25, context=3)
        ids = np.random.default_rng(1).integers(0, 5000, 20000)
        r = engine.detect(ids, scheme, deduplicate=False)
        self.assertAlmostEqual(r["green_ratio"], scheme.gamma, delta=0.02)
        self.assertLess(abs(r["z_score"]), 4)

    def test_short_input_unscorable(self):
        scheme = engine.HashScheme(KEY, range(50), context=3)
        self.assertIsNone(engine.detect([1, 2, 3], scheme)["z_score"])

    def test_window_scan_finds_embedded_span(self):
        scheme = engine.HashScheme(KEY, range(300), gamma=0.25, context=1)
        rng = np.random.default_rng(2)
        ids = [int(rng.integers(300))]
        for i in range(600):
            if 200 <= i < 400:  # watermarked middle third
                candidates = np.arange(300)
                st = engine.context_hash([np.array([ids[-1]])], scheme.k1)
                green = engine.green_flags(st, candidates, scheme.k2, scheme.threshold)
                ids.append(int(rng.choice(candidates[green])))
            else:
                ids.append(int(rng.integers(300)))
        r = engine.detect(ids, scheme, window=100)
        self.assertGreater(r["best_window_z"], 6)
        self.assertTrue(150 <= r["best_window_start"] <= 300)

    def test_min_hash_survives_edit_outside_the_minimum(self):
        k1 = 12345
        window = [np.array([10]), np.array([20]), np.array([30]), np.array([40])]
        base = int(engine.context_hash(window, k1, "min")[0])
        hashes = [int(engine.mix32(c ^ k1)[0]) for c in window]
        keep = int(np.argmin(hashes))
        replacement = np.array([999])
        self.assertGreater(int(engine.mix32(replacement ^ k1)[0]), hashes[keep])
        for i in range(4):
            if i == keep:
                continue
            edited = [c.copy() for c in window]
            edited[i] = replacement
            self.assertEqual(int(engine.context_hash(edited, k1, "min")[0]), base)
        edited = [c.copy() for c in window]
        edited[keep] = replacement
        self.assertNotEqual(int(engine.context_hash(edited, k1, "min")[0]), base)
        chain = int(engine.context_hash(window, k1, "chain")[0])
        self.assertNotEqual(chain, int(engine.context_hash(window[:3] + [replacement], k1, "chain")[0]))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Needs torch")
    def test_min_agg_torch_numpy_agreement(self):
        import torch
        scheme = engine.HashScheme(KEY, range(1000), context=4, agg="min")
        out = engine.make_processor(scheme, 3.0)(torch.tensor([[5, 9, 17, 3, 8]]), torch.zeros(1, 1024))
        ctx = np.array([9, 17, 3, 8])
        state = engine.context_hash([ctx[j:j + 1] for j in range(4)], scheme.k1, "min")
        expected = engine.green_flags(state, np.arange(1024), scheme.k2, scheme.threshold)
        expected &= scheme.eligible_mask(1024)
        self.assertTrue(np.array_equal(out[0].numpy() == 3.0, expected))

    @unittest.skipUnless(importlib.util.find_spec("torch") and
                         importlib.util.find_spec("transformers"), "Needs torch")
    def test_torch_numpy_agreement_and_bias(self):
        import torch
        scheme = engine.HashScheme(KEY, range(1000), gamma=0.25, context=2)
        proc = engine.make_processor(scheme, 3.0)
        inputs = torch.tensor([[5, 9, 17], [400, 2, 999]])
        out = proc(inputs, torch.zeros(2, 1024))
        for row in range(2):
            ctx = inputs[row, -2:].numpy()
            st = engine.context_hash([ctx[:1], ctx[1:]], scheme.k1)
            expected = engine.green_flags(st, np.arange(1024), scheme.k2, scheme.threshold)
            expected &= scheme.eligible_mask(1024)
            self.assertTrue(np.array_equal(out[row].numpy() == 3.0, expected))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Needs torch")
    def test_generated_text_detected(self):
        import torch
        scheme = engine.HashScheme(KEY, range(512), gamma=0.25, context=2)
        proc = engine.make_processor(scheme, 2.5)
        g = torch.Generator().manual_seed(3)
        ids = [1, 2]
        for _ in range(300):
            scores = proc(torch.tensor([ids]), torch.zeros(1, 512))
            ids.append(int(torch.multinomial(torch.softmax(scores[0], -1), 1, generator=g)))
        self.assertGreater(engine.detect(ids[2:], scheme)["z_score"], 6)


if __name__ == "__main__":
    unittest.main()
