import importlib.util
import math
import unittest

import numpy as np
from detect import detect_ids
from watermark import WatermarkScheme, make_processor


class WatermarkTests(unittest.TestCase):
    def test_partition_agreement_and_key_sensitivity(self):
        a = WatermarkScheme(range(64), b"first-secret-key!")
        b = WatermarkScheme(range(64), b"first-secret-key!")
        c = WatermarkScheme(range(64), b"other-secret-key!")
        for context in range(10):
            self.assertEqual(len(a.green_ids(context)), 16)
            self.assertEqual(a.green_set(context), b.green_set(context))
        self.assertTrue(any(a.green_set(i) != c.green_set(i) for i in range(10)))

    def test_invalid_parameters(self):
        for gamma in (0, 1, -0.1):
            with self.assertRaises(ValueError):
                WatermarkScheme(range(8), b"key", gamma)
        with self.assertRaises(ValueError):
            WatermarkScheme(range(8), b"")

    def test_empty_and_special_boundaries(self):
        scheme = WatermarkScheme(range(8), b"key")
        for ids in ([], [1], [99, 1], [1, 99, 2]):
            self.assertIsNone(detect_ids(ids, scheme)["z_score"])

    def test_known_z_and_deduplication(self):
        class Partition:
            eligible = set(range(8))
            gamma = 0.5

            def green_set(self, previous):
                return {1, 2, 3, 4}

        scheme = Partition()
        result = detect_ids([0, 1, 2, 3, 4], scheme)
        self.assertEqual(result["green_tokens"], 4)
        self.assertEqual(result["scored_tokens"], 4)
        self.assertTrue(math.isclose(result["z_score"], 2))
        self.assertEqual(detect_ids([0, 1, 0, 1, 0, 1], scheme)["scored_tokens"], 2)
        self.assertEqual(detect_ids([0, 1, 0, 1, 0, 1], scheme,
                                    deduplicate=False)["scored_tokens"], 5)

    def test_noncontiguous_vocabulary(self):
        scheme = WatermarkScheme([2, 5, 11, 20], b"key", 0.5)
        self.assertTrue(scheme.green_set(5).issubset({2, 5, 11, 20}))
        self.assertEqual(scheme.gamma, 0.5)

    @unittest.skipUnless(importlib.util.find_spec("torch") and
                         importlib.util.find_spec("transformers"), "Needs torch/transformers")
    def test_batched_processor(self):
        import torch
        scheme = WatermarkScheme(range(16), b"key")
        inputs = torch.tensor([[1, 2], [4, 5]])
        scores = torch.zeros((2, 16))
        output = make_processor(scheme, 2)(inputs, scores)
        self.assertTrue(torch.equal(scores, torch.zeros_like(scores)))
        for row, previous in enumerate([2, 5]):
            expected = torch.zeros(16)
            expected[list(scheme.green_ids(previous))] = 2
            self.assertTrue(torch.equal(output[row], expected))
        self.assertTrue(torch.equal(make_processor(scheme, 0)(inputs, scores), scores))

    def test_synthetic_bias_is_detectable(self):
        scheme = WatermarkScheme(range(128), b"simulation-key!!")
        rng = np.random.Generator(np.random.PCG64(42))
        ids = [0]
        for _ in range(300):
            logits = np.zeros(128)
            logits[scheme.green_ids(ids[-1])] = 2
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum()
            ids.append(int(rng.choice(128, p=probabilities)))
        self.assertGreater(detect_ids(ids, scheme)["z_score"], 4)


if __name__ == "__main__":
    unittest.main()
