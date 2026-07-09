import unittest

from model.predictor import interpret_prediction, Verdict


class TestInterpretPrediction(unittest.TestCase):
    """Tests for turning raw probabilities into human-readable verdicts."""

    @classmethod
    def setUpClass(cls):
        cls.config = {'min_unsure': 0.5, 'max_unsure': 0.75}

    def test_confidently_not_a_cat(self):
        verdict = interpret_prediction(0.1, self.config)
        self.assertIsInstance(verdict, Verdict)
        self.assertFalse(verdict.is_cat)
        self.assertTrue(verdict.confident)

    def test_unsure_band(self):
        verdict = interpret_prediction(0.6, self.config)
        self.assertFalse(verdict.is_cat)
        self.assertFalse(verdict.confident)

    def test_confidently_a_cat(self):
        verdict = interpret_prediction(0.9, self.config)
        self.assertTrue(verdict.is_cat)
        self.assertTrue(verdict.confident)

    def test_boundary_is_inclusive_lower(self):
        # Exactly at min_unsure falls into the unsure band, not "not a cat".
        verdict = interpret_prediction(0.5, self.config)
        self.assertFalse(verdict.confident)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            interpret_prediction(1.5, self.config)
        with self.assertRaises(ValueError):
            interpret_prediction(-0.1, self.config)


if __name__ == '__main__':
    unittest.main()
