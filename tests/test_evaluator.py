import unittest

import numpy as np

from model.evaluator import Evaluation


class TestEvaluation(unittest.TestCase):
    """Tests for the sklearn-backed Evaluation helper."""

    @classmethod
    def setUpClass(cls):
        cls.evaluation = Evaluation()
        # A simple, hand-checkable set of predictions.
        # TP=2, FP=1, FN=1  ->  precision=2/3, recall=2/3, f1=2/3
        cls.y_true = np.array([1, 1, 1, 0, 0])
        cls.y_pred = np.array([1, 1, 0, 1, 0])

    def test_precision(self):
        self.assertAlmostEqual(
            self.evaluation.precision(self.y_true, self.y_pred), 2 / 3)

    def test_recall(self):
        self.assertAlmostEqual(
            self.evaluation.recall(self.y_true, self.y_pred), 2 / 3)

    def test_f1(self):
        self.assertAlmostEqual(
            self.evaluation.f1(self.y_true, self.y_pred), 2 / 3)

    def test_perfect_prediction(self):
        labels = np.array([1, 0, 1, 0])
        self.assertEqual(self.evaluation.precision(labels, labels), 1.0)
        self.assertEqual(self.evaluation.recall(labels, labels), 1.0)
        self.assertEqual(self.evaluation.f1(labels, labels), 1.0)


if __name__ == '__main__':
    unittest.main()
