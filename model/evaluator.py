from typing import Callable

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from steps.logging_config import get_logger

logger = get_logger(__name__)


class Evaluation:
    """Evaluates model performance on the test set using sklearn metrics.
    Heavily inspired by
    https://github.com/zenml-io/zenfiles/blob/main/customer-satisfaction/model/evaluation.py

    Each public metric previously repeated an identical try/except/log block.
    That boilerplate now lives once in :meth:`_score`.
    """

    @staticmethod
    def _score(name: str,
               metric_fn: Callable[[np.ndarray, np.ndarray], float],
               y_true: np.ndarray,
               y_pred: np.ndarray) -> float:
        """Runs a single sklearn metric with uniform logging and error handling.

        Args:
            name (str): Human-readable metric name, used in log messages.
            metric_fn (Callable): An sklearn metric function.
            y_true (np.ndarray): Ground-truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            (float): The computed metric.

        Raises:
            Exception: Re-raised if the underlying metric computation fails.
        """
        try:
            score = metric_fn(y_true, y_pred)
            logger.info(f"{name}: {score}.")
            return score
        except Exception as e:
            logger.error(
                f"Exception occurred in the {name} method of the Evaluation "
                f"class. Exception message: {e}"
            )
            raise

    def precision(self,
                  y_true: np.ndarray,
                  y_pred: np.ndarray) -> float:
        return self._score("Precision", precision_score, y_true, y_pred)

    def recall(self,
               y_true: np.ndarray,
               y_pred: np.ndarray) -> float:
        return self._score("Recall", recall_score, y_true, y_pred)

    def f1(self,
           y_true: np.ndarray,
           y_pred: np.ndarray) -> float:
        return self._score("F1 score", f1_score, y_true, y_pred)
