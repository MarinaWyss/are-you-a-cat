import logging
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

logging.basicConfig(level=logging.DEBUG)


class Evaluation:
    """Evaluates model performance on the test set using sklearn metrics.
    Heavily inspired by
    https://github.com/zenml-io/zenfiles/blob/main/customer-satisfaction/model/evaluation.py
    """
    def __init__(self) -> None:
        pass

    def precision(self,
                  y_true: np.ndarray,
                  y_pred: np.ndarray) -> float:
        try:
            prec_score = precision_score(y_true, y_pred)
            logging.info(f"Precision: {str(prec_score)}.")
            return prec_score
        except Exception as e:
            logging.info(
                "Exception occurred in precision method of the Evaluation class. Exception message:  "
                + str(e),
            )
            logging.info(
                "Exiting the precision method of the Evaluation class",
            )
            raise Exception()

    def recall(self,
               y_true: np.ndarray,
               y_pred: np.ndarray) -> float:
        try:
            rec_score = recall_score(y_true, y_pred)
            logging.info(f"Recall: {str(rec_score)}.")
            return rec_score
        except Exception as e:
            logging.info(
                "Exception occurred in recall method of the Evaluation class. Exception message:  "
                + str(e),
            )
            logging.info(
                "Exiting the recall method of the Evaluation class",
            )
            raise Exception()

    def f1(self,
           y_true: np.ndarray,
           y_pred: np.ndarray) -> float:
        try:
            f1 = f1_score(y_true, y_pred)
            logging.info(f"F1 score: {str(f1)}.")
            return f1
        except Exception as e:
            logging.info(
                "Exception occurred in F1 method of the Evaluation class. Exception message:  "
                + str(e),
            )
            logging.info(
                "Exiting the F1 method of the Evaluation class",
            )
            raise Exception()
