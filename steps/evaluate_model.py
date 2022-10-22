import logging

import mlflow
import numpy as np
import tensorflow as tf

from zenml.steps import step, Output

from model.evaluator import Evaluation

logging.basicConfig(level=logging.DEBUG)


@step(experiment_tracker="mlflow_tracker")
def evaluate_model(model: tf.keras.Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> Output(
                                            precision=float,
                                            recall=float,
                                            f1=float):
    """Evaluates the model performance on the test set.
    Logs the model performance to MLFlow.

    Args:
        model (tf.keras.Model): Trained tf.keras model
        X_test (np.ndarray): Array of test images
        y_test (np.ndarray): Array of test labels

    Returns:
        (float, float, float): precision, recall, f1 scores

    Raises:
        Exception if any of the metrics calculations fail
    """
    logging.info("Beginning model evaluation...")

    try:
        logging.info("Predicting on the test set...")
        prediction = model.predict(X_test)
        evaluation = Evaluation()

        precision = evaluation.precision(y_test, prediction)
        mlflow.log_metric("test_precision", precision)

        recall = evaluation.recall(y_test, prediction)
        mlflow.log_metric("test_recall", recall)

        f1 = evaluation.f1(y_test, prediction)
        mlflow.log_metric("test_f1", f1)

        logging.info("Model evaluation done.")
        return precision, recall, f1

    except Exception as e:
        logging.error(e)
        raise e
