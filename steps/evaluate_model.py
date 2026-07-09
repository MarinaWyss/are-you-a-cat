import mlflow
import numpy as np
import tensorflow as tf

from zenml.steps import step, Output

from model.evaluator import Evaluation
from steps.config_loader import load_config
from steps.logging_config import get_logger

logger = get_logger(__name__)


@step(experiment_tracker="mlflow_tracker")
def evaluate_model(trained_model: tf.keras.Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> Output(
                                            precision=float,
                                            recall=float,
                                            f1=float):
    """Evaluates the model performance on the test set.
    Logs the model performance to MLFlow.

    Args:
        trained_model (tf.keras.Model): Trained tf.keras model
        X_test (np.ndarray): Array of test images
        y_test (np.ndarray): Array of test labels

    Returns:
        (float, float, float): precision, recall, f1 scores

    Raises:
        Exception if any of the metrics calculations fail
    """
    configs = load_config()

    logger.info("Beginning model evaluation...")

    try:
        logger.info("Predicting on the test set...")
        y_test = y_test[:, 0]
        evaluation = Evaluation()

        cutoff = configs['classification_cutoff']
        prediction = trained_model.predict(X_test)[:, 0]
        prediction = np.where(prediction > cutoff, 1, 0)

        logger.info(f"Calculating metrics with cut-off {cutoff}...")
        precision = evaluation.precision(y_test, prediction)
        mlflow.log_metric("test_precision", precision)

        recall = evaluation.recall(y_test, prediction)
        mlflow.log_metric("test_recall", recall)

        f1 = evaluation.f1(y_test, prediction)
        mlflow.log_metric("test_f1", f1)

        logger.info("Model evaluation done.")
        return precision, recall, f1

    except Exception as e:
        logger.error(e)
        raise e
