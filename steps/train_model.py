import numpy as np

import mlflow
import tensorflow as tf

from zenml.steps import step, Output
from zenml.integrations.tensorflow.materializers import KerasMaterializer

from model.cat_classifier import CatClassifier
from steps.config_loader import load_config
from steps.logging_config import get_logger

logger = get_logger(__name__)


@step(output_materializers=KerasMaterializer,
      experiment_tracker="mlflow_tracker")
def train_model(X_train: np.ndarray,
                y_train: np.ndarray) -> Output(model=tf.keras.Model):
    """Trains the cat classifier model, logs the run to MLFLow,
    and saves the trained model locally.

    Args:
        X_train (np.ndarray): Array of train images
        y_train (np.ndarray): Array of training labels

    Returns:
        (tf.keras.Model): Trained model
    """
    configs = load_config()

    mlflow.tensorflow.autolog()
    cat_classifier = CatClassifier(configs)

    logger.info("Starting training...")
    model = cat_classifier.train(X_train, y_train)

    logger.info("Saving model...")
    cat_classifier.save(model)
    logger.info("Done.")
    return model
