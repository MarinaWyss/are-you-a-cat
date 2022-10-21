import yaml
import logging
import numpy as np

import mlflow
import tensorflow as tf

from zenml.steps import step

from model.cat_classifier import CatClassifier

logging.basicConfig(level=logging.DEBUG)


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def train_model(X_train: np.ndarray,
                y_train: np.ndarray) -> tf.keras.Model:
    """Trains the cat classifier model, logs the run to MLFLow,
    and saves the trained model locally.

    Args:
        X_train (np.ndarray): Array of train images
        y_train (np.ndarray): Array of training labels

    Returns:
        (tf.Keras.model): Trained model
    """
    with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    mlflow.tensorflow.autolog()
    cat_classifier = CatClassifier(configs)

    logging.info("Starting training...")
    model = cat_classifier.train(X_train, y_train)

    logging.info("Model trained. Saving model...")
    cat_classifier.save(model)
    logging.info("Done.")
    return model
