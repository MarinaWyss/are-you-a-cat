import yaml
import numpy as np
import logging

import mlflow
from zenml.steps import step
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow

from model.cat_classifier import CatClassifier


@enable_mlflow
@step
def train_model(X_train: np.array,
                y_train: np.array):
    """Trains the cat classifier model, logs the run to MLFLow,
    and saves the trained model.

    Args:
        X_train (np.array): Array of train images
        y_train (np.array): Array of training labels

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
    logging.info("Model saved.")
