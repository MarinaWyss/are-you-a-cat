import yaml
import logging

import numpy as np
import tensorflow as tf

from zenml.steps import step

logging.basicConfig(level=logging.DEBUG)


@step
def evaluate_model(model: tf.keras.Model,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> float:
    """
    TODO actually set this up
    """
    with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    return 0.9
