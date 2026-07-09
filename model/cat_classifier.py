import os
import datetime

import numpy as np
import tensorflow as tf

from model.architecture import build_cnn
from steps.logging_config import get_logger
from steps.seed import set_random_seeds

logger = get_logger(__name__)


class CatClassifier:
    """Trains and persists the cat-classifier CNN.

    The network topology itself lives in :func:`model.architecture.build_cnn`,
    which is shared with the hyperparameter-tuning search so the two can never
    drift apart.
    """

    def __init__(self, args: dict):
        self.args = args
        # Seed every RNG up front so a given config yields a reproducible model.
        if 'random_seed' in self.args:
            set_random_seeds(self.args['random_seed'])

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray) -> tf.keras.Model:
        """Train model.

        Heavily inspired by
        https://github.com/rpeden/cat-or-not/blob/master/train.py

        Args:
            X_train (np.ndarray): Array of train images
            y_train (np.ndarray): Array of training labels

        Returns:
            (tf.keras.Model): Trained model
        """
        logger.info("Constructing model...")
        cnn2d = build_cnn(self.args)

        logger.info("Fitting model...")
        cnn2d.fit(
            x=X_train,
            y=y_train,
            epochs=self.args['num_epochs'],
            batch_size=self.args['batch_size'],
            validation_split=self.args['val_split'],
        )
        logger.info("Model successfully fit.")
        return cnn2d

    def save(self, model: tf.keras.Model) -> str:
        """Save model.

        Args:
            model: Trained tf.keras.Model

        Returns:
            (str): Path the model was saved to.
        """
        time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(self.args['output_path'], exist_ok=True)
        out_path = os.path.join(self.args['output_path'], f"model-{time}.h5")

        tf.keras.models.save_model(
            model,
            filepath=out_path,
            save_format='h5',
        )
        logger.info(f"Model saved to {out_path}.")
        return out_path
