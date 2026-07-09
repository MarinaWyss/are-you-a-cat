"""Shared CNN architecture builder.

This module resolves the ``# TODO set this up so that this isn't copy-pasta
with cat_classifier.py`` that lived in ``hyperparameter_tuning.py``. Both the
fixed-config training path and the Keras Tuner search path now construct their
network here, so the topology can only ever be defined in one place.
"""
from typing import Optional

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall

from steps.logging_config import get_logger

logger = get_logger(__name__)

# Number of convolutional blocks in the network. Each block is a
# Conv2D -> MaxPooling2D -> BatchNormalization stack.
NUM_CONV_BLOCKS = 5


def _conv_filters(config: dict, hp, block: int) -> int:
    """Resolves the filter count for a conv block, from ``hp`` if tuning."""
    if hp is not None:
        return hp.Int(
            f'conv_{block}_filter', min_value=32, max_value=512, step=16
        )
    return config[f'conv_{block}_filters']


def _kernel_size(config: dict, hp, block: int) -> int:
    """Resolves the kernel size for a conv block, from ``hp`` if tuning."""
    if hp is not None:
        return hp.Choice(f'conv_{block}_kernel', values=[2, 3, 5, 7])
    return config['kernel_size']


def _pool_size(config: dict, hp, block: int) -> int:
    """Resolves the pool size for a conv block, from ``hp`` if tuning."""
    if hp is not None:
        return hp.Choice(f'max_pool_{block}', values=[1, 2, 3])
    return config['max_pool']


def _dropout(config: dict, hp, index: int) -> float:
    """Resolves a dropout rate, from ``hp`` if tuning."""
    if hp is not None:
        return hp.Float(f'dropout_{index}', min_value=0.0, max_value=0.5)
    return config['dropout']


def _dense_units(config: dict, hp, index: int, max_value: int) -> int:
    """Resolves a dense-layer unit count, from ``hp`` if tuning."""
    if hp is not None:
        return hp.Int(
            f'dense_{index}_units', min_value=32, max_value=max_value, step=16
        )
    return config[f'dense_{index}_units']


def _learning_rate(config: dict, hp) -> float:
    """Resolves the optimizer learning rate, from ``hp`` if tuning."""
    if hp is not None:
        return hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    return config['learning_rate']


def build_cnn(config: dict, hp: Optional[object] = None) -> Sequential:
    """Builds and compiles the cat-classifier CNN.

    Args:
        config (dict): Project config providing default hyperparameters and
            the input ``image_size``.
        hp: Optional Keras Tuner ``HyperParameters`` object. When provided,
            tunable values are sampled from the search space instead of read
            from ``config``.

    Returns:
        (tf.keras.Sequential): A compiled Keras model.
    """
    logger.info("Constructing CNN architecture...")
    image_size = config['image_size']

    model = Sequential()
    for block in range(1, NUM_CONV_BLOCKS + 1):
        conv_kwargs = dict(
            filters=_conv_filters(config, hp, block),
            kernel_size=(_kernel_size(config, hp, block),) * 2,
            activation='relu',
        )
        # Only the first conv layer needs an explicit input shape.
        if block == 1:
            conv_kwargs['input_shape'] = (image_size, image_size, 1)

        model.add(Conv2D(**conv_kwargs))
        pool = _pool_size(config, hp, block)
        model.add(MaxPooling2D(pool_size=(pool, pool)))
        model.add(BatchNormalization())

    model.add(Dropout(_dropout(config, hp, index=1)))
    model.add(Flatten())
    model.add(
        Dense(units=_dense_units(config, hp, index=1, max_value=256),
              activation='relu')
    )
    model.add(Dropout(_dropout(config, hp, index=2)))
    model.add(
        Dense(units=_dense_units(config, hp, index=2, max_value=128),
              activation='relu')
    )
    model.add(Dense(units=config['dense_3_units'], activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=_learning_rate(config, hp)),
        metrics=[AUC(), Precision(), Recall()],
    )
    logger.info("Model compiled.")
    return model
