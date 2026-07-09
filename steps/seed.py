import os
import random

import numpy as np

from steps.logging_config import get_logger

logger = get_logger(__name__)


def set_random_seeds(seed: int) -> None:
    """Seeds every source of randomness used during training for reproducibility.

    This addresses the long-standing ``# TODO random seed`` in the model code.
    TensorFlow is imported lazily so that lightweight callers (tests, the
    Streamlit app) do not pay the import cost unless they actually train.

    Args:
        seed (int): The seed to apply to Python, NumPy, and TensorFlow.
    """
    logger.info(f"Setting global random seed to {seed}.")

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        logger.warning(
            "TensorFlow is not installed; skipping TF seed. "
            "Training reproducibility is not guaranteed."
        )
