import yaml
import logging

import numpy as np

from zenml.steps import step

from utils import load_data, format_data_for_model


@step
def prepare_data(train: bool) -> (np.array, np.array, np.array):
    """Loads the data and formats it for the model.

    Args:
        train (bool): If True, grab training data. Else, test data.

    Returns:
        (np.array, np.array, np.array): Images, labels, and image paths.
    """
    with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    data = load_data(train=train, configs=configs)

    logging.info('Formatting data.')
    X_train, y_train, train_paths = format_data_for_model(
        dat_list=data, configs=configs
    )
    return X_train, y_train, train_paths
