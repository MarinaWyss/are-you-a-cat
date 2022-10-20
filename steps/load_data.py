import logging
import numpy as np

from zenml.steps import step

from steps.utils import load_data, format_data_for_model

logging.basicConfig(level=logging.DEBUG)


@step
def prepare_data(train: bool,
                 configs: dict) -> (np.array, np.array, np.array):
    """Loads the data and formats it for the model.

    Args:
        train (bool): If True, grab training data. Else, test data.
        configs (dict): Config file

    Returns:
        (np.array, np.array, np.array): Images, labels, and image paths.
    """
    data = load_data(train=train, configs=configs)

    logging.info('Formatting data.')
    X_train, y_train, train_paths = format_data_for_model(
        dat_list=data, configs=configs
    )
    return X_train, y_train, train_paths
