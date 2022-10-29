import yaml
import logging
import numpy as np

from zenml.steps import step, Output

from steps.utils import load_data, format_data_for_model

logging.basicConfig(level=logging.DEBUG)


@step
def import_data() -> Output(
    X_train=np.ndarray,
    y_train=np.ndarray,
    X_test=np.ndarray,
    y_test=np.ndarray,
):
    """Loads the data and formats it for the model."""
    with open('steps/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    train_data = load_data(train=True, configs=configs)
    test_data = load_data(train=False, configs=configs)

    # TODO do something with image paths
    X_train, y_train, train_paths = format_data_for_model(
        dat_list=train_data, configs=configs
    )

    X_test, y_test, test_paths = format_data_for_model(
        dat_list=test_data, configs=configs
    )

    return X_train, y_train, X_test, y_test


@step()
def dynamic_importer() -> Output(data=np.ndarray):
    """Downloads the latest data from a mock API."""
    with open('steps/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
    test_data = load_data(train=False, configs=configs)
    X_test, y_test, test_paths = format_data_for_model(
        dat_list=test_data, configs=configs
    )
    return X_test
