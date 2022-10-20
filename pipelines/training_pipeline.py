import yaml
import logging

from zenml.pipelines import pipeline
from zenml.integrations.constants import MLFLOW

logging.basicConfig(level=logging.DEBUG)


@pipeline(enable_cache=False, required_integrations=[MLFLOW])
def train_pipeline(prepare_data, train_model, evaluate_model):
    """
    Args:
        prepare_data: DataClass
        train_model: DataClass
        evaluate_model: DataClass

    Returns:
        (placeholder) Output from eval
    """
    with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    logging.info("Preparing train data...")
    X_train, y_train, _ = prepare_data(train=True, configs=configs)
    logging.info("Preparing test data...")
    X_test, y_test, _ = prepare_data(train=False, configs=configs)

    logging.info("Beginning model training pipeline...")
    model = train_model(X_train, y_train, configs=configs)

    # TODO cleanup
    logging.info("Evaluating model performance...")
    placeholder = evaluate_model(model, X_test, y_test, configs=configs)
    return placeholder
