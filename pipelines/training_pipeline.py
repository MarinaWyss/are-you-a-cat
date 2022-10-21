import logging

from zenml.pipelines import pipeline

logging.basicConfig(level=logging.DEBUG)


@pipeline
def train_pipeline(import_data, train_model, evaluate_model):
    """
    Args:
        import_data: DataClass
        train_model: DataClass
        evaluate_model: DataClass

    Returns:
        (placeholder) Output from eval
    """
    logging.info("Loading and preparing data")
    X_train, y_train, X_test, y_test = import_data()

    logging.info("Beginning model training pipeline...")
    model = train_model(X_train, y_train)

    # TODO cleanup
    logging.info("Evaluating model performance...")
    placeholder = evaluate_model(model, X_test, y_test)
    return placeholder
