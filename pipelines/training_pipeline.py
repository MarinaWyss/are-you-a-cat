from zenml.pipelines import pipeline
from zenml.integrations.constants import MLFLOW


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
    X_train, y_train, _ = prepare_data(train=True)
    X_test, y_test, _ = prepare_data(train=False)

    model = train_model(X_train, y_train)

    # TODO cleanup
    placeholder = evaluate_model(model, X_test, y_test)
    return placeholder
