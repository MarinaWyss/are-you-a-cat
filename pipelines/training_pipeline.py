from zenml.integrations.constants import MLFLOW
from zenml.config import DockerSettings
from zenml.pipelines import pipeline

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline(import_data, train_model, evaluate_model):
    """
    Args:
        import_data: DataClass
        train_model: DataClass
        evaluate_model: DataClass

    Returns:
        (float, float, float) precision, recall, and f1 scores
            from test set evaluation
    """
    X_train, y_train, X_test, y_test = import_data()
    model = train_model(X_train, y_train)
    precision, recall, f1 = evaluate_model(model, X_test, y_test)
