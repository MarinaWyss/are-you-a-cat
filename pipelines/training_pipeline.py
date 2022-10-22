from zenml.pipelines import pipeline


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
    X_train, y_train, X_test, y_test = import_data()
    model = train_model(X_train, y_train)
    # TODO actually finish this
    placeholder = evaluate_model(model, X_test, y_test)
    return placeholder
