from zenml.pipelines import pipeline


@pipeline
def continuous_deployment_pipeline(
    import_data,
    train_model,
    evaluate_model,
    deployment_trigger,
    model_deployer,
):
    X_train, y_train, X_test, y_test = import_data()
    model = train_model(X_train, y_train)
    precision, recall, f1 = evaluate_model(model, X_test, y_test)
    deployment_decision = deployment_trigger(precision=precision, recall=recall)
    model_deployer(deployment_decision, model)
