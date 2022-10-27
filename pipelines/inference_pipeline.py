from zenml.pipelines import pipeline


@pipeline
def inference_pipeline(
    dynamic_importer,
    prediction_service_loader,
    predictor,
):
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader()
    predictor(model_deployment_service, batch_data)
