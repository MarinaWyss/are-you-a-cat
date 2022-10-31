from steps.import_data import dynamic_importer
from steps.inference import (
    MLFlowDeploymentLoaderStepParameters,
    prediction_service_loader,
    predictor
)
from pipelines.inference_pipeline import inference_pipeline


def run_main():
    inference = inference_pipeline(
        dynamic_importer=dynamic_importer(),
        prediction_service_loader=prediction_service_loader(
            MLFlowDeploymentLoaderStepParameters(
                pipeline_name="continuous_deployment_pipeline",
                step_name="mlflow_model_deployer_step",
            )
        ),
        predictor=predictor(),
    )
    inference.run()


if __name__ == "__main__":
    run_main()
