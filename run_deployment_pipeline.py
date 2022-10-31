import yaml

from zenml.integrations.mlflow.steps import MLFlowDeployerParameters
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from steps.import_data import import_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.deploy_model import DeploymentTriggerConfig, deployment_trigger
from pipelines.deployment_pipeline import continuous_deployment_pipeline


def run_main():
    with open('steps/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    deployment = continuous_deployment_pipeline(
        import_data=import_data(),
        train_model=train_model(),
        evaluate_model=evaluate_model(),
        deployment_trigger=deployment_trigger(
            config=DeploymentTriggerConfig(
                min_precision=configs['min_precision'],
                min_recall=configs['min_recall']
            )
        ),
        model_deployer=mlflow_model_deployer_step(
            params=MLFlowDeployerParameters(workers=2)
        ),
    )
    deployment.run()

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    service = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=True,
    )

    if service[0]:
        print(
            f"The MLflow prediction server is running locally as a daemon process "
            f"and accepts inference requests at:\n"
            f"    {service[0].prediction_url}\n"
        )


if __name__ == "__main__":
    run_main()
