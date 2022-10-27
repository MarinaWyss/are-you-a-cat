import json
import numpy as np

from zenml.steps import BaseParameters, step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters
    Attributes:
        pipeline_name (str): name of the pipeline that deployed the MLflow prediction
            server
        step_name (str): the name of the step that deployed the MLflow prediction
            server
        running (bool): when this flag is set, the step only returns a running service
    """
    pipeline_name: str
    step_name: str
    running: bool = True


@step()
def prediction_service_loader(
    params: MLFlowDeploymentLoaderStepParameters,
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline"""

    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=params.pipeline_name,
        pipeline_step_name=params.step_name,
        running=params.running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{params.step_name} step in the {params.pipeline_name} "
            f"pipeline is currently "
            f"running."
        )

    return existing_services[0]


@step()
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service.

    Args:
        service (MLFlowDeploymentService)
        data (str): Image data formatted as a string?

    Returns:
        (np.ndarray) Prediction
    """
    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    # TODO format image data
    json_list = json.loads(json.dumps(list(data.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction
