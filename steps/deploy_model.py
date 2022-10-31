from zenml.steps import step, BaseParameters


class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""
    min_precision: float
    min_recall: float


@step(enable_cache=False)
def deployment_trigger(
    precision: float,
    recall: float,
    config: DeploymentTriggerConfig,
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model precision and recall, and decides if the model is
    good enough to deploy.

    Args:
        precision (float): Model precision on test set
        recall (float): Model recall on test set
        config (DeploymentTriggerConfig): config with min metrics

    Returns:
        (bool) If True, deploy model. Else, stop pipeline.
    """
    precision_threshold_met = precision > config.min_precision
    recall_threshold_met = recall > config.min_recall
    conditions_met = precision_threshold_met and recall_threshold_met
    return conditions_met
