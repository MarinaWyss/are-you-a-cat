import logging

from zenml.steps import step

logging.basicConfig(level=logging.DEBUG)


@step
def evaluate_model():
    pass
