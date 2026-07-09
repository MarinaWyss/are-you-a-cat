from steps.import_data import import_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.logging_config import get_logger
from pipelines.training_pipeline import train_pipeline

logger = get_logger(__name__)


def run_training():
    logger.info("Kicking off the training pipeline...")
    training = train_pipeline(
        import_data(),
        train_model(),
        evaluate_model(),
    )

    training.run()


if __name__ == "__main__":
    run_training()
