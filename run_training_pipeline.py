from steps.import_data import import_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from pipelines.training_pipeline import train_pipeline


def run_training():
    training = train_pipeline(
        import_data(),
        train_model(),
        evaluate_model(),
    )

    training.run()
