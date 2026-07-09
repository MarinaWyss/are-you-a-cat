import keras_tuner as kt
import tensorflow as tf

from model.architecture import build_cnn
from steps.config_loader import load_config
from steps.logging_config import get_logger
from steps.seed import set_random_seeds
from steps.utils import load_data, format_data_for_model

logger = get_logger(__name__)

# Keras Tuner search settings. Pulled out as constants so a reviewer can see
# the search budget at a glance.
MAX_EPOCHS = 20
SEARCH_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
TUNER_PROJECT_NAME = "hyperband_tuner2"


def build_model(hp):
    """Adapts the shared architecture builder to the Keras Tuner interface.

    Previously this file re-implemented the entire network (a ``# TODO`` called
    out the copy-paste). It now delegates to :func:`model.architecture.build_cnn`
    and simply forwards the tuner's ``HyperParameters`` object.
    """
    configs = load_config()
    return build_cnn(configs, hp=hp)


def main():
    """Runs the steps for hyperparam tuning with keras_tuner."""
    configs = load_config()

    if 'random_seed' in configs:
        set_random_seeds(configs['random_seed'])

    logger.info("Loading and preparing data...")
    train_data = load_data(train=True, configs=configs)
    X_train, y_train, _ = format_data_for_model(
        dat_list=train_data, configs=configs
    )
    logger.info("Data loaded.")

    logger.info("Initializing tuner...")
    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=MAX_EPOCHS,
        project_name=TUNER_PROJECT_NAME,
    )

    logger.info("Starting tuning job...")
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=EARLY_STOPPING_PATIENCE)
    tuner.search(
        X_train,
        y_train,
        epochs=SEARCH_EPOCHS,
        validation_split=configs['val_split'],
        callbacks=[stop_early],
    )
    logger.info("Tuning job done.")

    logger.info("The best hyperparams are:")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info(best_hps.values)


if __name__ == '__main__':
    main()
