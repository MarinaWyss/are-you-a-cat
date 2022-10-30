import yaml
import logging

import keras_tuner as kt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization
)
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall

from steps.utils import load_data, format_data_for_model

logging.basicConfig(level=logging.DEBUG)


def build_model(hp):
    """Sets up the model with hyperparam choices."""
    with open('steps/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    logging.info("Setting up model and hyperparam choices...")
    # TODO set this up so that this isn't copy-pasta with cat_classifier.py
    cnn2d = Sequential()
    cnn2d.add(
        Conv2D(
            filters=hp.Int('conv_1_filter', min_value=32, max_value=512, step=16),
            kernel_size=hp.Choice('conv_1_kernel', values=[2, 3, 5, 7]),
            activation='relu',
            input_shape=(configs['image_size'], configs['image_size'], 1)))
    cnn2d.add(MaxPooling2D(
        pool_size=hp.Choice('max_pool_1', values=[1, 2, 3])))
    cnn2d.add(BatchNormalization())
    cnn2d.add(Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=512, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values=[2, 3, 5, 7]),
        activation='relu'))
    cnn2d.add(MaxPooling2D(
        pool_size=hp.Choice('max_pool_2', values=[1, 2, 3])))
    cnn2d.add(BatchNormalization())
    cnn2d.add(Conv2D(
        filters=hp.Int('conv_3_filter', min_value=32, max_value=512, step=16),
        kernel_size=hp.Choice('conv_3_kernel', values=[2, 3, 5, 7]),
        activation='relu'))
    cnn2d.add(MaxPooling2D(
        pool_size=hp.Choice('max_pool_3', values=[1, 2, 3])))
    cnn2d.add(BatchNormalization())
    cnn2d.add(Conv2D(
        filters=hp.Int('conv_4_filter', min_value=32, max_value=512, step=16),
        kernel_size=hp.Choice('conv_4_kernel', values=[2, 3, 5, 7]),
        activation='relu'))
    cnn2d.add(MaxPooling2D(
        pool_size=hp.Choice('max_pool_4', values=[1, 2, 3])))
    cnn2d.add(BatchNormalization())
    cnn2d.add(Conv2D(
        filters=hp.Int('conv_5_filter', min_value=32, max_value=512, step=16),
        kernel_size=hp.Choice('conv_5_kernel', values=[2, 3, 5, 7]),
        activation='relu'))
    cnn2d.add(MaxPooling2D(
        pool_size=hp.Choice('max_pool_5', values=[1, 2, 3])))
    cnn2d.add(BatchNormalization())
    cnn2d.add(Dropout(hp.Float("dropout_1", min_value=0, max_value=0.5)))
    cnn2d.add(Flatten())
    cnn2d.add(Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=256, step=16),
        activation='relu'))
    cnn2d.add(Dropout(hp.Float("dropout_2", min_value=0, max_value=0.5)))
    cnn2d.add(Dense(
        units=hp.Int('dense_2_units', min_value=32, max_value=128, step=16),
        activation='relu'))
    cnn2d.add(Dense(
        units=configs['dense_3_units'],
        activation='sigmoid'))
    cnn2d.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  metrics=[AUC(), Precision(), Recall()])
    logging.info("Model compiled.")
    return cnn2d


def main():
    """Runs the steps for hyperparam tuning with keras_tuner."""
    with open('steps/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)

    logging.info("Loading and preparing data...")
    train_data = load_data(train=True, configs=configs)
    X_train, y_train, _ = format_data_for_model(
        dat_list=train_data, configs=configs
    )
    logging.info("Data loaded.")

    logging.info("Initializing tuner...")
    tuner = kt.Hyperband(hypermodel=build_model,
                         objective=kt.Objective("val_auc", direction="max"),
                         max_epochs=20,
                         project_name="hyperband_tuner2")

    logging.info("Starting tuning job...")
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)
    tuner.search(
        X_train,
        y_train,
        epochs=50,
        validation_split=0.2,
        callbacks=[stop_early]
    )
    logging.info("Tuning job done.")

    logging.info("The best hyperparams are:")
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logging.info(best_hps)


if __name__ == '__main__':
    main()
