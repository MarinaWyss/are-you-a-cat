import os
import logging
from random import shuffle

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall

from utils import load_data, format_data_for_model

logging.basicConfig(level=logging.DEBUG)


class CatClassifier:
    def __init__(self, args):
        self.args = args
        self.model = None

    def construct(self):
        """Construct model.

        Heavily inspired by
        https://github.com/rpeden/cat-or-not/blob/master/train.py
        """
        logging.info("Constructing model.")

        cnn2d = Sequential()
        cnn2d.add(
            Conv2D(
                units=self.args['conv_1_units'],
                kernel_size=(self.args['kernel_size'], self.args['kernel_size']),
                activation='relu',
                input_shape=(self.args['image_size'], self.args['image_size'], 1)))
        cnn2d.add(MaxPooling2D(
            pool_size=(self.args['max_pool'], self.args['max_pool'])))
        cnn2d.add(BatchNormalization())
        cnn2d.add(Conv2D(
            units=self.args['conv_2_units'],
            kernel_size=(self.args['kernel_size'], self.args['kernel_size']),
            activation='relu'))
        cnn2d.add(MaxPooling2D(
            pool_size=(self.args['max_pool'], self.args['max_pool'])))
        cnn2d.add(BatchNormalization())
        cnn2d.add(Conv2D(
            units=self.args['conv_3_units'],
            kernel_size=(self.args['kernel_size'], self.args['kernel_size']),
            activation='relu'))
        cnn2d.add(MaxPooling2D(
            pool_size=(self.args['max_pool'], self.args['max_pool'])))
        cnn2d.add(BatchNormalization())
        cnn2d.add(Conv2D(
            units=self.args['conv_4_units'],
            kernel_size=(self.args['kernel_size'], self.args['kernel_size']),
            activation='relu'))
        cnn2d.add(MaxPooling2D(
            pool_size=(self.args['max_pool'], self.args['max_pool'])))
        cnn2d.add(BatchNormalization())
        cnn2d.add(Conv2D(
            units=self.args['conv_5_units'],
            kernel_size=(self.args['kernel_size'], self.args['kernel_size']),
            activation='relu'))
        cnn2d.add(MaxPooling2D(
            pool_size=(self.args['max_pool'], self.args['max_pool'])))
        cnn2d.add(BatchNormalization())
        cnn2d.add(Dropout(
            self.args['dropout']))
        cnn2d.add(Flatten())
        cnn2d.add(Dense(
            units=self.args['dense_1_units'],
            activation='relu'))
        cnn2d.add(Dropout(
            self.args['dropout']))
        cnn2d.add(Dense(
            units=self.args['dense_2_units'],
            activation='relu'))
        cnn2d.add(Dense(
            units=self.args['dense_3_units'],
            activation='sigmoid'))

        # TODO from_logits
        cnn2d.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=self.args['learning_rate']),
                      metrics=[AUC(), Precision(), Recall()])

        logging.info("Model compiled.")

        self.model = cnn2d
        return self.model

    def fit(self):
        """Fit model.

        Raises:
            Exception, if the model is not already initialized.
        """
        if not self.model:
            raise Exception('Error: Initialize model before fitting.')

        logging.info('Reading data.')
        data = load_data(train=True)

        logging.info('Formatting data.')
        X_train, y_train, train_paths = format_data_for_model(data)

        logging.info("Fitting model.")
        self.model.fit(x=X_train,
                       y=y_train,
                       epochs=self.args['num_epochs'],
                       batch_size=self.args['batch_size'],
                       validation_split=self.args['val_split'],
                       seed=self.args['random_seed'],
                       shuffle=True,
                       )

        logging.info("Model successfully fit.")

    def save(self):
        """Save model.

        Raises:
            Exception, if the model is not already initialized.
        """
        if not self.model:
            raise Exception('Error: Initialize model before saving.')

        self.model.save(self.args['output_path'])
        logging.info("Model saved.")
