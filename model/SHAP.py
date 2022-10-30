import yaml
import numpy as np

import shap
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

from steps.utils import load_data, format_data_for_model

with open('steps/config.yaml', 'r') as file:
    configs = yaml.safe_load(file)

train_data = load_data(train=True, configs=configs)
test_data = load_data(train=False, configs=configs)

X_train, y_train, train_paths = format_data_for_model(
    dat_list=train_data, configs=configs
)

X_test, y_test, test_paths = format_data_for_model(
    dat_list=test_data, configs=configs
)

model = tf.keras.models.load_model('saved_model/model-2022-10-30-12-53-05.h5')

background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

e = shap.DeepExplainer(model, background)

shap_values = e.shap_values(X_test[0:5])

shap.image_plot(shap_values, -X_test[0:5])

