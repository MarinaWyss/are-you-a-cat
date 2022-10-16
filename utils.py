import os
import yaml
import logging
from typing import List

import numpy as np

from PIL import Image
from random import choice

logging.basicConfig(level=logging.DEBUG)

with open('config.yaml', 'r') as file:
    configs = yaml.safe_load(file)

def label_img(dir_name: str) -> np.array:
    """Labels images based on their directory name.
    Images in the  `cats` directory are positive cases,
    while any other dir name is labeled negative.

    Args:
        dir_name (str): Directory name.

    Returns:
        (np.array): Array with shape (2, )
    """
    if dir_name == 'cats':
        return np.array([1, 0])
    else:
        return np.array([0, 1])

def load_data(train: bool = True) -> List:
    """Loads images, resizes them, and converts them to black and white.
    For faster training/testing, this code only pulls a sample of images
    from each directory, based on a parameter in the config file.

    This code is heavily inspired by
    https://github.com/rpeden/cat-or-not/blob/master/train.py

    Args:
        train (bool): If True, load train images. Else, test images.

    Returns:
        (list): List with the image array, label array, and filename
    """
    subdir = configs['image_dir_train'] if train else configs['image_dir_test']
    image_dir = os.path.join(os.path.abspath(os.getcwd()), subdir)

    data = []
    directories = next(os.walk(image_dir))[1]

    for dirname in directories:
      logging.info(f'Loading images from the {dirname} directory.')
      file_names = next(os.walk(os.path.join(image_dir, dirname)))[2]

      for i in range(configs['n_images_per_dir']):
        image_name = choice(file_names)
        image_path = os.path.join(image_dir, dirname, image_name)
        label = label_img(dirname)
        if 'DS_Store' not in image_path and '.csv' not in image_path:
          img = Image.open(image_path)
          img = img.convert('L')
          img = img.resize(
              (configs['training']['image_size'],
               configs['training']['image_size']),
              Image.ANTIALIAS)
          data.append([np.array(img), label, image_path])
    return data

def format_data_for_model(dat_list: List) -> (np.array, np.array, np.array):
    """Takes in a list including images as np.arrays,
    labels, and image_paths, and reformats them for model
    training/prediction.

    Args:
        dat_list (List[np.array, np.array, np.array])

    Returns:
         (np.array, np.array, np.array): formatted data for
         model training or prediction.
    """
    images = np.array(
        [i[0] for i in dat_list]).reshape(
            -1,
            configs['training']['image_size'],
            configs['training']['image_size'],
            1
        )
    labels = np.array([i[1] for i in dat_list])
    image_paths = np.array([i[2] for i in dat_list])
    return images, labels, image_paths
