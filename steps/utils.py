import os
from typing import List, Tuple

import numpy as np

from PIL import Image
from random import choice

from steps.logging_config import get_logger
from steps.preprocessing import preprocess_image

logger = get_logger(__name__)

# Files that live alongside the images but must never be treated as images.
IGNORED_SUFFIXES = ('DS_Store', '.csv')


def label_img(dir_name: str) -> np.ndarray:
    """Labels images based on their directory name.
    Images in the `cats` directory are positive cases,
    while any other dir name is labeled negative.

    Args:
        dir_name (str): Directory name.

    Returns:
        (np.ndarray): Array with shape (2, )
    """
    if dir_name == 'cats':
        return np.array([1, 0])
    return np.array([0, 1])


def _is_image_path(path: str) -> bool:
    """Returns True if the path points at something we should load as an image."""
    return not any(suffix in path for suffix in IGNORED_SUFFIXES)


def load_data(train: bool,
              configs: dict) -> List:
    """Loads images, resizes them, and converts them to black and white.
    For faster training/testing, this code only pulls a sample of images
    from each directory, based on a parameter in the config file.

    This code is heavily inspired by
    https://github.com/rpeden/cat-or-not/blob/master/train.py

    Args:
        train (bool): If True, load train images. Else, test images.
        configs (dict): Dictionary of configs

    Returns:
        (list): List with the image array, label array, and filename
    """
    subdir = configs['image_dir_train'] if train else configs['image_dir_test']
    logger.info(f'Loading data from the {subdir} sub-directory.')

    image_dir = os.path.join(os.path.abspath(os.getcwd()), subdir)
    image_size = configs['image_size']

    data = []
    directories = next(os.walk(image_dir))[1]

    for dirname in directories:
        logger.info(f'Loading images from the {dirname} directory.')
        file_names = next(os.walk(os.path.join(image_dir, dirname)))[2]
        label = label_img(dirname)

        for _ in range(configs['n_images_per_dir']):
            image_name = choice(file_names)
            image_path = os.path.join(image_dir, dirname, image_name)
            if not _is_image_path(image_path):
                continue
            img = Image.open(image_path)
            # Preprocessing is shared with the serving path via
            # steps.preprocessing so training and inference stay in sync.
            processed = preprocess_image(img, image_size)
            # Store as uint8 in the 0-255 range; normalization to [0, 1]
            # happens once, centrally, in format_data_for_model.
            data.append(
                [(processed * 255.0).astype(np.uint8), label, image_path]
            )
    return data


def format_data_for_model(dat_list: List,
                          configs: dict) -> Tuple:
    """Takes in a list including images as np.ndarrays,
    labels, and image_paths, and reformats them for model
    training/prediction.

    Args:
        dat_list (List[np.ndarray, np.ndarray, np.ndarray])
        configs (dict): Config dictionary

    Returns:
         (np.ndarray, np.ndarray, np.ndarray): formatted data for
         model training or prediction.
    """
    image_size = configs['image_size']
    images = np.array(
        [i[0] for i in dat_list]).reshape(-1, image_size, image_size, 1)
    # normalizing images to 0-1
    images = images / 255.

    labels = np.array([i[1] for i in dat_list])

    image_paths = np.array([i[2] for i in dat_list])
    return images, labels, image_paths
