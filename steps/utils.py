import boto3
import logging
from typing import List, Tuple, Iterable

import numpy as np

from io import BytesIO
from PIL import Image
from random import choice

logging.basicConfig(level=logging.DEBUG)


def label_img(dir_name: str) -> np.ndarray:
    """Labels images based on their directory name.
    Images in the  `cats` directory are positive cases,
    while any other dir name is labeled negative.
    Args:
        dir_name (str): Directory name.
    Returns:
        (np.ndarray): Array with shape (2, )
    """
    if dir_name == 'cats':
        return np.array([1, 0])
    else:
        return np.array([0, 1])


def get_items_in_bucket(client,
                        bucket_name: str,
                        prefix: str) -> Iterable[str]:
    """Gets all the items in an s3 bucket.
    Args:
        client (s3_client)
        bucket_name (str): S3 bucket name
        prefix (str): S3 sub-dir name
    Returns:
        (Iterable[str]): Iterable of all item names
    """
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    for page in pages:
        for obj in page.get("Contents", []):
            yield obj["Key"]


def get_image_dirs(client,
                   bucket_name: str,
                   prefix: str) -> List[str]:
    """Gets all the sub-dir names in an s3 bucket.
    Args:
        client (s3_client)
        bucket_name (str): S3 bucket name
        prefix (str): S3 sub-dir name
    Returns:
        (List[str]): List of bucket names
    """
    rsp = client.list_objects_v2(
        Bucket=bucket_name, Prefix=f"{prefix}/", Delimiter="/")
    image_dirs = list(obj["Prefix"] for obj in rsp["CommonPrefixes"])
    image_dirs = [d.replace(f'{prefix}', '').replace('/', '') for d in image_dirs]
    return image_dirs


def load_data(train: bool,
              configs: dict) -> List:
    """Loads images, resizes them, and converts them to black and white.
    For faster training/testing, this code only pulls a sample of images
    from each directory, based on a parameter in the config file.
    Args:
        train (bool): If True, load train images. Else, test images.
        configs (dict): Dictionary of configs
    Returns:
        (list): List with the image array, label array, and filename
    """
    # Set up s3 stuff
    client = boto3.client("s3")
    resource = boto3.resource("s3")
    bucket = resource.Bucket(configs['s3_bucket'])

    # Get s3 paths
    subdir = configs['image_dir_train'] if train else configs['image_dir_test']
    logging.info(f'Loading data from the {subdir} s3 sub-directory.')
    image_dirs = get_image_dirs(client, configs['s3_bucket'], subdir)

    data = []
    for dirname in image_dirs:
        logging.info(f'Loading images from the {dirname} directory.')

        file_names = list(
            get_items_in_bucket(
                client=client,
                bucket_name=configs['s3_bucket'],
                prefix=f"{subdir}/{dirname}/"
            )
        )

        for i in range(configs['n_images_per_dir']):
            image_name = choice(file_names)
            label = label_img(dirname)
            if 'DS_Store' not in image_name and '.csv' not in image_name:
                response = bucket.Object(image_name).get()
                img = Image.open(response['Body'])
                img = img.convert('L')
                img = img.resize(
                    (configs['image_size'], configs['image_size']),
                    Image.ANTIALIAS)
                data.append([np.array(img), label, image_name])
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
    images = np.array(
        [i[0] for i in dat_list]).reshape(
        -1, configs['image_size'], configs['image_size'], 1)

    # normalizing images to 0-1
    images = images/255.

    labels = np.array([i[1] for i in dat_list])

    image_paths = np.array([i[2] for i in dat_list])
    return images, labels, image_paths
