import os
import logging
from functools import lru_cache
from typing import Optional

import yaml

logging.basicConfig(level=logging.DEBUG)

# Default location of the project config, resolved relative to the repo root
# so that it can be loaded regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'steps', 'config.yaml')

# Keys that every valid config file is expected to define. This is a light
# sanity check so that a malformed or truncated config fails loudly at load
# time rather than deep inside model training.
REQUIRED_KEYS = (
    'image_dir_train',
    'image_dir_test',
    'n_images_per_dir',
    'image_size',
)


@lru_cache(maxsize=None)
def load_config(path: Optional[str] = None) -> dict:
    """Loads and validates the project YAML config.

    Previously every step re-implemented ``open('steps/config.yaml')`` inline,
    which made the config path impossible to override and easy to get wrong.
    This helper centralizes loading, resolves the path relative to the repo
    root, validates required keys, and caches the parsed result.

    Args:
        path (Optional[str]): Path to a YAML config file. Defaults to the
            project's ``steps/config.yaml``.

    Returns:
        (dict): Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required key is missing from the config.
    """
    config_path = path or DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}.")

    logging.info(f"Loading config from {config_path}.")
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)

    missing = [key for key in REQUIRED_KEYS if key not in configs]
    if missing:
        raise KeyError(
            f"Config at {config_path} is missing required keys: {missing}."
        )

    return configs
