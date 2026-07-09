import numpy as np
from PIL import Image

from steps.logging_config import get_logger

logger = get_logger(__name__)

# Pillow 10 removed the ``Image.ANTIALIAS`` alias in favour of the
# ``Resampling`` enum. Resolve the best available option once so the rest of
# the code base does not have to care which Pillow version is installed.
try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    RESAMPLE_FILTER = Image.ANTIALIAS


def preprocess_image(img: Image.Image, image_size: int) -> np.ndarray:
    """Converts a PIL image to the grayscale, resized, normalized array the
    model expects.

    This logic was previously duplicated between ``steps/utils.load_data`` and
    ``streamlit_app.py`` (and subtly diverged between the two). Centralizing it
    guarantees training and serving preprocess images identically.

    Args:
        img (PIL.Image.Image): Source image.
        image_size (int): Target square edge length in pixels.

    Returns:
        (np.ndarray): Float array of shape ``(image_size, image_size)`` with
            values scaled to the ``[0, 1]`` range.
    """
    img = img.convert('L')  # grayscale
    img = img.resize((image_size, image_size), RESAMPLE_FILTER)
    return np.asarray(img) / 255.0


def image_to_model_input(img: Image.Image, image_size: int) -> np.ndarray:
    """Preprocesses a single image and reshapes it into a batch of one.

    Args:
        img (PIL.Image.Image): Source image.
        image_size (int): Target square edge length in pixels.

    Returns:
        (np.ndarray): Array of shape ``(1, image_size, image_size, 1)`` ready
            to be passed to ``model.predict``.
    """
    processed = preprocess_image(img, image_size)
    return processed.reshape(-1, image_size, image_size, 1)
