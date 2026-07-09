import unittest

import numpy as np
from PIL import Image

from steps.preprocessing import preprocess_image, image_to_model_input


class TestPreprocessing(unittest.TestCase):
    """Tests for the shared image preprocessing helpers."""

    @classmethod
    def setUpClass(cls):
        cls.image_size = 64
        # A small RGB gradient image to exercise grayscale + resize.
        arr = np.tile(
            np.linspace(0, 255, 32, dtype=np.uint8), (32, 1))
        cls.image = Image.fromarray(np.stack([arr, arr, arr], axis=-1), 'RGB')

    def test_preprocess_shape_and_range(self):
        result = preprocess_image(self.image, self.image_size)
        self.assertEqual(result.shape, (self.image_size, self.image_size))
        self.assertLessEqual(result.max(), 1.0)
        self.assertGreaterEqual(result.min(), 0.0)

    def test_model_input_shape(self):
        result = image_to_model_input(self.image, self.image_size)
        self.assertEqual(
            result.shape, (1, self.image_size, self.image_size, 1))

    def test_grayscale_conversion_is_single_channel(self):
        # Preprocessing collapses the 3 RGB channels into one.
        result = preprocess_image(self.image, self.image_size)
        self.assertEqual(result.ndim, 2)


if __name__ == '__main__':
    unittest.main()
