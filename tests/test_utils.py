import yaml
import numpy as np
import unittest

import utils


class TestUtils(unittest.TestCase):
    """Test util functions."""
    @classmethod
    def setUpClass(cls):
        with open('tests/config_test.yaml', 'r') as file:
            configs = yaml.safe_load(file)

        cls.configs = configs

    def test_label_img_cat(self):
        res = utils.label_img('cats')
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, np.array([1, 0]))

    def test_label_img_not_cat(self):
        res = utils.label_img('dogs')
        self.assertIsInstance(res, np.ndarray)
        np.testing.assert_array_equal(res, np.array([0, 1]))

    def test_load_data(self):
        res = utils.load_data(train=True, configs=self.configs)
        # There are two train subdirectories, and we grab 10
        # images from each
        self.assertEqual(len(res), 20)
        # Each list in the main list has an image, label, and path
        self.assertEqual(len(res[0]), 3)
        # Check that the image looks correct
        self.assertEqual(len(res[0][0]), self.configs['image_size'])
        self.assertIsInstance(res[0][0], np.ndarray)
        # Check that the labels are correct
        np.testing.assert_array_equal(res[0][1], np.array([1, 0]))
        # Check that the path is correct
        self.assertIsInstance(res[0][2], str)

    def test_format_data_for_model(self):
        dat_list = utils.load_data(
            train=True,
            configs=self.configs
        )

        images, labels, paths = utils.format_data_for_model(
            dat_list,
            self.configs
        )

        self.assertEqual(len(images), 20)
        self.assertEqual(len(labels), 20)
        self.assertEqual(len(paths), 20)
        self.assertEqual(labels.mean(), 0.5)


if __name__ == '__main__':
    unittest.main()