import yaml

import numpy as np

import unittest
import unittest.mock as mock

from model.cat_classifier import CatClassifier


class TestCatClassifier(unittest.TestCase):
    """Test the CatClassifier model."""
    @classmethod
    def setUpClass(cls):
        with open('tests/config_test.yaml', 'r') as file:
            configs = yaml.safe_load(file)

        cls.args = configs
        cls.X_train = np.zeros(256)
        cls.y_train = np.array([1, 0])

    def test_initialize_class(self):
        # Initialize class
        cnn2d = CatClassifier(self.args)

        # Assert args are correct
        self.assertEqual(cnn2d.args['kernel_size'], 3)
        self.assertEqual(cnn2d.args['val_split'], 0.2)
        self.assertEqual(cnn2d.args['output_path'], 'model_test.h5')

    def test_train(self):
        # Construct model
        cnn2d = CatClassifier(self.args)

        model = cnn2d.train(self.X_train, self.y_train)

        # Assert model is constructed correctly
        self.assertIsNotNone(model)
        self.assertEqual(21, len(model.layers))
        self.assertEqual(1, len(model.outputs))

    @mock.patch('tensorflow.keras.models.save_model')
    def test_save(self, save_model):
        cnn2d = CatClassifier(self.args)
        model = cnn2d.train(self.X_train, self.y_train)
        cnn2d.save(model)

        save_model.assert_called_once_with(model,
                                           filepath=cnn2d.args['output_path'],
                                           save_format='h5')


if __name__ == '__main__':
    unittest.main()