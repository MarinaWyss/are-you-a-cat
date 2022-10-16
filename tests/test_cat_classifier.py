import yaml

import unittest
import unittest.mock as mock

from cat_classifier import CatClassifier


class TestCatClassifier(unittest.TestCase):
    """Test the CatClassifier model."""
    @classmethod
    def setUpClass(cls):
        with open('tests/config_test.yaml', 'r') as file:
            configs = yaml.safe_load(file)

        cls.args = configs['training']

    def test_initialize_class(self):
        # Initialize class
        cnn2d = CatClassifier(self.args)

        # Assert args are correct
        self.assertEqual(cnn2d.args['kernel_size'], 3)
        self.assertEqual(cnn2d.args['val_split'], 0.2)
        self.assertEqual(cnn2d.args['output_path'], 'model_test.h5')

        # Assert model is None
        self.assertIsNone(cnn2d.model)

    def test_construct(self):
        # Construct model
        cnn2d = CatClassifier(self.args)
        cnn2d.construct()
        model = cnn2d.model

        # Assert model is constructed correctly
        self.assertIsNotNone(model)
        self.assertEqual(21, len(model.layers))
        self.assertEqual(1, len(model.outputs))

    def test_fit(self):
        cnn2d = CatClassifier(self.args)

        with self.assertRaises(Exception) as ex:
            cnn2d.fit()  # Fit model that hasn't been initialized yet
        # Assert failure message is correct
        self.assertEqual('Error: Initialize model before fitting.',
                         str(ex.exception))

    def test_save_model_not_initialized(self):
        cnn2d = CatClassifier(self.args)

        with self.assertRaises(Exception) as ex:
            cnn2d.save()  # Save model that hasn't been initialized yet
        # Assert failure message is correct
        self.assertEqual('Error: Initialize model before saving.',
                         str(ex.exception))

    @mock.patch('tensorflow.keras.models.save_model')
    def test_save(self, save_model):
        cnn2d = CatClassifier(self.args)
        cnn2d.construct()
        cnn2d.save()

        save_model.assert_called_once_with(cnn2d.model,
                                           filepath=cnn2d.args['output_path'],
                                           save_format='h5')


if __name__ == '__main__':
    unittest.main()