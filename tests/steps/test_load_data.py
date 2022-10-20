import yaml
import unittest

from steps.load_data import prepare_data


class TestLoadData(unittest.TestCase):
    """Test data loading functions."""
    @classmethod
    def setUpClass(cls):
        with open('tests/config_test.yaml', 'r') as file:
            configs = yaml.safe_load(file)

        cls.configs = configs

    def test_prepare_data_train(self):
        images, labels, paths = prepare_data(
            train=True,
            configs=self.configs
        )

        self.assertEqual(len(images), 20)
        self.assertEqual(len(labels), 20)
        self.assertEqual(len(paths), 20)
        self.assertEqual(labels.mean(), 0.5)

    def test_prepare_data_test(self):
        images, labels, paths = prepare_data(
            train=False,
            configs=self.configs
        )

        self.assertEqual(len(images), 20)
        self.assertEqual(len(labels), 20)
        self.assertEqual(len(paths), 20)
        self.assertEqual(labels.mean(), 0.5)


if __name__ == '__main__':
    unittest.main()