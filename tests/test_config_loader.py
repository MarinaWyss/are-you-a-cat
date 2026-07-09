import os
import textwrap
import unittest

from steps import config_loader


class TestConfigLoader(unittest.TestCase):
    """Tests for the centralized config loader."""

    def setUp(self):
        # The loader is cached; clear it between tests so each run is isolated.
        config_loader.load_config.cache_clear()

    def _write_config(self, body: str) -> str:
        path = os.path.join(
            os.path.dirname(__file__), '_tmp_config.yaml')
        with open(path, 'w') as f:
            f.write(textwrap.dedent(body))
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def test_loads_project_config_by_default(self):
        configs = config_loader.load_config()
        self.assertIn('image_size', configs)
        self.assertIn('image_dir_train', configs)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            config_loader.load_config('does/not/exist.yaml')

    def test_missing_required_key_raises(self):
        path = self._write_config(
            """
            image_dir_train: 'a'
            image_dir_test: 'b'
            image_size: 64
            """
        )
        with self.assertRaises(KeyError):
            config_loader.load_config(path)

    def test_valid_custom_config_loads(self):
        path = self._write_config(
            """
            image_dir_train: 'a'
            image_dir_test: 'b'
            n_images_per_dir: 5
            image_size: 64
            """
        )
        configs = config_loader.load_config(path)
        self.assertEqual(configs['n_images_per_dir'], 5)
        self.assertEqual(configs['image_size'], 64)


if __name__ == '__main__':
    unittest.main()
