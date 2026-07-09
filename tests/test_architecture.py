import unittest

try:
    import tensorflow as tf
    from model.architecture import build_cnn, NUM_CONV_BLOCKS
    TF_AVAILABLE = True
except ImportError:  # pragma: no cover - CI installs TF, local may not
    TF_AVAILABLE = False


# A tiny config so the model builds quickly in tests.
SMALL_CONFIG = {
    'image_size': 32,
    'kernel_size': 3,
    'max_pool': 2,
    'conv_1_filters': 8,
    'conv_2_filters': 8,
    'conv_3_filters': 8,
    'conv_4_filters': 8,
    'conv_5_filters': 8,
    'dropout': 0.2,
    'dense_1_units': 16,
    'dense_2_units': 8,
    'dense_3_units': 2,
    'learning_rate': 0.001,
}


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow is not installed")
class TestArchitecture(unittest.TestCase):
    """Tests that the shared CNN builder produces a sane model."""

    def test_build_cnn_output_shape(self):
        model = build_cnn(SMALL_CONFIG)
        self.assertEqual(model.output_shape, (None, SMALL_CONFIG['dense_3_units']))

    def test_build_cnn_input_shape(self):
        model = build_cnn(SMALL_CONFIG)
        size = SMALL_CONFIG['image_size']
        self.assertEqual(model.input_shape, (None, size, size, 1))

    def test_build_cnn_has_expected_conv_blocks(self):
        model = build_cnn(SMALL_CONFIG)
        conv_layers = [
            layer for layer in model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        self.assertEqual(len(conv_layers), NUM_CONV_BLOCKS)

    def test_model_is_compiled(self):
        model = build_cnn(SMALL_CONFIG)
        self.assertIsNotNone(model.optimizer)


if __name__ == '__main__':
    unittest.main()
