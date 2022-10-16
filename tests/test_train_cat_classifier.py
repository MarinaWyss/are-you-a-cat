import yaml

import unittest
import unittest.mock as mock

import train_cat_classifier as tcc


class TestDNNSkillTagging(unittest.TestCase):
    """Test DNN skill tagging."""
    @classmethod
    def setUpClass(cls):
        with open('tests/config_test.yaml', 'r') as file:
            configs = yaml.safe_load(file)

        cls.args = configs['training']

    @mock.patch('cat_classifier.CatClassifier.construct')
    @mock.patch('cat_classifier.CatClassifier.fit')
    @mock.patch('cat_classifier.CatClassifier.save')
    def test_main(self,
                  cat_classifier_construct,
                  cat_classifier_fit,
                  cat_classifier_save):

        tcc.main(self.args)

        # Assert model fitted and saved
        cat_classifier_construct.assert_called_once()
        cat_classifier_fit.assert_called_once()
        cat_classifier_save.assert_called_once()


if __name__ == '__main__':
    unittest.main()