import unittest
from unittest.mock import Mock, patch
import numpy as np
from src.models.image_classification_model import (ImageClassificationModel,
                                                   UnsupportedModelError)

class TestImageClassificationModel(unittest.TestCase):
    def setUp(self):
        self.model = ImageClassificationModel()
        self.test_image = np.zeros((224, 224, 3), dtype=np.uint8)

    def test_validate_input_valid_image(self):
        """Tests validation of correct image input"""
        try:
            self.model._validate_input(self.test_image, 3)
        except ValueError:
            self.fail("_validate_input threw an error for valid input")

    def test_validate_input_invalid_dimensions(self):
        """Tests validation of image with incorrect dimensions"""
        invalid_image = np.zeros((224, 224), dtype=np.uint8)  # 2D image
        with self.assertRaises(ValueError):
            self.model._validate_input(invalid_image, 3)

    def test_validate_input_invalid_channels(self):
        """Tests validation of image with incorrect channel count"""
        invalid_image = np.zeros((224, 224, 4), dtype=np.uint8)  # 4 channels
        with self.assertRaises(ValueError):
            self.model._validate_input(invalid_image, 3)

    def test_validate_input_invalid_top_n(self):
        """Tests validation of incorrect top_n value"""
        with self.assertRaises(ValueError):
            self.model._validate_input(self.test_image, 20)  # value too large

    @patch('tensorflow.keras.applications.MobileNetV2')
    def test_load_model_supported(self, mock_model):
        """Tests loading a supported model"""
        model = ImageClassificationModel("mobilenetv2")
        self.assertIsNotNone(model.model)

    def test_load_model_unsupported(self):
        """Tests loading an unsupported model"""
        with self.assertRaises(UnsupportedModelError):
            ImageClassificationModel("does_not_exist")

    def test_classify_image_success(self):
        """Tests successful image classification"""
        # Mock prediction functions
        self.model.preprocess_input = Mock(return_value=self.test_image)
        self.model.model = Mock()
        self.model.model.predict = Mock(return_value=np.array([[0.8, 0.2]]))
        self.model.decode_predictions = Mock(
            return_value=[[('n01234', 'cat', 0.8), ('n05678', 'dog', 0.2)]]
        )
        results = self.model.classify_image(self.test_image, top_n=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 'cat')
        self.assertEqual(results[0][1], 0.8)

    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_add_custom_model(self, mock_makedirs, mock_exists):
        """Tests adding a custom model"""
        mock_exists.return_value = True
        with patch('shutil.copy2') as mock_copy:
            self.model.add_custom_model(
                'test_model.h5',
                'test_labels.txt'
            )
            mock_makedirs.assert_called_once()
            self.assertEqual(
                mock_copy.call_count,
                2)  # copying model and labels

    def test_get_available_models(self):
        """Tests retrieving available models"""
        models = self.model.get_available_models()
        self.assertIsInstance(models, dict)
        self.assertIn('mobilenetv2', models)
        self.assertEqual(models['mobilenetv2'], 'built-in')