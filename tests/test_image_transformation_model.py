import unittest
from unittest.mock import Mock, patch
import numpy as np
from src.models.image_transformation_model import ImageTransformationModel

class TestImageTransformationModel(unittest.TestCase):
    def setUp(self):
        self.model = ImageTransformationModel()
        self.test_image = np.zeros((224, 224, 3), dtype=np.uint8)

    def test_preprocess_image(self):
        """Tests image preprocessing"""
        processed = self.model.preprocess_image(self.test_image)
        self.assertEqual(processed.shape, (1, 256, 256, 3))
        self.assertTrue(np.all(processed >= 0) and np.all(processed <= 1))

    @patch('tensorflow_hub.load')
    def test_load_style_transfer_model(self, mock_load):
        """Tests loading the style transfer model"""
        mock_model = Mock()
        mock_load.return_value = mock_model
        self.model._load_style_transfer_model()
        self.assertIsNotNone(self.model.style_transfer_model)
        mock_load.assert_called_once()

    def test_set_current_image(self):
        """Tests setting the current image"""
        self.model.set_current_image(self.test_image)
        self.assertIsNotNone(self.model.current_image)
        self.assertIsNotNone(self.model.original_image)
        self.assertEqual(len(self.model.history), 1)

    def test_add_to_history(self):
        """Tests adding an operation to history"""
        self.model.add_to_history("Test Operation", self.test_image)
        self.assertEqual(len(self.model.history), 1)
        self.assertEqual(self.model.current_index, 0)

    def test_super_resolution(self):
        """Tests the super resolution function"""
        expected_result = np.zeros((448, 448, 3), dtype=np.uint8)
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'output_url': 'test_url'}
            mock_post.return_value = mock_response
            with patch('requests.get') as mock_get:
                mock_get_response = Mock()
                mock_get_response.content = b'test_content'
                mock_get.return_value = mock_get_response
                with patch('cv2.imdecode') as mock_decode:
                    mock_decode.return_value = expected_result
                    result = self.model.super_resolution(self.test_image)
                    self.assertIsNotNone(result)
                    np.testing.assert_array_equal(result, expected_result)

    def test_can_undo_redo(self):
        """Tests checking undo/redo possibilities"""
        self.assertFalse(self.model.can_undo())
        self.assertFalse(self.model.can_redo())
        self.model.set_current_image(self.test_image)
        self.model.add_to_history("Test", np.ones((224, 224, 3)))
        self.assertTrue(self.model.can_undo())
        self.assertFalse(self.model.can_redo())

    def test_undo_redo(self):
        """Tests undo/redo operations"""
        self.model.add_to_history("Test", self.test_image)
        self.model.undo()
        self.model.redo()
        self.assertIsNotNone(self.model.current_image)

    def test_reset_image(self):
        """Tests resetting the image"""
        self.model.set_current_image(self.test_image)
        self.model.add_to_history("Test", np.ones((224, 224, 3)))
        self.model.reset_image()
        np.testing.assert_array_equal(
            self.model.current_image, self.test_image)