import unittest
from unittest.mock import patch
import numpy as np
from src.models.image_editing_model import (ImageEditingModel, ImageLoadError,
                                            ImageProcessingError)

class TestImageEditingModel(unittest.TestCase):
    def setUp(self):
        self.model = ImageEditingModel()
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_load_image(self):
        with patch('cv2.imread') as mock_imread:
            mock_imread.return_value = self.test_image
            self.model.load_image("test.jpg")
            self.assertIsNotNone(self.model.current_image)
            self.assertIsNotNone(self.model.original_image)
        with patch('cv2.imread', return_value=None):
            with self.assertRaises(ImageLoadError):
                self.model.load_image("nonexistent.jpg")

    def test_is_grayscale(self):
        grayscale = np.zeros((100, 100), dtype=np.uint8)
        color = np.zeros((100, 100, 3), dtype=np.uint8)
        self.model.current_image = grayscale
        self.assertTrue(self.model.is_grayscale())
        self.model.current_image = color
        self.assertFalse(self.model.is_grayscale())

    def test_apply_function(self):
        """Tests calling a non-existent function"""
        self.model.current_image = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(ImageProcessingError) as context:
            self.model.apply_function("NonexistentFunction")
        self.assertIn("not found", str(context.exception))

    def test_undo_redo(self):
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.model.current_image = test_image.copy()
        self.model.original_image = test_image.copy()
        modified_image = np.ones((100, 100, 3), dtype=np.uint8)
        self.model.add_to_history("Test Operation", (), modified_image)
        self.model.undo()
        np.testing.assert_array_equal(self.model.current_image, test_image)
        self.model.redo()
        np.testing.assert_array_equal(self.model.current_image, modified_image)

    def test_reset_image(self):
        self.model.current_image = self.test_image
        self.model.original_image = self.test_image.copy()
        modified = np.ones((100, 100, 3), dtype=np.uint8)
        self.model.add_to_history("Test", (), modified)
        self.model.reset_image()
        np.testing.assert_array_equal(
            self.model.current_image, self.test_image)