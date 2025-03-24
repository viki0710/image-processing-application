import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PyQt5.QtWidgets import QApplication
from src.controllers.image_transformation_controller import \
    ImageTransformationController
from src.models.image_transformation_model import ImageTransformationModel
from src.views.image_transformation_view import ImageTransformationView

app = QApplication(sys.argv)

class TestImageTransformationIntegration(unittest.TestCase):
    def setUp(self):
        self.model = ImageTransformationModel()
        self.view = ImageTransformationView()
        self.controller = ImageTransformationController(self.model, self.view)
        self.test_image = np.zeros((224, 224, 3), dtype=np.uint8)

    def test_full_style_transfer_workflow(self):
        """Tests the complete style transfer workflow"""
        with patch('cv2.imread') as mock_imread:
            mock_imread.return_value = self.test_image
            self.controller.select_image()
        with patch('src.views.dialogs.style_transfer_dialog.StyleTransferDialog') as mock_dialog:
            mock_dialog_instance = Mock()
            mock_dialog.return_value = mock_dialog_instance
            mock_dialog_instance.exec_.return_value = True
            mock_dialog_instance.get_images.return_value = (
                self.test_image, self.test_image)
            with patch.object(self.model, 'apply_style_transfer') as mock_transfer:
                mock_transfer.return_value = self.test_image
                self.controller._handle_style_transfer()
        self.assertIsNotNone(self.model.current_image)
        self.assertEqual(len(self.model.history), 2)  # Original + style transfer

    def test_super_resolution_workflow(self):
        """Tests the super resolution workflow"""
        with patch('cv2.imread') as mock_imread:
            mock_imread.return_value = self.test_image
            self.controller.select_image()
        with patch('src.views.dialogs.super_resolution_dialog.SuperResolutionDialog') as mock_dialog:
            mock_dialog_instance = Mock()
            mock_dialog.return_value = mock_dialog_instance
            mock_dialog_instance.exec_.return_value = True
            mock_dialog_instance.get_scale_factor.return_value = 2
            with patch.object(self.model, 'super_resolution') as mock_sr:
                mock_sr.return_value = np.zeros((448, 448, 3))
                self.controller._handle_super_resolution()
        self.assertEqual(self.model.current_image.shape[:2], (448, 448))