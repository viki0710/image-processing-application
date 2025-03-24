import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PyQt5.QtWidgets import QApplication
from src.controllers.image_editing_controller import ImageEditingController
from src.views.image_editing_view import ImageEditingView

app = QApplication(sys.argv)

class TestImageEditingIntegration(unittest.TestCase):
    def setUp(self):
        self.view = ImageEditingView()
        self.controller = ImageEditingController(self.view)
        # asymmetric test image where rotation is visible
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[20:40, 60:80] = 255

    def test_full_editing_workflow(self):
        with patch('cv2.imread') as mock_imread:
            mock_imread.return_value = self.test_image
            self.controller.open_image()
        with patch('src.views.dialogs.rotate_dialog.RotateDialog') as mock_dialog:
            mock_dialog_instance = Mock()
            mock_dialog.return_value = mock_dialog_instance
            mock_dialog_instance.exec_.return_value = True
            mock_dialog_instance.get_rotation_angle.return_value = 90
            mock_item = Mock()
            mock_item.text.return_value = "Rotation"
            self.controller.on_function_selected(mock_item)
        self.assertIsNotNone(self.controller.model.current_image)
        self.assertEqual(len(self.controller.model.history),
                         2)

    def test_history_navigation(self):
        with patch('cv2.imread') as mock_imread:
            mock_imread.return_value = self.test_image
            self.controller.open_image()
        with patch('src.views.dialogs.rotate_dialog.RotateDialog') as mock_dialog:
            mock_dialog_instance = Mock()
            mock_dialog.return_value = mock_dialog_instance
            mock_dialog_instance.exec_.return_value = True
            mock_dialog_instance.get_rotation_angle.return_value = 90
            mock_item = Mock()
            mock_item.text.return_value = "Rotation"
            self.controller.on_function_selected(mock_item)
        self.controller.undo()
        np.testing.assert_array_equal(
            self.controller.model.current_image,
            self.test_image
        )
        self.controller.redo()
        self.assertFalse(np.array_equal(
            self.controller.model.current_image,
            self.test_image
        ))