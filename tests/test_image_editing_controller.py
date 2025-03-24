import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PyQt5.QtWidgets import QApplication
from src.controllers.image_editing_controller import ImageEditingController

app = QApplication(sys.argv)

class TestImageEditingController(unittest.TestCase):
    def setUp(self):
        self.qmessagebox_patcher = patch(
            'src.controllers.image_editing_controller.QMessageBox')
        self.mock_message_box = self.qmessagebox_patcher.start()
        self.addCleanup(self.qmessagebox_patcher.stop)
        self.view = Mock()
        self.controller = ImageEditingController(self.view)
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        scroll_size = Mock()
        scroll_size.width.return_value = 800
        scroll_size.height.return_value = 600
        self.view.scroll_area = Mock()
        self.view.scroll_area.size.return_value = scroll_size

    def test_open_image(self):
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName') as mock_dialog:
            mock_dialog.return_value = ('test.jpg', '')
            with patch('cv2.imread') as mock_imread:
                mock_imread.return_value = self.test_image
                self.controller.open_image()
                self.view.update_image_path_display.assert_called_once()

    def test_save_image(self):
        self.controller.model.current_image = self.test_image
        with patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName', return_value=('test.jpg', '')):
            with patch('cv2.imwrite') as mock_imwrite:
                self.controller.save_image()
                mock_imwrite.assert_called_once()
                self.mock_message_box.information.assert_called_once_with(
                    self.view, "Successful save", "The image has been successfully saved."
                )

    def test_on_function_selected(self):
        self.controller.model.current_image = self.test_image
        self.controller.model.original_image = self.test_image
        mock_item = Mock()
        mock_item.text.return_value = "Rotation"
        with patch('src.controllers.image_editing_controller.RotateDialog') as mock_dialog:
            mock_dialog_instance = Mock()
            mock_dialog.return_value = mock_dialog_instance
            mock_dialog_instance.exec_.return_value = True
            mock_dialog_instance.get_rotation_angle.return_value = 90
            self.controller.on_function_selected(mock_item)
            mock_dialog.assert_called_once()