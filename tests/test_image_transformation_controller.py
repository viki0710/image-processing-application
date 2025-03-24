import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PyQt5.QtWidgets import QApplication, QDialog

from src.controllers.image_transformation_controller import \
    ImageTransformationController

app = QApplication(sys.argv)


class TestImageTransformationController(unittest.TestCase):
    def setUp(self):
        self.qmessagebox_patcher = patch(
            'src.controllers.image_transformation_controller.QMessageBox')
        self.mock_message_box = self.qmessagebox_patcher.start()
        self.addCleanup(self.qmessagebox_patcher.stop)

        self.view = Mock()
        self.model = Mock()
        self.controller = ImageTransformationController(self.model, self.view)
        self.test_image = np.zeros((224, 224, 3), dtype=np.uint8)

    def test_handle_style_transfer(self):
        """Tests style transfer handling"""
        qmessagebox_patcher = patch(
            'src.controllers.image_transformation_controller.QMessageBox')
        self.mock_message_box = qmessagebox_patcher.start()
        self.addCleanup(qmessagebox_patcher.stop)

        dialog_patcher = patch(
            'src.controllers.image_transformation_controller.StyleTransferDialog')
        mock_dialog = dialog_patcher.start()
        self.addCleanup(dialog_patcher.stop)

        mock_dialog_instance = Mock()
        mock_dialog.return_value = mock_dialog_instance
        mock_dialog.Accepted = QDialog.Accepted
        mock_dialog_instance.exec_.return_value = QDialog.Accepted
        mock_dialog_instance.show_progress = Mock()
        mock_dialog_instance.get_images.return_value = (
            self.test_image, self.test_image)

        self.model.current_image = self.test_image

        with patch.object(self.model, 'apply_style_transfer') as mock_transfer:
            mock_transfer.return_value = self.test_image

            self.controller._handle_style_transfer()

            mock_dialog.assert_called_once()
            mock_dialog_instance.exec_.assert_called_once()
            mock_dialog_instance.get_images.assert_called_once()
            mock_dialog_instance.show_progress.assert_any_call(
                True)  # Check that it was called with True
            mock_dialog_instance.show_progress.assert_called_with(
                False)  # And with False
            mock_transfer.assert_called_once()

    def test_save_image(self):
        self.model.current_image = self.test_image
        with patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName', return_value=('test.jpg', '')):
            with patch('cv2.imwrite') as mock_imwrite:
                self.controller.save_image()
                mock_imwrite.assert_called_once()
                self.mock_message_box.information.assert_called_once_with(
                    self.view, "Save successful", "The image has been successfully saved"
                )

    def test_save_image_no_image(self):
        """Tests saving when no image is loaded"""
        self.model.current_image = None
        self.controller.save_image()
        self.mock_message_box.critical.assert_called_once_with(
            self.view, "Save error", "No image to save"
        )

    def test_select_image(self):
        """Tests image selection"""
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName', return_value=('test.jpg', '')):
            with patch('cv2.imread', return_value=self.test_image):
                scroll_area_size = Mock()
                scroll_area_size.width.return_value = 800
                scroll_area_size.height.return_value = 600
                self.view.scroll_area = Mock()
                self.view.scroll_area.size.return_value = scroll_area_size

                self.test_image = np.zeros((224, 224, 3), dtype=np.uint8)

                self.controller.select_image()
                self.assertEqual(self.model.set_current_image.call_count, 1)
                call_args = self.model.set_current_image.call_args[0][0]
                self.assertTrue(np.array_equal(call_args, self.test_image))

    def test_select_image_error(self):
        """Tests image loading error"""
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName',
                   return_value=('test.jpg', '')):
            with patch('cv2.imread', return_value=None):
                self.controller.select_image()
                self.mock_message_box.critical.assert_called_once_with(
                    self.view, "Loading error",
                    "Failed to load the image: Image loading failed"
                )

    def test_update_history_list(self):
        """Tests updating the history list"""
        self.model.history = [("Test Operation", self.test_image)]
        self.model.current_index = 0
        self.controller.update_history_list()
        self.view.history_list.clear.assert_called_once()
        self.view.history_list.addItem.assert_called_once()

    def test_update_button_states(self):
        """Tests updating button states"""
        self.model.current_image = self.test_image
        self.model.can_undo.return_value = True
        self.model.can_redo.return_value = False

        self.controller.update_button_states()
        self.view.undo_button.setEnabled.assert_called_with(True)
        self.view.redo_button.setEnabled.assert_called_with(False)

    def test_display_image(self):
        """Tests image display"""
        self.model.scale = 1.0
        self.controller.display_image(self.test_image)
        self.view.update_image_display.assert_called_once()