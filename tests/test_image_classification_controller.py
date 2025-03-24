import os
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PyQt5.QtWidgets import QApplication

from src.controllers.image_classification_controller import \
    ImageClassificationController
from src.views.image_classification_view import ImageClassificationView

app = QApplication(sys.argv)


class TestImageClassificationController(unittest.TestCase):
    def setUp(self):
        """Runs before each test, prepares the test environment"""
        self.view = Mock(spec=ImageClassificationView)
        self.view.parent.return_value = None

        self.qmessagebox_patcher = patch(
            'src.controllers.image_classification_controller.QMessageBox')
        self.mock_message_box = self.qmessagebox_patcher.start()
        self.addCleanup(self.qmessagebox_patcher.stop)

        # UI texts
        self.view.UI_TEXTS = {
            'no_image': "No image loaded",
            'select_image': "Select image",
            'select_folder': "Select folder",
            'auto_sort': "Automatic folder organization",
            'move': "Move",
            'copy': "Copy",
            'classify': "Start classification",
            'result_placeholder': "Classification result will appear here",
            'image_radio': "Classify individual image",
            'folder_radio': "Batch classify folder",
            'save_results': "Save results",
            'model_select': "Select model:",
            'top_n': "Number of Top-N predictions:",
            'add_custom_model': "Add custom model...",
            'model_info': "Model information",
            'model_type': "Type:"
        }

        # Model descriptions
        self.view.MODEL_DESCRIPTIONS = {
            "mobilenetv2": {
                "description": "Test model description",
                "type": "built-in",
                "link": "test_link"
            },
            "resnet50": {
                "description": "ResNet50 description",
                "type": "built-in",
                "link": "test_link"
            },
            "inceptionv3": {
                "description": "InceptionV3 description",
                "type": "built-in",
                "link": "test_link"
            }
        }

        # All buttons and containers
        self.view.select_output_dir_button = Mock()
        self.view.output_dir_display = Mock()
        self.view.save_results_button = Mock()
        self.view.classify_button = Mock()
        self.view.image_label = Mock()
        self.view.select_image_button = Mock()
        self.view.select_folder_button = Mock()
        self.view.auto_sort_checkbox = Mock()
        self.view.move_radio = Mock()
        self.view.copy_radio = Mock()
        self.view.image_radio = Mock()
        self.view.folder_radio = Mock()
        self.view.result_label = Mock()
        self.view.progress_bar = Mock()
        self.view.add_model_button = Mock()
        self.view.button_box = Mock()
        self.view.ok_button = Mock()
        self.view.cancel_button = Mock()

        # Scrollable area and all containers
        self.view.scroll_area = Mock()
        scroll_size = Mock()
        scroll_size.width.return_value = 800
        scroll_size.height.return_value = 600
        self.view.scroll_area.size.return_value = scroll_size
        self.view.thumbnail_grid = Mock()
        self.view.display_container = Mock()
        self.view.preview_container = Mock()
        self.view.content_preview = Mock()
        self.view.style_preview = Mock()
        self.view.preview_label = Mock()

        # Model selector and ALL information elements
        self.view.model_selector = Mock()
        self.view.top_n_spinbox = Mock()
        self.view.model_type_label = Mock()
        self.view.model_description_label = Mock()
        self.view.result_layout = Mock()
        self.view.info_label = Mock()
        self.view.size_label = Mock()
        self.view.content_preview_size = Mock()
        self.view.style_preview_size = Mock()
        self.view.preview_style_image = Mock()
        self.view.preview_content_image = Mock()

        # ALL layouts and lists
        self.view.history_list = Mock()
        self.view.function_list = Mock()
        self.view.result_list = Mock()
        self.view.main_layout = Mock()
        self.view.controls_layout = Mock()
        self.view.buttons_layout = Mock()
        self.view.preview_layout = Mock()
        self.view.image_layout = Mock()
        self.view.model_layout = Mock()
        self.view.options_layout = Mock()

        # Setting ALL return values
        self.view.image_radio.isChecked.return_value = True
        self.view.move_radio.isEnabled.return_value = True
        self.view.copy_radio.isEnabled.return_value = True
        self.view.top_n_spinbox.value.return_value = 3
        self.view.model_selector.currentText.return_value = "mobilenetv2"
        self.view.result_label.text.return_value = "Classification result will appear here"

        # ALL update and display methods
        self.view.update_image_path_display = Mock()
        self.view.update_folder_path_display = Mock()
        self.view.update_available_models = Mock()
        self.view.update_model_info = Mock()
        self.view.update_progress = Mock()
        self.view.update_image_display = Mock()
        self.view.update_preview = Mock()
        self.view.display_image = Mock()
        self.view.display_results = Mock()
        self.view.show_progress = Mock()

        # ALL dialog and message methods
        self.view.show_warning_message = Mock()
        self.view.show_error_message = Mock()
        self.view.show_info_message = Mock()
        self.view.show_add_model_dialog = Mock()
        self.view.show_confirmation_dialog = Mock()

        # ALL enable/disable methods
        self.view.enable_editing_controls = Mock()
        self.view.enable_all_controls = Mock()
        self.view.disable_all_controls = Mock()

        # ALL clear methods
        self.view.clear_all = Mock()
        self.view.clear_results = Mock()
        self.view.clear_preview = Mock()
        self.view.clear_history = Mock()

        self.controller = ImageClassificationController(self.view)
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_initialization(self):
        """Tests correct controller initialization"""
        self.assertIsNotNone(self.controller.model)
        self.assertIsNotNone(self.controller.view)
        self.assertIsNone(self.controller.selected_image)
        self.assertIsNone(self.controller.selected_folder)

    @patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName')
    def test_select_image_valid(self, mock_dialog):
        """Tests valid image selection"""
        mock_dialog.return_value = ('test.jpg', '')
        with patch('cv2.imread') as mock_imread:
            mock_imread.return_value = self.test_image
            self.controller.select_image()
            self.assertEqual(self.controller.selected_image, 'test.jpg')
            self.view.update_image_path_display.assert_called_with('test.jpg')

    def test_select_image_invalid(self):
        """Tests invalid image selection"""
        with patch('PyQt5.QtWidgets.QFileDialog.getOpenFileName',
                   return_value=('test.jpg', '')):
            with patch('cv2.imread', return_value=None):
                self.controller.select_image()
                self.mock_message_box.warning.assert_called_once_with(
                    self.view, "Error", "Failed to load image."
                )

    def test_enable_image_selection(self):
        """Tests enabling image selection"""
        self.controller.enable_image_selection()
        self.view.select_image_button.setEnabled.assert_called_with(True)
        self.view.select_folder_button.setEnabled.assert_called_with(False)
        self.assertIsNone(self.controller.selected_folder)
        self.view.update_folder_path_display.assert_called_with("")
        self.view.image_label.setText.assert_called_with("No image loaded")

    def test_enable_folder_selection(self):
        """Tests enabling folder selection"""
        self.controller.enable_folder_selection()
        self.view.select_image_button.setEnabled.assert_called_with(False)
        self.view.select_folder_button.setEnabled.assert_called_with(True)
        self.assertIsNone(self.controller.selected_image)
        self.view.update_image_path_display.assert_called_with("")

    def test_toggle_move_copy_options(self):
        """Tests toggling move/copy options"""
        self.controller.toggle_move_copy_options()
        enabled = self.view.auto_sort_checkbox.isChecked()
        self.view.move_radio.setEnabled.assert_called_with(enabled)
        self.view.copy_radio.setEnabled.assert_called_with(enabled)
        self.view.select_output_dir_button.setEnabled.assert_called_with(
            enabled)
        self.view.output_dir_display.setEnabled.assert_called_with(enabled)

    def test_select_folder(self):
        """Tests folder selection"""
        with patch('PyQt5.QtWidgets.QFileDialog.getExistingDirectory', return_value='test_folder'):
            with patch('os.listdir', return_value=['test1.jpg', 'test2.jpg']):
                self.controller.select_folder()
                self.assertEqual(
                    self.controller.selected_folder,
                    'test_folder')
                self.view.update_folder_path_display.assert_called_with(
                    'test_folder')

    def test_start_classification_no_selection(self):
        """Tests starting classification without selection"""
        self.controller.selected_image = None
        self.controller.selected_folder = None
        self.controller.start_classification()
        self.mock_message_box.warning.assert_called_once_with(
            self.view, "Warning",
            "Please select an image or folder."
        )

    def test_save_results_success(self):
        """Tests successful saving of results"""
        self.controller.batch_results = ['Result 1', 'Result 2']
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            with patch('PyQt5.QtWidgets.QFileDialog.getSaveFileName',
                       return_value=('results.txt', '')):
                self.controller.save_results_to_file()
                mock_file().write.assert_called()
                self.mock_message_box.information.assert_called_once_with(
                    self.view, "Save successful",
                    "Results were successfully saved."
                )

    def test_save_results_no_results(self):
        """Tests saving results with empty results"""
        self.controller.batch_results = []
        self.controller.save_results_to_file()
        self.mock_message_box.warning.assert_called_once_with(
            self.view, "No data to save", "No results to save."
        )

    @patch('os.path.exists')
    def test_create_class_directory(self, mock_exists):
        """Tests creating class folder"""
        mock_exists.return_value = False
        with patch('os.makedirs'):
            directory = self.controller._create_class_directory(
                '/base', 'test_class')
            self.assertEqual(
                os.path.normpath(directory),
                os.path.normpath('/base/test_class'))

    @patch('shutil.move')
    def test_move_image_to_class_folder(self, mock_move):
        """Tests moving image"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            self.controller.move_image_to_class_folder(
                'test.jpg', 'class1', '/base')
            mock_move.assert_called_once()

    @patch('shutil.copy')
    def test_copy_image_to_class_folder(self, mock_copy):
        """Tests copying image"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            self.controller.copy_image_to_class_folder(
                'test.jpg', 'class1', '/base')
            mock_copy.assert_called_once()

    def test_update_available_models(self):
        """Tests updating available models"""
        self.view.update_available_models.reset_mock()
        self.controller._update_available_models()
        self.view.update_available_models.assert_called_once()

    def test_display_image(self):
        """Tests displaying image"""
        self.controller.scale = 1.0
        self.assertIsNotNone(self.test_image)
        self.assertEqual(self.test_image.shape,
                         (100, 100, 3))
        self.controller.display_image(self.test_image)
        self.view.image_label.setPixmap.assert_called_once()

    @patch('cv2.resize')
    def test_display_image_with_scale(self, mock_resize):
        """Tests displaying image with scaling"""
        mock_resize.return_value = self.test_image
        self.controller.scale = 2.0
        self.controller.display_image(self.test_image)
        mock_resize.assert_called_once()

    @patch('PyQt5.QtWidgets.QMessageBox.warning')
    def test_handle_classification_result(self, mock_warning):
        """Tests handling classification result"""
        self.controller.total_images = 1
        self.controller.processed_images = 0
        result = ('test.jpg', [('class1', 0.8)])
        self.controller._handle_classification_result(result)
        self.assertEqual(len(self.controller.batch_results), 1)
        self.view.progress_bar.setValue.assert_called_with(100)

    def test_handle_classification_error(self):
        """Tests handling classification error"""
        self.controller.total_images = 1
        error_message = "Test error"
        self.controller._handle_classification_error(error_message)
        self.mock_message_box.warning.assert_not_called()
        self.view.progress_bar.setValue.assert_called()

    def test_on_model_changed(self):
        """Tests model switching"""
        self.controller._on_model_changed("mobilenetv2")
        self.view.model_type_label.setText.assert_called()
        self.view.model_description_label.setText.assert_called()


if __name__ == '__main__':
    unittest.main()
