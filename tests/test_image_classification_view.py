import sys
import unittest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from src.views.image_classification_view import ImageClassificationView

app = QApplication(sys.argv)

class TestImageClassificationView(unittest.TestCase):
    def setUp(self):
        """Initializes the test environment"""
        self.view = ImageClassificationView()
        self.view.auto_sort_checkbox.setChecked(False)
        self.view.copy_radio.setChecked(False)
        self.view.move_radio.setChecked(False)
        self.view.MODEL_DESCRIPTIONS = {
            "mobilenetv2": {
                "description": "Test description",
                "type": "built-in",
                "link": "https://test.link"
            }
        }

    def test_initialization(self):
        """Tests the correct initialization of the view"""
        self.assertEqual(self.view.scale, 1.0)
        self.assertIsNone(self.view.current_image)
        # self.assertTrue(self.view.image_radio.isChecked())
        self.assertFalse(self.view.folder_radio.isChecked())
        self.assertEqual(
            self.view.top_n_spinbox.value(),
            self.view.DEFAULT_TOP_N)
        self.assertFalse(self.view.auto_sort_checkbox.isChecked())

    def test_update_model_info(self):
        """Tests updating the model information"""
        model_name = "mobilenetv2"
        model_info = self.view.MODEL_DESCRIPTIONS[model_name]
        self.view.update_model_info(model_name)
        self.assertEqual(self.view.model_type_label.text(), model_info["type"])
        expected_description = model_info["description"]
        if "link" in model_info:
            expected_description += f"<br><a href='{model_info['link']}'>More information</a>"
        self.assertEqual(
            self.view.model_description_label.text(),
            expected_description)

    def test_update_progress(self):
        """Tests updating the progress indicator"""
        self.view.update_progress(50)
        self.assertEqual(self.view.progress_bar.value(), 50)

    def test_toggle_move_copy_options_enabled(self):
        """Tests enabling the move/copy options"""
        self.view.auto_sort_checkbox.setChecked(True)
        self.view.auto_sort_checkbox.stateChanged.emit(Qt.Checked)
        self.assertTrue(self.view.move_radio.isEnabled())
        self.assertTrue(self.view.copy_radio.isEnabled())
        self.assertTrue(self.view.select_output_dir_button.isEnabled())
        self.assertTrue(self.view.output_dir_display.isEnabled())

    def test_toggle_move_copy_options_disabled(self):
        """Tests disabling the move/copy options"""
        self.view.auto_sort_checkbox.setChecked(False)
        self.assertFalse(self.view.move_radio.isEnabled())
        self.assertFalse(self.view.copy_radio.isEnabled())