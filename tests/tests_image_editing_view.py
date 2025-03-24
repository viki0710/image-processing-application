import sys
import unittest
from PyQt5.QtWidgets import QApplication
from src.views.image_editing_view import ImageEditingView

app = QApplication(sys.argv)

class TestImageEditingView(unittest.TestCase):
    """Test class for checking the functionality of the ImageEditingView."""
    
    def setUp(self):
        """Sets up the test environment by instantiating ImageEditingView."""
        self.view = ImageEditingView()
        
    def test_initialization(self):
        """
        Checks the initialization of ImageEditingView.
        Tests the default class-level constants and the creation of important elements.
        """
        self.assertEqual(self.view.SPLITTER_RATIO, 0.5)
        self.assertEqual(self.view.SPACING, 10)
        self.assertIsNotNone(self.view.function_list)
        self.assertIsNotNone(self.view.history_list)
        
    def test_update_image_path_display(self):
        """Checks the functionality of updating the path in the widget displaying the image path."""
        test_path = "test/path/image.jpg"
        self.view.update_image_path_display(test_path)
        self.assertEqual(self.view.image_path_display.text(), test_path)
        
    def test_enable_editing_controls(self):
        """Tests enabling and disabling the control buttons."""
        self.view.enable_editing_controls(True)
        self.assertTrue(self.view.save_button.isEnabled())
        self.assertTrue(self.view.function_list.isEnabled())
        self.assertTrue(self.view.undo_button.isEnabled())
        self.assertTrue(self.view.redo_button.isEnabled())
        self.assertTrue(self.view.reset_button.isEnabled())
        self.view.enable_editing_controls(False)