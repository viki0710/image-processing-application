import sys
import unittest
from unittest.mock import Mock
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import QApplication
from src.views.image_transformation_view import ImageTransformationView

app = QApplication(sys.argv)

class TestImageTransformationView(unittest.TestCase):
    def setUp(self):
        self.view = ImageTransformationView()
        self.test_image = Mock()

    def test_initialization(self):
        """Tests view initialization"""
        self.assertIsNotNone(self.view.function_list)
        self.assertIsNotNone(self.view.history_list)

    def test_update_image_display(self):
        """Tests updating the image display"""
        self.view.update_image_display(None)
        self.assertEqual(
            self.view.image_label.text(),
            self.view.UI_TEXTS['no_image'])
        mock_label = Mock()
        self.view.image_label = mock_label
        mock_pixmap = Mock()
        self.view.update_image_display(mock_pixmap)
        mock_label.setPixmap.assert_called_once_with(mock_pixmap)

    def test_enable_editing_controls(self):
        """Tests enabling editing controls"""
        self.view.enable_editing_controls(True)
        self.assertTrue(self.view.save_button.isEnabled())
        self.assertTrue(self.view.function_list.isEnabled())
        self.view.enable_editing_controls(False)
        self.assertFalse(self.view.save_button.isEnabled())
        self.assertFalse(self.view.function_list.isEnabled())

    def test_update_image_path_display(self):
        """Tests updating the path display"""
        test_path = "test/path/image.jpg"
        self.view.update_image_path_display(test_path)
        self.assertEqual(self.view.image_path_display.text(), test_path)

    def test_resizeEvent(self):
        event = QResizeEvent(QSize(100, 100), QSize(50, 50))
        self.view.resizeEvent(event)
        self.assertNotEqual(self.view.splitter.sizes(), [0, 0])