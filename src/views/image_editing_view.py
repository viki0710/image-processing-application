"""
This module contains the PyQt5-based UI for the image editing view.
It allows for image display and execution of editing operations.
"""

import logging
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QPushButton, QScrollArea, QSplitter,
                             QVBoxLayout, QWidget)

from src.image_editing_functions import IMAGE_EDITING_FUNCTIONS


class ImageEditingView(QWidget):
    """
    The view for the image editing application.

    This class is responsible for displaying and managing the user interface.
    The interface consists of two main parts: left-side image display and right-side control panel.

    Attributes:
        SPLITTER_RATIO (float): Size ratio between left and right sides
        DEFAULT_WINDOW_SIZE (tuple): Default window size
        BORDER_STYLE (str): Default border style
        SPACING (int): General spacing size
    """

    # UI constants
    SPLITTER_RATIO = 0.5
    DEFAULT_WINDOW_SIZE = (1200, 800)
    SPACING = 10

    # UI texts
    UI_TEXTS = {
        'select_image': "Select Image",
        'save_image': "Save Image",
        'editing_functions': "Image Editing Functions:",
        'history': "Operation History:",
        'undo': "Undo",
        'redo': "Redo",
        'reset': "Reset",
        'no_image': "No image loaded"
    }

    def __init__(self) -> None:
        """Initializes the image editing view."""
        super().__init__()
        self.init_ui()
        self.setup_tooltips()
        self.populate_function_list()

        logging.debug("ImageEditingView initialized")

    def init_ui(self) -> None:
        """Initializes and arranges the UI elements."""
        main_layout = QHBoxLayout(self)
        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # Create left and right panels
        left_widget = self._create_left_panel()
        right_widget = self._create_right_panel()

        # Set up splitter
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self._setup_splitter_sizes()

    def _create_left_panel(self) -> QWidget:
        """
        Creates the left panel with image display.

        Returns:
            QWidget: The left panel widget
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        # layout.setContentsMargins(0, 0, 0, 0)

        # Scrollable image display area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setWidgetResizable(True)

        layout.addWidget(self.scroll_area)

        # Image display label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)

        # Set default text
        self.image_label.setText(self.UI_TEXTS['no_image'])

        return widget

    def _create_right_panel(self) -> QWidget:
        """
        Creates the right panel with control elements.

        Returns:
            QWidget: The right panel widget
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # File operations section
        layout.addLayout(self._create_file_operations_section())

        # Separator
        layout.addWidget(self._create_separator())

        # Editing functions list
        layout.addWidget(QLabel(self.UI_TEXTS['editing_functions']))
        self.function_list = QListWidget()
        layout.addWidget(self.function_list)

        # History list
        layout.addWidget(QLabel(self.UI_TEXTS['history']))
        self.history_list = QListWidget()
        layout.addWidget(self.history_list)

        # Operation buttons
        layout.addLayout(self._create_operation_buttons())

        return widget

    def _create_file_operations_section(self) -> QVBoxLayout:
        """
        Creates the file operations section.

        Returns:
            QVBoxLayout: The file operations section layout
        """
        layout = QVBoxLayout()

        # Image selection row
        file_select_layout = QHBoxLayout()
        self.select_image_button = QPushButton(self.UI_TEXTS['select_image'])
        self.image_path_display = QLineEdit()
        self.image_path_display.setReadOnly(True)

        file_select_layout.addWidget(self.select_image_button)
        file_select_layout.addWidget(self.image_path_display)
        layout.addLayout(file_select_layout)

        # Save button
        self.save_button = QPushButton(self.UI_TEXTS['save_image'])
        layout.addWidget(self.save_button)

        return layout

    def _create_operation_buttons(self) -> QHBoxLayout:
        """
        Creates the operation buttons.

        Returns:
            QHBoxLayout: The operation buttons layout
        """
        layout = QHBoxLayout()

        self.undo_button = QPushButton(self.UI_TEXTS['undo'])
        self.redo_button = QPushButton(self.UI_TEXTS['redo'])
        self.reset_button = QPushButton(self.UI_TEXTS['reset'])

        layout.addWidget(self.undo_button)
        layout.addWidget(self.redo_button)
        layout.addWidget(self.reset_button)

        return layout

    def _create_separator(self) -> QFrame:
        """
        Creates a horizontal separator line.

        Returns:
            QFrame: The separator line widget
        """
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator

    def _setup_splitter_sizes(self) -> None:
        """Sets the initial size ratios of the splitter."""
        total_width = self.width()
        self.splitter.setSizes([
            int(total_width * self.SPLITTER_RATIO),
            int(total_width * (1 - self.SPLITTER_RATIO))
        ])

    def setup_tooltips(self) -> None:
        """Sets tooltip texts for UI elements."""
        self.select_image_button.setToolTip(
            "Select an image for editing")
        self.save_button.setToolTip("Save the edited image")
        self.undo_button.setToolTip("Undo last operation")
        self.redo_button.setToolTip("Redo undone operation")
        self.reset_button.setToolTip("Reset to original image")
        self.function_list.setToolTip("Choose an editing operation")
        self.history_list.setToolTip("List of performed operations")

    def populate_function_list(self) -> None:
        """
        Loads the list of editing functions to display in the user interface.

        Reads functions from the IMAGE_EDITING_FUNCTIONS list and populates
        the list view with function names.
        """
        try:
            self.function_list.clear()
            for func in IMAGE_EDITING_FUNCTIONS:
                self.function_list.addItem(func["name"])
            logging.debug("Function list successfully populated")
        except Exception as e:
            logging.error(f"Error populating function list: {str(e)}")

    def update_image_path_display(self, path: str) -> None:
        """
        Updates the display of the image path in the user interface.

        Args:
            path: Full path of the selected image.
        """
        self.image_path_display.setText(path)
        logging.debug(f"Image path updated: {path}")

    def update_image_display(self, pixmap: Optional[QPixmap]) -> None:
        """
        Updates the image display.

        Args:
            pixmap: The QPixmap object of the image to display or None
        """
        try:
            if pixmap is None:
                self.image_label.setText(self.UI_TEXTS['no_image'])
                return

            scaled_pixmap = pixmap.scaled(
                pixmap.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())
            logging.debug("Image display successfully updated")
        except Exception as e:
            logging.error(f"Error updating image display: {str(e)}")
            self.image_label.setText("Error displaying image")

    def enable_editing_controls(self, enabled: bool = True) -> None:
        """
        Enables or disables the editing controls.

        Args:
            enabled: True to enable controls, False to disable
        """
        self.save_button.setEnabled(enabled)
        self.function_list.setEnabled(enabled)
        self.undo_button.setEnabled(enabled)
        self.redo_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)

    def resizeEvent(self, event) -> None:
        """
        Handles window resizing.

        Args:
            event: The resize event
        """
        super().resizeEvent(event)
        self._setup_splitter_sizes()