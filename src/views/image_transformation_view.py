"""
This module contains the PyQt5-based UI for the image transformation view.
It allows for image display and handling of various transformation functions.
"""

import logging
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QProgressBar, QPushButton,
                             QScrollArea, QSplitter, QVBoxLayout, QWidget)


class ImageTransformationView(QWidget):
    """
    Image transformation module view.

    This class is responsible for displaying the image transformation interface.
    The interface consists of two main parts: left-side image display and right-side
    control panel with transformation functions.

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
        'transformation_functions': "Transformation Functions:",
        'no_image': "No image loaded"
    }

    def __init__(self) -> None:
        """Initializes the image transformation view."""
        super().__init__()
        self.init_ui()
        self.setup_tooltips()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        logging.debug("ImageTransformationView initialized")

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

    def _create_operation_buttons(self) -> QHBoxLayout:
        """Creates operation buttons."""
        layout = QHBoxLayout()

        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.reset_button = QPushButton("Reset")

        # Set tooltips
        self.undo_button.setToolTip("Undo last operation")
        self.redo_button.setToolTip("Redo undone operation")
        self.reset_button.setToolTip("Reset to original image")

        layout.addWidget(self.undo_button)
        layout.addWidget(self.redo_button)
        layout.addWidget(self.reset_button)

        return layout

    def _create_left_panel(self) -> QWidget:
        """
        Creates the left panel with image display.

        Returns:
            QWidget: The left panel widget.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
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
            QWidget: The right panel widget.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # File operations section
        layout.addLayout(self._create_file_operations_section())

        # Separator
        layout.addWidget(self._create_separator())

        # Transformation functions list
        layout.addWidget(QLabel(self.UI_TEXTS['transformation_functions']))
        self.function_list = QListWidget()
        layout.addWidget(self.function_list)

        # Operation history list
        layout.addWidget(QLabel("Operation History:"))
        self.history_list = QListWidget()
        layout.addWidget(self.history_list)

        # Operation buttons section
        operation_layout = QHBoxLayout()

        # Create and set up buttons
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.reset_button = QPushButton("Reset")

        # Set tooltips
        self.undo_button.setToolTip("Undo last operation")
        self.redo_button.setToolTip("Redo undone operation")
        self.reset_button.setToolTip("Reset to original image")

        # Initially disable buttons
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        self.reset_button.setEnabled(False)

        # Add buttons to layout
        operation_layout.addWidget(self.undo_button)
        operation_layout.addWidget(self.redo_button)
        operation_layout.addWidget(self.reset_button)

        layout.addLayout(operation_layout)

        return widget

    def _create_file_operations_section(self) -> QHBoxLayout:
        """
        Creates the file operations section.

        Returns:
            QHBoxLayout: The file operations section layout.
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

    def _create_separator(self) -> QFrame:
        """
        Creates a horizontal separator line.

        Returns:
            QFrame: The separator line widget.
        """
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator

    def _create_info_panel(self) -> QWidget:
        """
        Creates the information panel.

        Returns:
            QWidget: The information panel widget.
        """
        info_panel = QFrame()
        info_panel.setFrameShape(QFrame.StyledPanel)
        info_layout = QVBoxLayout(info_panel)

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        info_layout.addWidget(self.info_label)

        return info_panel

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
            "Select an image for transformation")
        self.save_button.setToolTip("Save the transformed image")
        self.function_list.setToolTip("Select a transformation function")

    def update_image_path_display(self, path: str) -> None:
        """
        Updates the path display of the selected image.

        Args:
            path: The path of the selected image.
        """
        self.image_path_display.setText(path)
        logging.debug(f"Image path updated: {path}")

    def update_image_display(self, pixmap: Optional[QPixmap]) -> None:
        """
        Updates the image display.

        Args:
            pixmap: The QPixmap object of the image to display or None.
        """
        try:
            if pixmap is None:
                self.image_label.setText(self.UI_TEXTS['no_image'])
                return

            # Keep the image size, don't scale to scroll_area size
            self.image_label.setPixmap(pixmap)
            self.image_label.resize(pixmap.size())
            self.image_label.setAlignment(Qt.AlignCenter)

            logging.debug("Image display successfully updated")

        except Exception as e:
            logging.error(f"Error updating image display: {str(e)}")
            self.image_label.setText("Error displaying image")

    def enable_editing_controls(self, enabled: bool = True) -> None:
        """
        Enables or disables editing controls.

        Args:
            enabled: True to enable controls, False to disable.
        """
        self.save_button.setEnabled(enabled)
        self.function_list.setEnabled(enabled)

    def update_info_text(self, text: str) -> None:
        """
        Updates the text in the information panel.

        Args:
            text: The new information text.
        """
        self.info_label.setText(text)

    def show_progress_bar(self) -> None:
        """Shows the progress bar."""
        # TODO: Implement progress bar display

    def update_progress(self, value: int) -> None:
        """
        Updates the progress bar value.

        Args:
            value: The new value (0-100).
        """
        # TODO: Implement progress bar update

    def resizeEvent(self, event) -> None:
        """
        Handles window resizing.

        Args:
            event: The resize event.
        """
        super().resizeEvent(event)
        self._setup_splitter_sizes()