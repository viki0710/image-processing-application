"""
This module implements a dialog window for flipping images.

The `FlipDialog` class allows users to flip images
horizontally, vertically, or both with real-time preview.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QGroupBox,
                             QLabel, QVBoxLayout, QWidget)


class FlipDialog(QDialog):
    """
    Image flip settings dialog window.

    This class enables interactive configuration of horizontal and/or vertical
    image flipping with real-time preview.

    Attributes:
        MAX_PREVIEW_WIDTH (int): Maximum width of preview image
        FLIP_CODES (dict): OpenCV flip codes for different cases
    """

    # UI constants
    MAX_PREVIEW_WIDTH = 300

    # OpenCV flip codes
    FLIP_CODES = {
        'horizontal': 1,
        'vertical': 0,
        'both': -1
    }

    def __init__(
            self,
            image: np.ndarray,
            parent: Optional[QWidget] = None) -> None:
        """
        Initialize the dialog window.

        Args:
            image: The original image as a numpy array (in BGR format)
            parent: The parent widget (optional)
        """
        super().__init__(parent)
        self.setWindowTitle("Flip Image")
        self.setModal(True)

        # Store original image and create preview image
        self.original_image = image
        self.preview_image = self._resize_for_preview(image)

        self._setup_ui()
        self._create_preview()
        self._setup_connections()

        logging.debug("FlipDialog initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)

        # Preview image
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)

        # Group flip settings
        flip_group = QGroupBox("Flip Settings")
        flip_layout = QVBoxLayout()

        # Create checkboxes
        self.horizontal_flip_checkbox = QCheckBox("Horizontal Flip")
        self.vertical_flip_checkbox = QCheckBox("Vertical Flip")

        # Add tooltips
        self.horizontal_flip_checkbox.setToolTip(
            "Flips the image horizontally")
        self.vertical_flip_checkbox.setToolTip("Flips the image vertically")

        flip_layout.addWidget(self.horizontal_flip_checkbox)
        flip_layout.addWidget(self.vertical_flip_checkbox)
        flip_group.setLayout(flip_layout)
        layout.addWidget(flip_group)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _setup_connections(self) -> None:
        """Set up the signal-slot connections."""
        self.horizontal_flip_checkbox.stateChanged.connect(
            self._on_flip_change)
        self.vertical_flip_checkbox.stateChanged.connect(self._on_flip_change)

    def _create_preview(self) -> None:
        """Create the preview image."""
        try:
            self._update_preview_image()
            logging.debug("Preview image successfully created")
        except Exception as e:
            logging.error(f"Error creating preview image: {str(e)}")
            self.reject()

    def _resize_for_preview(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the image for preview.

        Args:
            image: The original image

        Returns:
            np.ndarray: The resized image
        """
        height, width = image.shape[:2]
        if width > self.MAX_PREVIEW_WIDTH:
            scale_factor = self.MAX_PREVIEW_WIDTH / width
            new_size = (self.MAX_PREVIEW_WIDTH, int(height * scale_factor))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image.copy()

    def _has_flip_selected(self) -> bool:
        """
        Check if any flip is selected.

        Returns:
            bool: True if at least one flip is selected
        """
        return (self.horizontal_flip_checkbox.isChecked() or
                self.vertical_flip_checkbox.isChecked())

    def _get_flip_code(self) -> Optional[int]:
        """
        Determine the appropriate flip code.

        Returns:
            Optional[int]: The flip code or None if no flip is selected
        """
        horizontal = self.horizontal_flip_checkbox.isChecked()
        vertical = self.vertical_flip_checkbox.isChecked()

        # If neither is selected, return None
        if not horizontal and not vertical:
            return None

        if horizontal and vertical:
            return self.FLIP_CODES['both']
        elif horizontal:
            return self.FLIP_CODES['horizontal']
        elif vertical:
            return self.FLIP_CODES['vertical']

        return None

    def _on_flip_change(self) -> None:
        """Handle changes in flip settings."""
        try:
            self._update_preview_image()
            logging.debug("Flip settings successfully updated")
        except Exception as e:
            logging.error(
                f"Error updating flip settings: {str(e)}")

    def _update_preview_image(self) -> None:
        """Update the preview image."""
        try:
            flip_code = self._get_flip_code()
            if flip_code is not None:
                flipped_image = self._apply_flip(self.preview_image, flip_code)
                self._display_image(flipped_image)
                logging.debug(f"Image flipped with code {flip_code}")
            else:
                # If no flip is selected, show the original image
                self._display_image(self.preview_image)
                logging.debug("Original image displayed (no flip)")
        except Exception as e:
            logging.error(f"Error updating preview: {str(e)}")

    def _apply_flip(self, image: np.ndarray, flip_code: int) -> np.ndarray:
        """
        Apply the flip to the image.

        Args:
            image: The image to modify
            flip_code: The type of flip (OpenCV flip code)

        Returns:
            np.ndarray: The flipped image
        """
        return cv2.flip(image, flip_code)

    def _display_image(self, image: np.ndarray) -> None:
        """
        Display the image on the interface.

        Args:
            image: The image to display
        """
        try:
            logging.debug(
                f"Displaying image: {'Flipped' if np.any(image != self.preview_image) else 'Original'}")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width

            q_image = QImage(
                rgb_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.preview_label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"Error displaying image: {str(e)}")

    def get_flip_code(self) -> Optional[int]:
        """
        Return the set flip code.

        Returns:
            Optional[int]: The flip code or None if no flip is selected
        """
        return self._get_flip_code()

    def accept(self) -> None:
        """Only accept the dialog if a flip is selected."""
        if self._has_flip_selected():
            super().accept()
        else:
            super().reject()