"""
This module implements a dialog window for super-resolution (image resolution enhancement)
configuration.

The `SuperResolutionDialog` class allows interactive selection of image magnification ratio
with real-time preview and size information display.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                             QLabel, QProgressBar, QVBoxLayout, QWidget)


class SuperResolutionDialog(QDialog):
    """
    Super-resolution settings dialog.

    This class allows enhancing image resolution with different 
    magnification ratios, with real-time preview.
    """

    # UI constants
    MAX_PREVIEW_WIDTH = 300
    SUPPORTED_SCALES = ["2x", "4x"]

    def __init__(
            self,
            image: np.ndarray,
            parent: Optional[QWidget] = None) -> None:
        """
        Initialize the dialog window.

        Args:
            image: The original image as a numpy array (BGR format)
            parent: The parent widget (optional)
        """
        super().__init__(parent)
        self.setWindowTitle("Super Resolution")
        self.setModal(True)

        if image is None:
            raise ValueError("Input image cannot be None")

        # Store original image and create preview image
        self.original_image = image
        self.preview_image = self._resize_for_preview(image)

        self._setup_ui()
        self._create_preview()
        self._setup_connections()

        self._update_size_info()

        logging.debug("SuperResolutionDialog initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)

        # Preview image - with fixed size
        self.preview_label = QLabel(self)
        self.preview_label.setFixedSize(
            self.MAX_PREVIEW_WIDTH, self.MAX_PREVIEW_WIDTH)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet(
            "QLabel { background-color: white; border: 2px solid gray; }"
        )
        layout.addWidget(self.preview_label)

        # Magnification ratio selector
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Magnification level:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(self.SUPPORTED_SCALES)
        self.scale_combo.setCurrentText("2x")
        self.scale_combo.setToolTip(
            "Select the desired magnification ratio:\n"
            "2x = double size\n"
            "4x = quadruple size"
        )
        scale_layout.addWidget(self.scale_combo)
        layout.addLayout(scale_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Information label
        info = QLabel(
            "Super-resolution improves the image resolution. "
            "Due to the online API, the process may take a few seconds."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Size information
        self.size_label = QLabel()
        self.size_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.size_label)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _setup_connections(self) -> None:
        """Set up signal-slot connections."""
        self.scale_combo.currentTextChanged.connect(self._update_size_info)

    def _create_preview(self) -> None:
        """Create the preview image."""
        try:
            self._display_image(self.preview_image)
            logging.debug("Preview image successfully created")
        except Exception as e:
            logging.error(f"Error creating preview image: {str(e)}")
            self.reject()

    def _resize_for_preview(self, image: np.ndarray) -> np.ndarray:
        """Resize the image for preview, maintaining aspect ratio."""
        height, width = image.shape[:2]

        # Calculate aspect ratio
        aspect_ratio = width / height

        if width > height:
            new_width = min(self.MAX_PREVIEW_WIDTH, width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(self.MAX_PREVIEW_WIDTH, height)
            new_width = int(new_height * aspect_ratio)

        return cv2.resize(image, (new_width, new_height),
                          interpolation=cv2.INTER_AREA)

    def _display_image(self, image: np.ndarray) -> None:
        """Display the image on the preview_label."""
        try:
            if image is None:
                self.preview_label.setText("No image loaded")
                return

            # RGB conversion
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width

            # Create QImage from RGB image
            q_image = QImage(
                rgb_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888)

            # Create pixmap and scale it to the preview_label size
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.preview_label.setPixmap(scaled_pixmap)
            logging.debug(
                f"Image successfully displayed ({width}x{height} -> {
                    scaled_pixmap.width()}x{
                    scaled_pixmap.height()})")

        except Exception as e:
            logging.error(f"Error displaying image: {str(e)}")
            self.preview_label.setText("Error displaying image")

    def _update_size_info(self) -> None:
        """Update size information based on original image dimensions."""
        try:
            # Use the ORIGINAL image dimensions, not the preview
            current_height, current_width = self.original_image.shape[:2]
            scale = int(self.scale_combo.currentText().replace("x", ""))
            new_width = current_width * scale
            new_height = current_height * scale

            self.size_label.setText(
                f"Original size: {current_width}×{current_height} pixels\n"
                f"New size: {new_width}×{new_height} pixels"
            )
        except Exception as e:
            logging.error(f"Error updating size information: {str(e)}")

    def show_progress(self, show: bool = True) -> None:
        """
        Show or hide the progress bar.

        Args:
            show: True to show, False to hide
        """
        self.progress_bar.setVisible(show)

    def update_progress(self, value: int) -> None:
        """
        Update the progress bar value.

        Args:
            value: The new value (0-100)
        """
        self.progress_bar.setValue(value)

    def get_scale_factor(self) -> int:
        """
        Return the selected magnification ratio.

        Returns:
            int: The magnification ratio (2 or 4)
        """
        return int(self.scale_combo.currentText().replace("x", ""))