"""
This module implements a dialog window for configuring the blur effect.

The `BlurDialog` class contains the functions needed for interactively adjusting
the blur intensity, such as creating real-time previews and applying settings.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                             QSlider, QVBoxLayout, QWidget)


class BlurDialog(QDialog):
    """
    Blur effect settings dialog window.

    This class enables interactive adjustment of Gaussian blur intensity
    with real-time preview.

    Attributes:
        MAX_PREVIEW_WIDTH (int): Maximum width of preview image
        MIN_KERNEL_SIZE (int): Minimum kernel size (1)
        MAX_KERNEL_SIZE (int): Maximum kernel size (20)
        DEFAULT_KERNEL_SIZE (int): Default kernel size (1)
    """

    # UI constants
    MAX_PREVIEW_WIDTH = 300
    MIN_KERNEL_SIZE = 1
    MAX_KERNEL_SIZE = 20
    DEFAULT_KERNEL_SIZE = 1

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
        self.setWindowTitle("Blur")
        self.setModal(True)

        # Store original image and create preview image
        self.original_image = image
        self.preview_image = self._resize_for_preview(image)

        self._setup_ui()
        self._create_preview()
        self._setup_connections()

        # Update preview with default value
        self._update_preview_image(self.DEFAULT_KERNEL_SIZE)

        logging.debug("BlurDialog initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)

        # Preview image
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)

        # Blur intensity slider and label
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("Blur intensity:"))
        self.kernel_size_label = QLabel(f"{self.DEFAULT_KERNEL_SIZE}")
        blur_layout.addWidget(self.kernel_size_label)

        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(self.MIN_KERNEL_SIZE, self.MAX_KERNEL_SIZE)
        self.blur_slider.setValue(self.DEFAULT_KERNEL_SIZE)
        layout.addLayout(blur_layout)
        layout.addWidget(self.blur_slider)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _setup_connections(self) -> None:
        """Set up the signal-slot connections."""
        self.blur_slider.valueChanged.connect(self._on_slider_change)

    def _create_preview(self) -> None:
        """Create the preview image."""
        try:
            self._update_preview_image(self.DEFAULT_KERNEL_SIZE)
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

    def _on_slider_change(self) -> None:
        """Handle slider value changes."""
        try:
            kernel_size = self._get_valid_kernel_size()
            self.kernel_size_label.setText(f"{kernel_size}")
            self._update_preview_image(kernel_size)
        except Exception as e:
            logging.error(f"Error processing slider value: {str(e)}")

    def _get_valid_kernel_size(self) -> int:
        """
        Return valid kernel size (always an odd number).

        Returns:
            int: The valid kernel size
        """
        kernel_size = self.blur_slider.value()
        # Gaussian blur kernel size must be odd
        return kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    def _update_preview_image(self, kernel_size: int) -> None:
        """
        Update the preview image.

        Args:
            kernel_size: The Gaussian kernel size
        """
        try:
            blurred_image = self._apply_gaussian_blur(
                self.preview_image, kernel_size)
            self._display_image(blurred_image)
            logging.debug(f"Preview updated with kernel size {kernel_size}")
        except Exception as e:
            logging.error(f"Error updating preview: {str(e)}")

    def _apply_gaussian_blur(
            self,
            image: np.ndarray,
            kernel_size: int) -> np.ndarray:
        """
        Apply Gaussian blur.

        Args:
            image: The image to modify
            kernel_size: The Gaussian kernel size

        Returns:
            np.ndarray: The blurred image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def _display_image(self, image: np.ndarray) -> None:
        """
        Display the image on the interface.

        Args:
            image: The image to display
        """
        try:
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

    def get_blur_intensity(self) -> int:
        """
        Return the set blur value.

        Returns:
            int: The set kernel size
        """
        return self._get_valid_kernel_size()