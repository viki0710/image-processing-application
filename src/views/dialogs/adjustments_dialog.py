"""
This module implements the dialog window used for adjusting contrast and brightness.

The `AdjustmentsDialog` class contains the necessary functions for this,
such as creating real-time previews and applying modifications.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                             QSlider, QVBoxLayout, QWidget)


class AdjustmentsDialog(QDialog):
    """
    Contrast and brightness settings dialog window.

    This class enables interactive modification of image contrast and brightness
    with real-time preview.

    Attributes:
        MAX_PREVIEW_WIDTH (int): Maximum width of preview image
        MIN_CONTRAST (float): Minimum contrast value (0.1)
        MAX_CONTRAST (float): Maximum contrast value (3.0)
        MIN_BRIGHTNESS (int): Minimum brightness value (-100)
        MAX_BRIGHTNESS (int): Maximum brightness value (100)
        DEFAULT_CONTRAST (float): Default contrast value (1.0)
        DEFAULT_BRIGHTNESS (int): Default brightness value (0)
    """

    # UI constants
    MAX_PREVIEW_WIDTH = 300
    MIN_CONTRAST = 0.1
    MAX_CONTRAST = 3.0
    MIN_BRIGHTNESS = -100
    MAX_BRIGHTNESS = 100
    DEFAULT_CONTRAST = 1.0
    DEFAULT_BRIGHTNESS = 0

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
        self.setWindowTitle("Adjust Contrast and Brightness")
        self.setModal(True)

        # Store original image and create preview image
        self.original_image = image
        self.preview_image = self._resize_for_preview(image)

        self._setup_ui()
        self._create_preview()
        self._setup_connections()

        # Update preview with default values
        self._update_preview_image(
            self.DEFAULT_CONTRAST,
            self.DEFAULT_BRIGHTNESS)

        logging.debug("AdjustmentsDialog initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)

        # Preview image
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)

        # Contrast slider
        contrast_layout = QHBoxLayout()
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.contrast_value_label = QLabel(f"{self.DEFAULT_CONTRAST:.1f}")
        contrast_layout.addWidget(self.contrast_value_label)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(1, 30)  # Values 0.1 - 3.0
        self.contrast_slider.setValue(10)  # 1.0 = default value
        layout.addLayout(contrast_layout)
        layout.addWidget(self.contrast_slider)

        # Brightness slider
        brightness_layout = QHBoxLayout()
        brightness_layout.addWidget(QLabel("Brightness:"))
        self.brightness_value_label = QLabel(f"{self.DEFAULT_BRIGHTNESS}")
        brightness_layout.addWidget(self.brightness_value_label)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(
            self.MIN_BRIGHTNESS, self.MAX_BRIGHTNESS)
        self.brightness_slider.setValue(self.DEFAULT_BRIGHTNESS)
        layout.addLayout(brightness_layout)
        layout.addWidget(self.brightness_slider)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _setup_connections(self) -> None:
        """Set up the signal-slot connections."""
        self.contrast_slider.valueChanged.connect(self._on_slider_change)
        self.brightness_slider.valueChanged.connect(self._on_slider_change)

    def _create_preview(self) -> None:
        """Create the preview image."""
        try:
            self._update_preview_image(
                self.DEFAULT_CONTRAST, self.DEFAULT_BRIGHTNESS)
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
            contrast = self.contrast_slider.value() / 10
            brightness = self.brightness_slider.value()

            # Display values
            self.contrast_value_label.setText(f"{contrast:.1f}")
            self.brightness_value_label.setText(f"{brightness}")

            self._update_preview_image(contrast, brightness)
        except Exception as e:
            logging.error(f"Error processing slider value: {str(e)}")

    def _update_preview_image(self, contrast: float, brightness: int) -> None:
        """
        Update the preview image.

        Args:
            contrast: Contrast value (0.1 - 3.0)
            brightness: Brightness value (-100 - 100)
        """
        try:
            adjusted_image = self._apply_contrast_brightness(
                self.preview_image, contrast, brightness
            )
            self._display_image(adjusted_image)
        except Exception as e:
            logging.error(f"Error updating preview: {str(e)}")

    def _apply_contrast_brightness(
        self, image: np.ndarray, contrast: float, brightness: int
    ) -> np.ndarray:
        """
        Apply contrast and brightness modifications.

        Args:
            image: The image to modify
            contrast: Contrast value
            brightness: Brightness value

        Returns:
            np.ndarray: The modified image
        """
        adjusted = np.clip(
            image *
            contrast +
            brightness,
            0,
            255).astype(
            np.uint8)
        return adjusted

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

    def get_values(self) -> Tuple[float, int]:
        """
        Return the set values.

        Returns:
            Tuple[float, int]: (contrast, brightness) values
        """
        contrast = self.contrast_slider.value() / 10
        brightness = self.brightness_slider.value()
        return contrast, brightness