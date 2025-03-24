"""
This module implements a dialog window for adjusting saturation.

The `SaturationDialog` class allows interactive adjustment of image saturation
with real-time preview. For grayscale images, it notifies the user.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout,
                             QLabel, QMessageBox, QSlider, QVBoxLayout,
                             QWidget)


class SaturationDialog(QDialog):
    """
    Saturation settings dialog window.

    This class enables interactive adjustment of image saturation
    with real-time preview. Works only on color images,
    and displays a warning for grayscale images.

    Attributes:
        MAX_PREVIEW_WIDTH (int): Maximum width of preview image
        MIN_SATURATION (float): Minimum saturation value (0.1)
        MAX_SATURATION (float): Maximum saturation value (2.0)
        DEFAULT_SATURATION (float): Default saturation value (1.0)
    """

    # UI constants
    MAX_PREVIEW_WIDTH = 300
    MIN_SATURATION = 0.1
    MAX_SATURATION = 2.0
    DEFAULT_SATURATION = 1.0
    SLIDER_MULTIPLIER = 10  # Multiplier between slider values and saturation values

    def __init__(
            self,
            image: np.ndarray,
            parent: Optional[QWidget] = None) -> None:
        """
        Initialize the dialog window.

        Args:
            image: The original image as a numpy array (in BGR format)
            parent: The parent widget (optional)

        Raises:
            ValueError: If the image has an invalid format
        """
        super().__init__(parent)
        self.setWindowTitle("Adjust Saturation")
        self.setModal(True)

        if image is None:
            raise ValueError("Input image cannot be None")

        # Store original image and create preview image
        self.original_image = image
        self.preview_image = self._resize_for_preview(image)

        # Check if grayscale image
        if self._is_grayscale(image):
            QMessageBox.information(
                self,
                "Black and White Image",
                "This image is already black and white, so saturation cannot be modified.")
            self.reject()
            return

        self._setup_ui()
        self._create_preview()
        self._setup_connections()

        logging.debug("SaturationDialog initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)

        # Preview image
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)

        # Saturation settings group
        saturation_group = QGroupBox("Saturation Settings")
        saturation_layout = QVBoxLayout()

        # Saturation slider and label
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Saturation:"))
        self.saturation_value_label = QLabel(f"{self.DEFAULT_SATURATION:.2f}")
        slider_layout.addWidget(self.saturation_value_label)

        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(
            int(self.MIN_SATURATION * self.SLIDER_MULTIPLIER),
            int(self.MAX_SATURATION * self.SLIDER_MULTIPLIER)
        )
        self.saturation_slider.setValue(
            int(self.DEFAULT_SATURATION * self.SLIDER_MULTIPLIER))
        self.saturation_slider.setToolTip(
            "Drag the slider to modify saturation.\n"
            "1.0 = original colors\n"
            "< 1.0 = less saturated colors\n"
            "> 1.0 = more vibrant colors"
        )

        saturation_layout.addLayout(slider_layout)
        saturation_layout.addWidget(self.saturation_slider)

        # Information label
        info_label = QLabel(
            "Adjusting saturation affects the intensity of colors in the image."
        )
        info_label.setWordWrap(True)
        saturation_layout.addWidget(info_label)

        saturation_group.setLayout(saturation_layout)
        layout.addWidget(saturation_group)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _setup_connections(self) -> None:
        """Set up the signal-slot connections."""
        self.saturation_slider.valueChanged.connect(self._on_slider_change)

    def _create_preview(self) -> None:
        """Create the preview image."""
        try:
            self._update_preview_image(self.DEFAULT_SATURATION)
            logging.debug("Preview image successfully created")
        except Exception as e:
            logging.error(f"Error creating preview image: {str(e)}")
            self.reject()

    def _is_grayscale(self, image: np.ndarray) -> bool:
        """
        Check if the image is grayscale.

        Args:
            image: The image to check

        Returns:
            bool: True if the image is grayscale, False if it's color
        """
        return len(image.shape) < 3 or image.shape[2] == 1

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
            saturation = self.saturation_slider.value() / self.SLIDER_MULTIPLIER
            self.saturation_value_label.setText(f"{saturation:.2f}")
            self._update_preview_image(saturation)
            logging.debug(f"Saturation modified: {saturation}")
        except Exception as e:
            logging.error(f"Error modifying saturation: {str(e)}")

    def _update_preview_image(self, saturation: float) -> None:
        """
        Update the preview image.

        Args:
            saturation: The saturation value
        """
        try:
            saturated_image = self._apply_saturation(
                self.preview_image, saturation)
            self._display_image(saturated_image)
        except Exception as e:
            logging.error(f"Error updating preview: {str(e)}")

    def _apply_saturation(
            self,
            image: np.ndarray,
            saturation: float) -> np.ndarray:
        """
        Apply saturation modification.

        Args:
            image: The image to modify
            saturation: The saturation value

        Returns:
            np.ndarray: The image with modified saturation
        """
        # Convert to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Modify saturation
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation, 0, 255)

        # Convert back to BGR color space
        return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

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

    def get_saturation_scale(self) -> float:
        """
        Return the set saturation value.

        Returns:
            float: The set saturation value
        """
        return self.saturation_slider.value() / self.SLIDER_MULTIPLIER