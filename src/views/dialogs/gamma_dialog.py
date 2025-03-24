"""
This module implements a dialog window for adjusting gamma correction.

The `GammaDialog` class allows interactive modification of the gamma value,
using real-time preview and slider-based adjustment.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                             QSlider, QVBoxLayout, QWidget)


class GammaDialog(QDialog):
    """
    Gamma correction settings dialog window.

    This class enables interactive adjustment of the image gamma value
    with real-time preview. Gamma correction allows for non-linear
    modification of the image brightness.

    Attributes:
        MAX_PREVIEW_WIDTH (int): Maximum width of preview image
        MIN_GAMMA (float): Minimum gamma value (0.1)
        MAX_GAMMA (float): Maximum gamma value (3.0)
        DEFAULT_GAMMA (float): Default gamma value (1.0)
    """

    # UI constants
    MAX_PREVIEW_WIDTH = 300
    MIN_GAMMA = 0.1
    MAX_GAMMA = 3.0
    DEFAULT_GAMMA = 1.0
    SLIDER_MULTIPLIER = 10  # Multiplier between slider values and gamma values

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
        self.setWindowTitle("Gamma Correction")
        self.setModal(True)

        # Store original image and create preview image
        self.original_image = image
        self.preview_image = self._resize_for_preview(image)

        # Initialize lookup table
        self.gamma_table = None

        self._setup_ui()
        self._create_preview()
        self._setup_connections()

        # Update preview with default value
        self._update_preview_image(self.DEFAULT_GAMMA)

        logging.debug("GammaDialog initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)

        # Preview image
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)

        # Gamma value label and slider
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma value:"))
        self.gamma_value_label = QLabel(f"{self.DEFAULT_GAMMA:.2f}")
        gamma_layout.addWidget(self.gamma_value_label)

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(
            int(self.MIN_GAMMA * self.SLIDER_MULTIPLIER),
            int(self.MAX_GAMMA * self.SLIDER_MULTIPLIER)
        )
        self.gamma_slider.setValue(
            int(self.DEFAULT_GAMMA * self.SLIDER_MULTIPLIER))

        # Add tooltip
        self.gamma_slider.setToolTip(
            "Drag the slider to modify the gamma value.\n"
            "1.0 = original image\n"
            "< 1.0 = emphasize darker tones\n"
            "> 1.0 = emphasize lighter tones"
        )

        layout.addLayout(gamma_layout)
        layout.addWidget(self.gamma_slider)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _setup_connections(self) -> None:
        """Set up the signal-slot connections."""
        self.gamma_slider.valueChanged.connect(self._on_slider_change)

    def _create_preview(self) -> None:
        """Create the preview image."""
        try:
            self._update_preview_image(self.DEFAULT_GAMMA)
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
            gamma = self.gamma_slider.value() / self.SLIDER_MULTIPLIER
            self.gamma_value_label.setText(f"{gamma:.2f}")
            self._update_preview_image(gamma)
            logging.debug(f"Gamma value modified: {gamma}")
        except Exception as e:
            logging.error(f"Error modifying gamma value: {str(e)}")

    def _create_gamma_table(self, gamma: float) -> np.ndarray:
        """
        Create the gamma correction lookup table.

        Args:
            gamma: The gamma value

        Returns:
            np.ndarray: The gamma correction lookup table
        """
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype("uint8")
        return table

    def _update_preview_image(self, gamma: float) -> None:
        """
        Update the preview image.

        Args:
            gamma: The gamma value
        """
        try:
            # Create or reuse lookup table
            self.gamma_table = self._create_gamma_table(gamma)

            # Apply gamma correction
            corrected_image = self._apply_gamma(self.preview_image)
            self._display_image(corrected_image)
        except Exception as e:
            logging.error(f"Error updating preview: {str(e)}")

    def _apply_gamma(self, image: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction to the image.

        Args:
            image: The image to modify

        Returns:
            np.ndarray: The gamma-corrected image
        """
        return cv2.LUT(image, self.gamma_table)

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

    def get_gamma_value(self) -> float:
        """
        Return the set gamma value.

        Returns:
            float: The set gamma value
        """
        return self.gamma_slider.value() / self.SLIDER_MULTIPLIER