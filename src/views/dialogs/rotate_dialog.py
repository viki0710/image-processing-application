"""
This module implements a dialog window for configuring image rotation.

The `RotateDialog` class allows interactive modification of the image angle,
with real-time preview and precise size information.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout,
                             QLabel, QSpinBox, QVBoxLayout, QWidget)


class RotateDialog(QDialog):
    """
    Image rotation settings dialog window.

    This class enables rotating an image at any arbitrary angle
    with real-time preview. During rotation, the image size is
    automatically adjusted to preserve all content.

    Attributes:
        MAX_PREVIEW_WIDTH (int): Maximum width of preview image
        MIN_ANGLE (int): Minimum rotation angle (-180 degrees)
        MAX_ANGLE (int): Maximum rotation angle (180 degrees)
        DEFAULT_ANGLE (int): Default rotation angle (0 degrees)
    """

    # UI constants
    MAX_PREVIEW_WIDTH = 300
    MIN_ANGLE = -180
    MAX_ANGLE = 180
    DEFAULT_ANGLE = 0

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
        self.setWindowTitle("Rotate Image")
        self.setModal(True)

        # Store original image and create preview image
        self.original_image = image
        self.preview_image = self._resize_for_preview(image)

        self._setup_ui()
        self._create_preview()
        self._setup_connections()

        logging.debug("RotateDialog initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)

        # Preview image
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_label)

        # Rotation settings group
        rotation_group = QGroupBox("Rotation Settings")
        rotation_layout = QVBoxLayout()

        # Rotation angle setting
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Rotation angle:"))

        self.rotation_input = QSpinBox()
        self.rotation_input.setRange(self.MIN_ANGLE, self.MAX_ANGLE)
        self.rotation_input.setValue(self.DEFAULT_ANGLE)
        self.rotation_input.setSuffix("°")
        self.rotation_input.setToolTip(
            "Enter rotation angle between -180° and +180°.\n"
            "Positive value: counter-clockwise direction\n"
            "Negative value: clockwise direction"
        )

        self.angle_display = QLabel(f"{self.DEFAULT_ANGLE}°")

        angle_layout.addWidget(self.rotation_input)
        angle_layout.addWidget(self.angle_display)

        rotation_layout.addLayout(angle_layout)
        rotation_group.setLayout(rotation_layout)
        layout.addWidget(rotation_group)

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
        """Set up the signal-slot connections."""
        self.rotation_input.valueChanged.connect(self._on_rotation_change)

    def _create_preview(self) -> None:
        """Create the preview image."""
        try:
            self._update_preview_image(self.DEFAULT_ANGLE)
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

    def _on_rotation_change(self) -> None:
        """Handle rotation angle changes."""
        try:
            angle = self.rotation_input.value()
            self.angle_display.setText(f"{angle}°")
            self._update_preview_image(angle)
        except Exception as e:
            logging.error(f"Error modifying rotation angle: {str(e)}")

    def _calculate_new_size(self, image: np.ndarray,
                            angle: float) -> Tuple[int, int]:
        """
        Calculate the new size of the rotated image.

        Args:
            image: The image to be rotated
            angle: The rotation angle

        Returns:
            Tuple[int, int]: The new width and height
        """
        height, width = image.shape[:2]

        # Convert angle to radians
        rangle = np.radians(angle)

        # Calculate new dimensions
        cos_a = abs(np.cos(rangle))
        sin_a = abs(np.sin(rangle))

        new_width = int(width * cos_a + height * sin_a)
        new_height = int(width * sin_a + height * cos_a)

        return new_width, new_height

    def _update_preview_image(self, angle: float) -> None:
        """
        Update the preview image and size information.

        Args:
            angle: The rotation angle
        """
        try:
            rotated_image = self._apply_rotation(self.preview_image, angle)
            self._display_image(rotated_image)

            # Use the original_image size instead of preview_image size here!
            # <- This change!
            orig_h, orig_w = self.original_image.shape[:2]
            new_w, new_h = self._calculate_new_size(
                self.original_image, angle)  # <- And this!

            self.size_label.setText(
                f"Original size: {orig_w}×{orig_h} pixels\n"
                f"Rotated size: {new_w}×{new_h} pixels"
            )

            logging.debug(f"Preview updated with angle {angle}°")
        except Exception as e:
            logging.error(f"Error updating preview: {str(e)}")

    def _apply_rotation(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Apply rotation to the image.

        Args:
            image: The image to rotate
            angle: The rotation angle

        Returns:
            np.ndarray: The rotated image
        """
        # Image center
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new dimensions
        new_width, new_height = self._calculate_new_size(image, angle)

        # Modify transformation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Execute rotation
        rotated_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return rotated_image

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

    def get_rotation_angle(self) -> float:
        """
        Return the set rotation angle.

        Returns:
            float: The set rotation angle
        """
        return float(self.rotation_input.value())