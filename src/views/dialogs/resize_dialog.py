"""
This module implements a dialog window for configuring image resizing.

The `ResizeDialog` class allows the image size to be adjusted based on percentage or absolute values,
using different interpolation methods.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QRadioButton, QSpinBox, QVBoxLayout, QWidget)


class ResizeDialog(QDialog):
    """
    Image resizing settings dialog window.

    This class enables modification of image size based on percentage
    or absolute values, with different interpolation methods.

    Attributes:
        MAX_WIDTH (int): Maximum allowed width
        MAX_HEIGHT (int): Maximum allowed height
        MIN_PERCENTAGE (int): Minimum percentage value
        MAX_PERCENTAGE (int): Maximum percentage value
        DEFAULT_PERCENTAGE (int): Default percentage value
    """

    # Sizing limits
    MAX_WIDTH = 20000
    MAX_HEIGHT = 20000
    MIN_WIDTH = 10
    MIN_HEIGHT = 10
    MIN_PERCENTAGE = 1
    MAX_PERCENTAGE = 500
    DEFAULT_PERCENTAGE = 100

    # Interpolation methods
    INTERPOLATION_METHODS = {
        "Bicubic": "bicubic",
        "Nearest Neighbor": "nearest",
        "Bilinear": "bilinear"
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
        self.setWindowTitle("Resize")
        self.setModal(True)

        # Extract current dimensions from the image
        if image is None:
            raise ValueError("Input image cannot be None")

        height, width = image.shape[:2]
        self.current_width = width
        self.current_height = height
        self.aspect_ratio = width / height

        self._setup_ui()
        self._setup_connections()

        logging.debug(
            f"ResizeDialog initialized with image of size {width}x{height}")

    def _setup_ui(self) -> None:
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)

        # Display current size
        self.size_label = QLabel(
            f"Current size: {self.current_width} × {self.current_height} pixels")
        layout.addWidget(self.size_label)

        # Resize mode selector
        self._setup_resize_mode_group(layout)

        # Percentage resize settings
        self._setup_percentage_group(layout)

        # Absolute resize settings
        self._setup_absolute_group(layout)

        # Common settings
        self._setup_common_settings(layout)

        # OK and Cancel buttons
        self._setup_buttons(layout)

        self.setLayout(layout)
        self._update_input_state()

    def _setup_resize_mode_group(self, layout: QVBoxLayout) -> None:
        """
        Set up the resize mode selection group.

        Args:
            layout: The main layout
        """
        mode_group = QGroupBox("Resize Mode")
        mode_layout = QVBoxLayout()

        self.percentage_radio = QRadioButton("By Percentage")
        self.absolute_radio = QRadioButton("By Absolute Size")
        self.percentage_radio.setChecked(True)

        mode_layout.addWidget(self.percentage_radio)
        mode_layout.addWidget(self.absolute_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

    def _setup_percentage_group(self, layout: QVBoxLayout) -> None:
        """
        Set up the percentage resize group.

        Args:
            layout: The main layout
        """
        percentage_group = QGroupBox("Percentage Size")
        percentage_layout = QHBoxLayout()

        self.percentage_input = QSpinBox()
        self.percentage_input.setRange(
            self.MIN_PERCENTAGE, self.MAX_PERCENTAGE)
        self.percentage_input.setValue(self.DEFAULT_PERCENTAGE)
        self.percentage_input.setSuffix("%")

        percentage_layout.addWidget(QLabel("Scale:"))
        percentage_layout.addWidget(self.percentage_input)
        percentage_group.setLayout(percentage_layout)
        layout.addWidget(percentage_group)

    def _setup_absolute_group(self, layout: QVBoxLayout) -> None:
        """
        Set up the absolute resize group.

        Args:
            layout: The main layout
        """
        absolute_group = QGroupBox("Absolute Size")
        absolute_layout = QVBoxLayout()

        # Width setting
        width_layout = QHBoxLayout()
        self.width_input = QSpinBox()
        self.width_input.setRange(self.MIN_WIDTH, self.MAX_WIDTH)
        self.width_input.setValue(self.current_width)
        width_layout.addWidget(QLabel("Width:"))
        width_layout.addWidget(self.width_input)
        width_layout.addWidget(QLabel("pixels"))

        # Height setting
        height_layout = QHBoxLayout()
        self.height_input = QSpinBox()
        self.height_input.setRange(self.MIN_HEIGHT, self.MAX_HEIGHT)
        self.height_input.setValue(self.current_height)
        height_layout.addWidget(QLabel("Height:"))
        height_layout.addWidget(self.height_input)
        height_layout.addWidget(QLabel("pixels"))

        absolute_layout.addLayout(width_layout)
        absolute_layout.addLayout(height_layout)

        # Lock aspect ratio
        self.aspect_ratio_checkbox = QCheckBox("Maintain aspect ratio")
        self.aspect_ratio_checkbox.setChecked(True)
        absolute_layout.addWidget(self.aspect_ratio_checkbox)

        absolute_group.setLayout(absolute_layout)
        layout.addWidget(absolute_group)

    def _setup_common_settings(self, layout: QVBoxLayout) -> None:
        """
        Set up the common settings.

        Args:
            layout: The main layout
        """
        common_group = QGroupBox("Interpolation Settings")
        common_layout = QHBoxLayout()

        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(self.INTERPOLATION_METHODS.keys())
        self.interpolation_combo.setToolTip(
            "Select the interpolation method used during resizing"
        )

        common_layout.addWidget(QLabel("Interpolation:"))
        common_layout.addWidget(self.interpolation_combo)
        common_group.setLayout(common_layout)
        layout.addWidget(common_group)

    def _setup_buttons(self, layout: QVBoxLayout) -> None:
        """
        Set up the dialog buttons.

        Args:
            layout: The main layout
        """
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.validate_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _setup_connections(self) -> None:
        """Set up the signal-slot connections."""
        self.percentage_radio.toggled.connect(self._update_input_state)
        self.width_input.valueChanged.connect(
            self._update_height_based_on_width)
        self.height_input.valueChanged.connect(
            self._update_width_based_on_height)
        self.percentage_input.valueChanged.connect(self._update_size_preview)

    def _update_input_state(self) -> None:
        """Update the input field states based on the selected mode."""
        is_percentage = self.percentage_radio.isChecked()

        # Enable/disable percentage input
        self.percentage_input.setEnabled(is_percentage)

        # Enable/disable absolute size input
        self.width_input.setEnabled(not is_percentage)
        self.height_input.setEnabled(not is_percentage)
        self.aspect_ratio_checkbox.setEnabled(not is_percentage)

    def _update_height_based_on_width(self) -> None:
        """Update the height based on width if aspect ratio is locked."""
        if (self.aspect_ratio_checkbox.isChecked() and
                not self.width_input.signalsBlocked()):
            try:
                self.height_input.blockSignals(True)
                new_height = int(self.width_input.value() / self.aspect_ratio)
                self.height_input.setValue(new_height)
            finally:
                self.height_input.blockSignals(False)

    def _update_width_based_on_height(self) -> None:
        """Update the width based on height if aspect ratio is locked."""
        if (self.aspect_ratio_checkbox.isChecked() and
                not self.height_input.signalsBlocked()):
            try:
                self.width_input.blockSignals(True)
                new_width = int(self.height_input.value() * self.aspect_ratio)
                self.width_input.setValue(new_width)
            finally:
                self.width_input.blockSignals(False)

    def _update_size_preview(self) -> None:
        """Update the size preview in percentage mode."""
        if self.percentage_radio.isChecked():
            percentage = self.percentage_input.value()
            new_width = int(self.current_width * (percentage / 100))
            new_height = int(self.current_height * (percentage / 100))
            self.size_label.setText(
                f"Current size: {self.current_width} × {self.current_height} pixels\n"
                f"New size: {new_width} × {new_height} pixels"
            )

    def validate_and_accept(self) -> None:
        """Validate the set values and accept the dialog."""
        try:
            if self.percentage_radio.isChecked():
                percentage = self.percentage_input.value()
                new_width = int(self.current_width * (percentage / 100))
                new_height = int(self.current_height * (percentage / 100))
            else:
                new_width = self.width_input.value()
                new_height = self.height_input.value()

            if new_width > self.MAX_WIDTH or new_height > self.MAX_HEIGHT:
                QMessageBox.warning(
                    self,
                    "Invalid Size",
                    f"The maximum allowed size is "
                    f"{self.MAX_WIDTH}×{self.MAX_HEIGHT} pixels."
                )
                return
            elif new_width < self.MIN_WIDTH or new_height < self.MIN_HEIGHT:
                QMessageBox.warning(
                    self,
                    "Invalid Size",
                    f"The minimum allowed size is "
                    f"{self.MIN_WIDTH}×{self.MIN_HEIGHT} pixels."
                )
                return

            self.accept()

        except Exception as e:
            logging.error(f"Error validating values: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"An error occurred while validating values: {str(e)}"
            )

    def get_values(self) -> Dict[str, Any]:
        """
        Return the set values.

        Returns:
            Dict[str, Any]: A dictionary of the set values
        """
        if self.percentage_radio.isChecked():
            return {
                "type": "percentage",
                "percentage": self.percentage_input.value(),
                "interpolation": self.INTERPOLATION_METHODS[
                    self.interpolation_combo.currentText()
                ]
            }
        else:
            return {
                "type": "absolute",
                "width": self.width_input.value(),
                "height": self.height_input.value(),
                "aspect_ratio": self.aspect_ratio_checkbox.isChecked(),
                "interpolation": self.INTERPOLATION_METHODS[
                    self.interpolation_combo.currentText()
                ]
            }