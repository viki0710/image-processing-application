"""
This module implements a dialog window for image sharpening.

The `SharpenDialog` class allows interactive adjustment of image sharpening intensity
with real-time preview, applying a sharpening kernel.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout,
                            QLabel, QSlider, QVBoxLayout, QWidget)


class SharpenDialog(QDialog):
   """
   Dialog window for image sharpening.

   This class allows interactive adjustment of image sharpening
   with real-time preview. The sharpening is performed using a special kernel
   that enhances edges in the image.

   Attributes:
       MAX_PREVIEW_WIDTH (int): Maximum width of the preview image
       MIN_INTENSITY (int): Minimum sharpening intensity (1)
       MAX_INTENSITY (int): Maximum sharpening intensity (10)
       DEFAULT_INTENSITY (int): Default sharpening intensity (1)
   """

   # UI constants
   MAX_PREVIEW_WIDTH = 300
   MIN_INTENSITY = 1
   MAX_INTENSITY = 10
   DEFAULT_INTENSITY = 1

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
           ValueError: If the image format is invalid
       """
       super().__init__(parent)
       self.setWindowTitle("Sharpening")
       self.setModal(True)

       if image is None:
           raise ValueError("Input image cannot be None")

       # Store original image and create preview image
       self.original_image = image
       self.preview_image = self._resize_for_preview(image)

       # Cache for storing kernels
       self.kernel_cache = {}

       self._setup_ui()
       self._create_preview()
       self._setup_connections()

       logging.debug("SharpenDialog initialized")

   def _setup_ui(self) -> None:
       """Set up the user interface elements."""
       layout = QVBoxLayout(self)

       # Preview image
       self.preview_label = QLabel(self)
       self.preview_label.setAlignment(Qt.AlignCenter)
       layout.addWidget(self.preview_label)

       # Sharpening settings group
       sharpen_group = QGroupBox("Sharpening Settings")
       sharpen_layout = QVBoxLayout()

       # Intensity slider and label
       intensity_layout = QHBoxLayout()
       intensity_layout.addWidget(QLabel("Sharpening intensity:"))
       self.intensity_value_label = QLabel(f"{self.DEFAULT_INTENSITY}")
       intensity_layout.addWidget(self.intensity_value_label)

       self.sharpen_slider = QSlider(Qt.Horizontal)
       self.sharpen_slider.setRange(self.MIN_INTENSITY, self.MAX_INTENSITY)
       self.sharpen_slider.setValue(self.DEFAULT_INTENSITY)
       self.sharpen_slider.setToolTip(
           "Drag the slider to adjust the sharpening intensity.\n"
           "1 = mild sharpening\n"
           "10 = strong sharpening"
       )

       sharpen_layout.addLayout(intensity_layout)
       sharpen_layout.addWidget(self.sharpen_slider)

       # Information label
       info_label = QLabel(
           "Sharpening enhances the edges and details in the image. "
           "At high values, noise may appear in the image."
       )
       info_label.setWordWrap(True)
       sharpen_layout.addWidget(info_label)

       sharpen_group.setLayout(sharpen_layout)
       layout.addWidget(sharpen_group)

       # OK and Cancel buttons
       button_box = QDialogButtonBox(
           QDialogButtonBox.Ok | QDialogButtonBox.Cancel
       )
       button_box.accepted.connect(self.accept)
       button_box.rejected.connect(self.reject)
       layout.addWidget(button_box)

   def _setup_connections(self) -> None:
       """Set up signal-slot connections."""
       self.sharpen_slider.valueChanged.connect(self._on_slider_change)

   def _create_preview(self) -> None:
       """Create the preview image."""
       try:
           self._update_preview_image(self.DEFAULT_INTENSITY)
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
           intensity = self.sharpen_slider.value()
           self.intensity_value_label.setText(str(intensity))
           self._update_preview_image(intensity)
           logging.debug(f"Sharpening intensity modified: {intensity}")
       except Exception as e:
           logging.error(
               f"Error modifying sharpening intensity: {str(e)}")

   def _create_sharpen_kernel(self, intensity: int) -> np.ndarray:
       """
       Create a sharpening kernel.

       Args:
           intensity: The sharpening intensity

       Returns:
           np.ndarray: The sharpening kernel
       """
       # Check if we already have a cached kernel
       if intensity in self.kernel_cache:
           return self.kernel_cache[intensity]

       # Create new kernel
       kernel = np.array([
           [-1, -1, -1],
           [-1, 8 + intensity, -1],
           [-1, -1, -1]
       ], dtype=np.float32)

       # Normalize kernel
       kernel = kernel / kernel.sum() if kernel.sum() != 0 else kernel

       # Cache the kernel
       self.kernel_cache[intensity] = kernel
       return kernel

   def _update_preview_image(self, intensity: int) -> None:
       """
       Update the preview image.

       Args:
           intensity: The sharpening intensity
       """
       try:
           sharpened_image = self._apply_sharpening(
               self.preview_image, intensity)
           self._display_image(sharpened_image)
       except Exception as e:
           logging.error(f"Error updating preview: {str(e)}")

   def _apply_sharpening(
           self,
           image: np.ndarray,
           intensity: int) -> np.ndarray:
       """
       Apply sharpening to the image.

       Args:
           image: The image to modify
           intensity: The sharpening intensity

       Returns:
           np.ndarray: The sharpened image
       """
       kernel = self._create_sharpen_kernel(intensity)
       return cv2.filter2D(image, -1, kernel)

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

   def get_sharpen_intensity(self) -> int:
       """
       Return the set sharpening intensity.

       Returns:
           int: The set sharpening intensity
       """
       return self.sharpen_slider.value()

   def closeEvent(self, event) -> None:
       """
       Overridden event handler for closing the dialog.

       Args:
           event: The close event
       """
       # Clear cache
       self.kernel_cache.clear()
       super().closeEvent(event)