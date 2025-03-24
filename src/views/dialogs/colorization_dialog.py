"""
This module implements a dialog window used for colorizing black and white images.

The `ColorizationDialog` class contains colorization functions, such as
creating previews, progress indication, and artificial intelligence-based colorization.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QFrame, QHBoxLayout,
                             QLabel, QProgressBar, QPushButton, QVBoxLayout,
                             QWidget)

from src.models.image_transformation_model import ImageTransformationModel


class ColorizationDialog(QDialog):
    """
    Colorization settings dialog.

    This class enables colorization of black and white images
    with real-time preview.

    Attributes:
        MAX_PREVIEW_WIDTH (int): Maximum width of preview images
    """

    MAX_PREVIEW_WIDTH = 300

    def __init__(
            self,
            image: np.ndarray,
            model: ImageTransformationModel,
            parent: Optional[QWidget] = None) -> None:
        """
        Initialize the dialog window.

        Args:
            image: The original image as a numpy array (in BGR format)
            model: The image transformation model
            parent: The parent widget (optional)
        """
        super().__init__(parent)
        self.setWindowTitle("Image Colorization")
        self.setModal(True)
        self.setMinimumWidth(800)

        if image is None:
            raise ValueError("Input image cannot be None")

        print(f"Dialog - Received image size: {image.shape}")
        self.original_image = image
        self.preview_image = self._resize_for_preview(image)
        print(f"Dialog - Preview image size: {self.preview_image.shape}")
        self.colorized_image = None
        self.model = model

        self._setup_ui()
        self._create_preview()

    def _setup_ui(self) -> None:
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)

        # Preview images container
        preview_container = QWidget()
        preview_layout = QHBoxLayout(preview_container)
        preview_layout.setSpacing(20)  # Spacing between the two previews

        # Original image section
        original_section = self._create_preview_section(
            "Original Image",
            "original_preview"
        )
        preview_layout.addWidget(original_section)

        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        preview_layout.addWidget(separator)

        # Colorized image section
        colorized_section = self._create_preview_section(
            "Colorized Preview",
            "colorized_preview"
        )
        preview_layout.addWidget(colorized_section)

        layout.addWidget(preview_container)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Information panel
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.StyledPanel)
        info_frame.setStyleSheet("QFrame { background-color: #f0f0f0; }")
        info_layout = QVBoxLayout(info_frame)

        info_title = QLabel("Information")
        info_title.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(info_title)

        info_text = QLabel(
            "Colorization is performed using artificial intelligence.\n"
            "Results may depend on the content and quality of the image.\n"
            "The process may take a few seconds."
        )
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)

        layout.addWidget(info_frame)

        # Control buttons
        controls_layout = QHBoxLayout()

        # Colorize button
        self.colorize_button = QPushButton("Colorize")
        self.colorize_button.setToolTip("Start image colorization")
        self.colorize_button.clicked.connect(self._on_colorize_clicked)
        controls_layout.addWidget(self.colorize_button)

        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setToolTip("Restore original image")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        self.reset_button.setEnabled(False)
        controls_layout.addWidget(self.reset_button)

        layout.addLayout(controls_layout)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        self.ok_button = button_box.button(QDialogButtonBox.Ok)
        self.ok_button.setEnabled(False)
        layout.addWidget(button_box)

    def _create_preview_section(self, title: str, name: str) -> QWidget:
        """
        Create a preview section.

        Args:
            title: The section title
            name: The preview label name

        Returns:
            QWidget: The preview section widget
        """
        section = QWidget()
        layout = QVBoxLayout(section)

        # Title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label)

        # Preview image frame
        preview_frame = QFrame()
        preview_frame.setFrameShape(QFrame.StyledPanel)
        preview_frame.setStyleSheet("QFrame { background-color: white; }")
        preview_layout = QVBoxLayout(preview_frame)

        # Preview label
        preview_label = QLabel()
        preview_label.setFixedSize(
            self.MAX_PREVIEW_WIDTH,
            self.MAX_PREVIEW_WIDTH)
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setStyleSheet("border: 1px solid #cccccc;")
        # Dynamically store the reference
        setattr(self, name, preview_label)

        preview_layout.addWidget(preview_label)
        layout.addWidget(preview_frame)

        # Size information
        size_label = QLabel()
        size_label.setAlignment(Qt.AlignCenter)
        setattr(self, f"{name}_size", size_label)
        layout.addWidget(size_label)

        return section

    def _create_preview(self) -> None:
        """Create the preview images."""
        try:
            # Display original image
            self._display_image(self.preview_image, self.original_preview)
            self._update_size_info(
                self.preview_image,
                self.original_preview_size
            )

            # Colorized preview placeholder
            self.colorized_preview.setText("No colorized image yet")
            self.colorized_preview_size.clear()

            logging.debug("Preview images successfully created")

        except Exception as e:
            logging.error(f"Error creating preview images: {str(e)}")
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

    def _display_image(self, image: np.ndarray, label: QLabel) -> None:
        """
        Display the image on the specified label.

        Args:
            image: The image to display
            label: The label to display it on
        """
        try:
            if len(image.shape) == 2:  # Grayscale image
                h, w = image.shape
                bytes_per_line = w
                q_image = QImage(
                    image.data,
                    w, h,
                    bytes_per_line,
                    QImage.Format_Grayscale8
                )
            else:  # Color image
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = 3 * w
                q_image = QImage(
                    rgb_image.data,
                    w, h,
                    bytes_per_line,
                    QImage.Format_RGB888
                )

            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)

        except Exception as e:
            logging.error(f"Error displaying image: {str(e)}")
            label.setText("Error displaying")

    def _update_size_info(self, image: np.ndarray, label: QLabel) -> None:
        """
        Update the size information.

        Args:
            image: The image whose size to display
            label: The label to display the information on
        """
        height, width = image.shape[:2]
        label.setText(f"Size: {width}×{height} pixels")

    def _on_colorize_clicked(self) -> None:
        """Handle colorize button click."""
        try:
            self.colorize_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.colorized_preview.setText("Preparing colorization...")

            self.progress_bar.setValue(25)
            self.colorized_preview.setText("Uploading and colorizing image...")

            # Here we directly use the model
            print(
                f"Dialog - Image size sent for colorization: {self.original_image.shape}")
            self.colorized_image = self.model.colorize_image(
                self.original_image)
            print(
                f"Dialog - Size of colorized image received: {
                    self.colorized_image.shape if self.colorized_image is not None else 'None'}")

            if self.colorized_image is None:
                raise ValueError("Colorization failed")

            self.progress_bar.setValue(75)
            self.colorized_preview.setText("Processing result...")

            # Update preview
            preview = self._resize_for_preview(self.colorized_image)
            self._display_image(preview, self.colorized_preview)
            self._update_size_info(preview, self.colorized_preview_size)

            self.reset_button.setEnabled(True)
            self.ok_button.setEnabled(True)
            self.progress_bar.setValue(100)

        except Exception as e:
            logging.error(f"Error during colorization: {str(e)}")
            self.colorized_preview.setText(
                f"An error occurred during colorization: {str(e)}")
        finally:
            self.colorize_button.setEnabled(True)
            self.progress_bar.setVisible(False)

    def _on_reset_clicked(self) -> None:
        """Handle reset button click."""
        self.colorized_image = None
        self.colorized_preview.setText("No colorized image yet")
        self.colorized_preview_size.clear()
        self.reset_button.setEnabled(False)
        self.ok_button.setEnabled(False)

    def get_colorized_image(self) -> Optional[np.ndarray]:
        """
        Return the colorized image.

        Returns:
            Optional[np.ndarray]: The colorized image or None
        """
        return self.colorized_image

    def update_progress(self, value: int) -> None:
        """
        Update the progress indicator value.

        Args:
            value: The new value (0-100)
        """
        self.progress_bar.setValue(value)

    def show_progress(self, show: bool = True) -> None:
        """
        Show or hide the progress indicator.

        Args:
            show: True to show, False to hide
        """
        self.progress_bar.setVisible(show)