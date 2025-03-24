"""
This module implements a dialog window for style transfer configuration.

The `StyleTransferDialog` class allows users to apply the style of a selected
style image to a content image, with real-time preview and progress indicator.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog,
                             QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                             QPushButton, QVBoxLayout, QWidget)


class StyleTransferDialog(QDialog):
    """
    Style transfer configuration dialog.

    This class allows selecting and previewing a style image,
    and executing style transfer with real-time feedback.

    Attributes:
        MAX_PREVIEW_WIDTH (int): Maximum width of preview images
        SUPPORTED_FORMATS (tuple): Supported image formats
    """

    MAX_PREVIEW_WIDTH = 300
    SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(self, content_image: np.ndarray,
                 parent: Optional[QWidget] = None) -> None:
        """
        Initialize the dialog window.

        Args:
            content_image: The original (content) image as numpy array (BGR format)
            parent: The parent widget (optional)
        """
        super().__init__(parent)
        self.setWindowTitle("Style Transfer")
        self.setModal(True)
        self.setMinimumWidth(800)

        if content_image is None:
            raise ValueError("Content image cannot be None")

        self.content_image = content_image
        self.style_image = None

        self._setup_ui()
        self._setup_connections()

        # Display content image preview
        self.preview_content_image = self._resize_for_preview(content_image)
        self._display_content_image()

        logging.debug("StyleTransferDialog initialized")

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Display images
        images_layout = QHBoxLayout()

        # Content image section
        content_layout = QVBoxLayout()
        content_layout.addWidget(
            QLabel(
                "Content image:",
                alignment=Qt.AlignCenter))
        self.content_preview = QLabel()
        self.content_preview.setFixedSize(
            self.MAX_PREVIEW_WIDTH, self.MAX_PREVIEW_WIDTH)
        self.content_preview.setAlignment(Qt.AlignCenter)
        self.content_preview.setStyleSheet("border: 2px solid gray;")
        content_layout.addWidget(self.content_preview)
        images_layout.addLayout(content_layout)

        # Style image section
        style_layout = QVBoxLayout()
        style_layout.addWidget(QLabel("Style image:", alignment=Qt.AlignCenter))
        self.style_preview = QLabel()
        self.style_preview.setFixedSize(
            self.MAX_PREVIEW_WIDTH, self.MAX_PREVIEW_WIDTH)
        self.style_preview.setAlignment(Qt.AlignCenter)
        self.style_preview.setStyleSheet("border: 2px solid gray;")
        self.style_preview.setText("No style image selected")
        style_layout.addWidget(self.style_preview)
        images_layout.addLayout(style_layout)

        layout.addLayout(images_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.select_style_button = QPushButton("Select Style Image")
        layout.addWidget(self.select_style_button)

        # Information label
        info_label = QLabel(
            "Select a style image whose style you want to transfer to the content image. "
            "The style transfer may take a few seconds.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # OK and Cancel buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        layout.addWidget(self.button_box)

    def _setup_connections(self) -> None:
        """Set up signal-slot connections."""
        self.select_style_button.clicked.connect(self._on_select_style)

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

    def _display_content_image(self) -> None:
        """Display the content image."""
        try:
            self._display_image(
                self.preview_content_image,
                self.content_preview)
            logging.debug("Content image successfully displayed")
        except Exception as e:
            logging.error(f"Error displaying content image: {str(e)}")
            self._show_error("Display Error",
                             "Failed to display content image")

    def _on_select_style(self) -> None:
        """Handle style image selection."""
        try:
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter(
                f"Image files (*{' *'.join(self.SUPPORTED_FORMATS)})"
            )
            
            if file_dialog.exec_() == QFileDialog.Accepted:
                selected_files = file_dialog.selectedFiles()
                if not selected_files:
                    return
                    
                file_name = selected_files[0]
                
                try:
                    # Try with cv2.imdecode
                    style_image = cv2.imdecode(
                        np.fromfile(file_name, dtype=np.uint8),
                        cv2.IMREAD_UNCHANGED
                    )
                    
                    if style_image is None:
                        # Fallback to PIL
                        from PIL import Image
                        with Image.open(file_name) as pil_image:
                            style_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                    if style_image is None:
                        raise ValueError("Failed to load style image")

                    self.style_image = style_image
                    self.preview_style_image = self._resize_for_preview(self.style_image)
                    self._display_image(self.preview_style_image, self.style_preview)

                    # Enable OK button
                    self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)
                    logging.info("Style image successfully loaded")

                except Exception as load_error:
                    logging.error(f"Detailed loading error: {str(load_error)}")
                    self._show_error(
                        "Loading Error",
                        "Failed to load style image. Check the file format and accessibility."
                    )

        except Exception as e:
            logging.error(f"Error during style image loading dialog: {str(e)}")
            self._show_error(
                "Loading Error",
                "An error occurred while displaying the image loading window."
            )

    def _display_image(self, image: np.ndarray, label: QLabel) -> None:
        """
        Display the image on the specified label.

        Args:
            image: The image to display
            label: The label on which to display
        """
        try:
            if image is None:
                label.setText("No image loaded")
                return
                
            # RGB conversion
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width

            q_image = QImage(
                rgb_image.data,
                width,
                height,
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

    def _show_error(self, title: str, message: str) -> None:
        """
        Display an error message.

        Args:
            title: The message title
            message: The error message text
        """
        QMessageBox.critical(self, title, message)

    def get_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return the selected images.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            A (content_image, style_image) tuple
        """
        return self.content_image, self.style_image

    def update_progress(self, value: int) -> None:
        """
        Update the progress bar.

        Args:
            value: The new value (0-100)
        """
        self.progress_bar.setValue(value)

    def show_progress(self, show: bool = True) -> None:
        """
        Show or hide the progress bar.

        Args:
            show: True to show, False to hide
        """
        self.progress_bar.setVisible(show)