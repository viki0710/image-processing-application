"""This file implements the image editor controller, which coordinates communication between the Model and View."""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QFileDialog, QListWidgetItem, QMessageBox

from src.models.image_editing_model import (ImageEditingModel, ImageLoadError,
                                            ImageProcessingError)
from src.views.dialogs import (AdjustmentsDialog, BlurDialog, FlipDialog,
                               GammaDialog, ResizeDialog, RotateDialog,
                               SaturationDialog, SharpenDialog)
from src.views.image_editing_view import ImageEditingView


class ImageEditingController:
    """
    Controller for the image editing application.

    This class is responsible for coordinating communication between the Model and View,
    handling user interactions, and executing image processing operations.

    Attributes:
        SUPPORTED_FORMATS (tuple): Supported image formats
        MAX_IMAGE_SIZE (tuple): Maximum allowed image size
        VIEW_SCALE_FACTOR (float): Viewport scaling factor
    """

    SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp")
    MAX_IMAGE_SIZE = (20000, 20000)  # Maximum 20000x20000 pixels
    VIEWPORT_SCALE = 0.95  # 95% of viewport size

    MAX_SCALE = 5.0
    MIN_SCALE = 0.2
    DEFAULT_SCALE = 1.0
    ZOOM_STEP = 1.2

    def __init__(self, view: ImageEditingView) -> None:
        """
        Initialize the controller.

        Args:
            view: Instance of the image editing view
        """
        self.view = view
        self.model = ImageEditingModel()
        self.update_status_bar_callback = None

        self._connect_signals()
        self._setup_logging()
        self.update_ui()

        logging.info("ImageEditingController initialized")

    def _setup_logging(self) -> None:
        """Set up the logging system."""
        logging.basicConfig(
            filename='image_editing.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _connect_signals(self) -> None:
        """Connect view events to the appropriate handler functions."""
        self.view.select_image_button.clicked.connect(self.open_image)
        self.view.save_button.clicked.connect(self.save_image)
        self.view.reset_button.clicked.connect(self.reset_image)
        self.view.undo_button.clicked.connect(self.undo)
        self.view.redo_button.clicked.connect(self.redo)
        self.view.function_list.itemClicked.connect(self.on_function_selected)
        self.view.history_list.itemClicked.connect(
            self.on_history_item_clicked)

    def open_image(self) -> None:
        """
        Open an image for editing.

        Loads and displays the image selected by the user,
        and sets the appropriate scale ratio for the viewport.
        """
        try:
            file_dialog = QFileDialog(self.view)
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter(f"Image Files (*{' *'.join(self.SUPPORTED_FORMATS)})")
            
            if file_dialog.exec_() == QFileDialog.Accepted:
                selected_files = file_dialog.selectedFiles()
                if not selected_files:
                    return
                    
                file_name = selected_files[0]
                
                # Convert the path to the appropriate format
                try:
                    # Try to load directly first
                    image = cv2.imdecode(
                        np.fromfile(file_name, dtype=np.uint8),
                        cv2.IMREAD_UNCHANGED
                    )
                    
                    if image is None:
                        # If that failed, try with PIL
                        from PIL import Image
                        with Image.open(file_name) as pil_image:
                            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    
                    if image is None:
                        raise ValueError("Failed to load image")
                    
                    self.model.load_image(file_name)
                    self.view.update_image_path_display(file_name)
                    
                    # Calculate viewport scaling
                    self._calculate_and_set_scale()
                    
                    # Update UI
                    self.update_ui()
                    self.display_image_info()
                    logging.info(f"Image successfully loaded: {file_name}")
                    
                except Exception as load_error:
                    logging.error(f"Detailed loading error: {str(load_error)}")
                    self._show_error(
                        "Loading Error",
                        "Failed to load image. Check the file format and accessibility."
                    )
                    
        except Exception as e:
            logging.error(f"Error during image loading dialog: {str(e)}")
            self._show_error(
                "Loading Error",
                "An error occurred while displaying the image loading window."
            )

    def _load_and_display_image(self, file_name: str) -> None:
        """
        Load and display the image, set the appropriate scale ratio.

        Args:
            file_name: Path to the image file to load
        """
        self.model.load_image(file_name)
        self.view.update_image_path_display(file_name)

        # Calculate viewport scaling
        self._calculate_and_set_scale()

        # Update UI
        self.update_ui()
        self.display_image_info()
        logging.info(f"Image successfully loaded: {file_name}")

    def _calculate_and_set_scale(self) -> None:
        """Calculate and set the appropriate image scaling ratio."""
        scroll_area_size = self.view.scroll_area.size()
        view_width = scroll_area_size.width() * self.VIEWPORT_SCALE
        view_height = scroll_area_size.height() * self.VIEWPORT_SCALE

        img_width = self.model.current_image.shape[1]
        img_height = self.model.current_image.shape[0]

        width_ratio = view_width / img_width
        height_ratio = view_height / img_height

        # Set scaling ratio
        self.model.scale = min(width_ratio, height_ratio) if (
            img_width > view_width or img_height > view_height
        ) else 1.0

    def save_image(self) -> None:
        """Save the edited image."""
        try:
            if self.model.get_current_image() is None:
                raise ValueError("No image to save")

            file_name, selected_filter = QFileDialog.getSaveFileName(
                self.view,
                "Save Image",
                "",
                "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
            )

            if file_name:
                # Normalize path
                normalized_path = self._normalize_path(file_name)
                
                # Check if we have write permission
                directory = os.path.dirname(normalized_path)
                if not os.path.exists(directory):
                    try:
                        os.makedirs(directory)
                    except OSError:
                        self._show_error("Save Error", "Failed to create the target folder")
                        return

                # Save using PIL
                try:
                    from PIL import Image
                    image = cv2.cvtColor(self.model.get_current_image(), cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image)
                    pil_image.save(normalized_path)
                    pil_image.close()
                except Exception as e:
                    # If PIL failed, try with cv2
                    if not cv2.imwrite(normalized_path, self.model.get_current_image()):
                        raise Exception("Failed to save image")

                logging.info(f"Image successfully saved: {normalized_path}")
                self._show_info("Save Successful", "The image was successfully saved.")

        except Exception as e:
            self._show_error(
                "Save Error",
                f"Failed to save the image: {str(e)}\n"
                f"Try saving with a filename without accents."
            )

    def display_image(self, image: Optional[np.ndarray]) -> None:
        """
        Display the image on the interface.

        Args:
            image: The OpenCV format image to display, or None.
        """
        try:
            if image is None:
                self.view.update_image_display(None)
                return

            q_image = self._convert_cv_to_qimage(image)
            if q_image is not None:
                pixmap = QPixmap.fromImage(q_image)
                self.view.update_image_display(pixmap)

                if self.update_status_bar_callback:
                    self.update_status_bar_callback()

        except Exception as e:
            self._show_error("Display Error",
                             f"Error displaying image: {str(e)}")

    def _convert_cv_to_qimage(self, image: np.ndarray) -> Optional[QImage]:
        """
        Convert an OpenCV image to QImage format.

        Args:
            image: The OpenCV image

        Returns:
            Optional[QImage]: The converted QImage or None in case of error
        """
        try:
            if len(image.shape) == 2:  # Grayscale image
                height, width = image.shape
                bytes_per_line = width
                return QImage(image.data, width, height,
                              bytes_per_line, QImage.Format_Grayscale8)
            else:  # Color image
                height, width, channel = image.shape
                bytes_per_line = 3 * width
                return QImage(
                    cv2.cvtColor(
                        image,
                        cv2.COLOR_BGR2RGB),
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_RGB888)
        except Exception as e:
            logging.error(f"Image conversion error: {str(e)}")
            return None

    def on_function_selected(self, item: QListWidgetItem) -> None:
        """
        Handle the application of the selected image editing function.

        Args:
            item: The selected function list item
        """
        function_name = item.text()
        try:
            if function_name in self._get_dialog_handlers():
                self._handle_dialog_function(function_name)
            else:
                self._handle_simple_function(function_name)

            self._update_after_operation(function_name)

        except ImageProcessingError as e:
            self._show_error("Processing Error", str(e))
        except Exception as e:
            self._show_error("Unexpected Error",
                             f"Error executing function: {str(e)}")

    def _get_dialog_handlers(self) -> Dict[str, Any]:
        """
        Return handlers for functions that use dialog windows.

        Returns:
            Dict[str, Any]: Dictionary of functions and their handlers
        """
        return {
            "Rotation": (RotateDialog, "get_rotation_angle"),
            "Resize": (ResizeDialog, "get_values"),
            "Flip": (FlipDialog, "get_flip_code"),
            "Blur": (BlurDialog, "get_blur_intensity"),
            "Sharpen": (SharpenDialog, "get_sharpen_intensity"),
            "Contrast and Brightness": (AdjustmentsDialog, "get_values"),
            "Saturation": (SaturationDialog, "get_saturation_scale"),
            "Gamma Correction": (GammaDialog, "get_gamma_value")
        }

    def _handle_dialog_function(self, function_name: str) -> None:
        """
        Handle functions that use dialog windows.

        Args:
            function_name: Name of the function to execute
        """
        dialog_class, getter_method = self._get_dialog_handlers()[
            function_name]

        # Special handling for Saturation
        if (function_name == "Saturation" and
                self.model.is_grayscale()):
            self._show_info("Black and White Image",
                            "This image is already black and white, "
                            "so saturation cannot be modified.")
            return

        dialog = dialog_class(self.model.current_image)
        if dialog.exec_() == QDialog.Accepted:
            params = getattr(dialog, getter_method)()
            if isinstance(params, tuple):
                self.model.apply_function(function_name, *params)
            else:
                self.model.apply_function(function_name, params)
            self.display_image(self.model.get_current_image())

    def _handle_simple_function(self, function_name: str) -> None:
        """
        Handle simple (dialog-less) functions.

        Args:
            function_name: Name of the function to execute
        """
        self.model.apply_function(function_name)
        self.display_image(self.model.get_current_image())

    def _update_after_operation(self, function_name: str) -> None:
        """
        Update the UI after an operation is executed.

        Args:
            function_name: Name of the executed function
        """
        self.update_history_list()
        self.update_button_states()
        self.update_image_info()
        logging.info(f"Function successfully executed: {function_name}")

    def on_history_item_clicked(self, item: QListWidgetItem) -> None:
        """
        Handle navigation between operation history items.

        Args:
            item: The selected history list item
        """
        index = self.view.history_list.row(item)
        try:
            self.model.set_state_to_history_index(index)
            self.update_ui()
        except Exception as e:
            self._show_error(
                "History Error",
                f"Error restoring history state: {
                    str(e)}")

    def update_ui(self) -> None:
        """Update the entire user interface."""
        self.display_image(self.model.get_current_image())
        self.update_history_list()
        self.update_button_states()

        if (self.model.current_image is not None and
                self.model.image_path is not None):
            self.display_image_info()

    def update_history_list(self) -> None:
        """Update the operation history list."""
        try:
            self.view.history_list.clear()
            for i, (function_name, args, _) in enumerate(
                    self.model.get_history()):
                item = self._create_history_item(function_name, args, i)
                self.view.history_list.addItem(item)
        except Exception as e:
            logging.error(f"Error updating history list: {str(e)}")

    def _create_history_item(
            self,
            function_name: str,
            args: Tuple,
            index: int) -> QListWidgetItem:
        """
        Create a list item for the history list.
        """
        # Special display for initial state
        if index == 0:
            display_text = "Original image"
        else:
            args_str = ', '.join(map(str, args))
            display_text = f"{function_name}({args_str})"

        item = QListWidgetItem(display_text)

        # Mark current state
        if index == self.model.current_index:
            item.setBackground(QColor(200, 200, 255))

        # Display undone operations
        if index > self.model.current_index:
            item.setForeground(QColor(128, 128, 128))
            font = item.font()
            font.setItalic(True)
            item.setFont(font)

        return item

    def update_button_states(self) -> None:
        """Update the state of control buttons."""
        self.view.undo_button.setEnabled(self.model.can_undo())
        self.view.redo_button.setEnabled(self.model.can_redo())
        self.view.reset_button.setEnabled(
            self.model.current_image is not None and not np.array_equal(
                self.model.current_image, self.model.original_image))
        self.view.function_list.setEnabled(
            self.model.current_image is not None)

    def undo(self) -> None:
        """Undo the last operation."""
        try:
            self.model.undo()
            self.update_ui()
            logging.info("Operation successfully undone")
        except Exception as e:
            self._show_error("Undo Error",
                             f"Error undoing operation: {str(e)}")

    def redo(self) -> None:
        """Redo the last undone operation."""
        try:
            self.model.redo()
            self.update_ui()
            logging.info("Operation successfully redone")
        except Exception as e:
            self._show_error(
                "Redo Error",
                f"Error redoing operation: {
                    str(e)}")

    def reset_image(self) -> None:
        """Reset the image to its original state."""
        try:
            reply = self._show_confirmation(
                "Reset",
                "Are you sure you want to reset to the original image? "
                "All modifications will be lost!"
            )

            if reply == QMessageBox.Yes:
                self.model.reset_image()
                self.update_ui()
                logging.info(
                    "Image successfully reset to original state")

        except Exception as e:
            self._show_error("Reset Error",
                             f"Error resetting image: {str(e)}")

    def display_image_info(self) -> None:
        """Update the status bar with image information."""
        if self.update_status_bar_callback:
            height, width = self.model.current_image.shape[:2]
            # zoom_percentage = int(self.model.scale * 100)
            self.update_status_bar_callback()

    def update_status_bar(self) -> None:
        """Update the status bar."""
        if (self.model.current_image is not None and
                self.update_status_bar_callback):
            height, width = self.model.current_image.shape[:2]
            # zoom_percentage = int(self.model.scale * 100)
            self.update_status_bar_callback()

    def _show_error(self, title: str, message: str) -> None:
        """
        Display an error message.

        Args:
            title: Message title
            message: Error message text
        """
        logging.error(f"{title}: {message}")
        QMessageBox.critical(self.view, title, message)

    def _show_info(self, title: str, message: str) -> None:
        """
        Display an information message.

        Args:
            title: Message title
            message: Information message text
        """
        logging.info(f"{title}: {message}")
        QMessageBox.information(self.view, title, message)

    def _show_confirmation(self, title: str, message: str) -> int:
        """
        Display a confirmation dialog.

        Args:
            title: Message title
            message: Confirmation message text

        Returns:
            int: User's choice (QMessageBox.Yes or QMessageBox.No)
        """
        return QMessageBox.question(
            self.view,
            title,
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

    def update_image_info(self) -> None:
        """Update the display of image information."""
        self.display_image_info()

    def can_zoom_in(self) -> bool:
        """Check if zoom in is possible."""
        return self.model.scale * self.ZOOM_STEP <= self.MAX_SCALE

    def can_zoom_out(self) -> bool:
        """Check if zoom out is possible."""
        return self.model.scale / self.ZOOM_STEP >= self.MIN_SCALE

    def zoom_in(self) -> None:
        """Zoom in on the current image."""
        if self.model.current_image is not None:
            new_scale = self.model.scale * self.ZOOM_STEP
            self.model.scale = min(new_scale, self.MAX_SCALE)
            self.display_image(self.model.get_current_image())

    def zoom_out(self) -> None:
        """Zoom out from the current image."""
        if self.model.current_image is not None:
            new_scale = self.model.scale / self.ZOOM_STEP
            self.model.scale = max(new_scale, self.MIN_SCALE)
            self.display_image(self.model.get_current_image())

    def zoom_reset(self) -> None:
        """Reset to the original size."""
        if self.model.current_image is not None:
            self.model.scale = self.DEFAULT_SCALE
            self.display_image(self.model.get_current_image())

    def fit_to_view(self) -> None:
        """Fit the image to the view."""
        if self.model.current_image is not None:
            height, width = self.model.current_image.shape[:2]
            scroll_area_size = self.view.scroll_area.size()

            # Also use 0.95 here
            view_width = scroll_area_size.width() * self.VIEWPORT_SCALE
            view_height = scroll_area_size.height() * self.VIEWPORT_SCALE

            width_ratio = view_width / width
            height_ratio = view_height / height

            self.model.scale = min(width_ratio, height_ratio)
            self.display_image(self.model.get_current_image())

    def _normalize_path(self, path: str) -> str:
        """
        Normalize the path to handle special characters.

        Args:
            path: The original path

        Returns:
            str: The normalized path
        """
        try:
            # Unicode normalization
            normalized_path = os.path.normpath(path)
            # Make sure it's a string and not bytes
            if isinstance(normalized_path, bytes):
                normalized_path = normalized_path.decode('utf-8')
            return normalized_path
        except Exception as e:
            logging.error(f"Error normalizing path: {str(e)}")
            return path