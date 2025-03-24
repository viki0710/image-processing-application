import logging
import os
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QMessageBox

from src.views.dialogs.colorization_dialog import ColorizationDialog
from src.views.dialogs.style_transfer_dialog import StyleTransferDialog
from src.views.dialogs.super_resolution_dialog import SuperResolutionDialog


class ImageTransformationController:
    """
    Controller for the image transformation module.

    This class is responsible for coordinating various AI-based image transformation operations,
    handling user interactions, and controlling model-view communication.
    """

    VIEWPORT_SCALE = 0.95
    MAX_DIMENSION = 4000  # Maximum allowed image size

    TRANSFORMATION_FUNCTIONS = [{"name": "Style Transfer",
                                 "function": "apply_style_transfer",
                                 "description": "Transfer the style of another image to the current image"},
                                {"name": "Colorize Black and White",
                                 "function": "colorize_image",
                                 "description": "Automatically colorize black and white images"},
                                {"name": "Super Resolution",
                                 "function": "super_resolution",
                                 "description": "Intelligently increase the resolution of an image"}]

    def __init__(self, model, view) -> None:
        """
        Initialize the controller.

        Args:
            model: The image transformation model
            view: The image transformation view
        """
        self.model = model
        self.view = view
        self.image_path: Optional[str] = None
        self.update_status_bar_callback = None

        self._setup_logging()
        self._setup_connections()
        self._populate_function_list()

        logging.info("ImageTransformationController initialized")

    def _setup_logging(self) -> None:
        """Set up the logging system."""
        logging.basicConfig(
            filename='image_transformation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _setup_connections(self) -> None:
        """Set up event handlers."""
        # Basic functions
        self.view.function_list.itemClicked.connect(self._on_function_selected)
        self.view.select_image_button.clicked.connect(self.select_image)
        self.view.save_button.clicked.connect(self.save_image)

        # History management
        self.view.undo_button.clicked.connect(self.undo)
        self.view.redo_button.clicked.connect(self.redo)
        self.view.reset_button.clicked.connect(self.reset_image)
        self.view.history_list.itemClicked.connect(
            self._on_history_item_clicked)

        """
        # Connect zoom controls
        parent = self.view.parent()
        if parent and hasattr(parent, 'zoom_in_button'):
            parent.zoom_in_button.clicked.connect(self.zoom_in)
            parent.zoom_out_button.clicked.connect(self.zoom_out)
            parent.reset_zoom_button.clicked.connect(self.zoom_reset)
            parent.fit_to_view_button.clicked.connect(self.fit_to_view)
        """

    def _populate_function_list(self) -> None:
        """Populate the list of transformation functions."""
        try:
            self.view.function_list.clear()
            for func in self.TRANSFORMATION_FUNCTIONS:
                self.view.function_list.addItem(func["name"])

            logging.debug("Function list successfully populated")

        except Exception as e:
            logging.error(f"Error populating function list: {str(e)}")
            self._show_error("List Error",
                             "Failed to load functions")

    def select_image(self) -> None:
        """Select an image for processing."""
        try:
            file_dialog = QFileDialog(self.view)
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("Image Files (*.png *.jpg *.jpeg *.bmp)")
            
            if file_dialog.exec_() == QFileDialog.Accepted:
                selected_files = file_dialog.selectedFiles()
                if not selected_files:
                    return
                    
                file_name = selected_files[0]
                
                # Load image safely
                try:
                    # Try with cv2.imdecode
                    image = cv2.imdecode(
                        np.fromfile(file_name, dtype=np.uint8),
                        cv2.IMREAD_UNCHANGED
                    )
                    
                    if image is None:
                        # Fallback to PIL
                        from PIL import Image
                        with Image.open(file_name) as pil_image:
                            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                    if image is None:
                        raise ValueError("Failed to load image")

                    # Calculate viewport scaling
                    scroll_area_size = self.view.scroll_area.size()
                    img_height, img_width = image.shape[:2]
                    view_width = scroll_area_size.width() * self.VIEWPORT_SCALE
                    view_height = scroll_area_size.height() * self.VIEWPORT_SCALE

                    width_ratio = view_width / img_width
                    height_ratio = view_height / img_height
                    self.model.scale = min(width_ratio, height_ratio)

                    # Load image into the model and update UI
                    self.model.set_current_image(image)
                    self.image_path = file_name
                    self.view.update_image_path_display(file_name)
                    self.display_image(self.model.current_image)
                    self.view.enable_editing_controls()
                    self.update_button_states()
                    self.update_image_info()

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

    def save_image(self) -> None:
        """Save the processed image."""
        if self.model.current_image is None:
            self._show_error("Save Error", "No image to save")
            return

        try:
            file_dialog = QFileDialog(self.view)
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            file_dialog.setNameFilter("PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)")
            
            if file_dialog.exec_() == QFileDialog.Accepted:
                file_name = file_dialog.selectedFiles()[0]
                if file_name:
                    # Check if there's an appropriate extension
                    name_filter = file_dialog.selectedNameFilter()
                    if "PNG" in name_filter and not file_name.lower().endswith('.png'):
                        file_name += '.png'
                    elif "JPEG" in name_filter and not file_name.lower().endswith(('.jpg', '.jpeg')):
                        file_name += '.jpg'
                    elif "BMP" in name_filter and not file_name.lower().endswith('.bmp'):
                        file_name += '.bmp'

                    # Check and create target folder if necessary
                    directory = os.path.dirname(file_name)
                    if not os.path.exists(directory):
                        try:
                            os.makedirs(directory)
                        except OSError as e:
                            self._show_error("Save Error", f"Failed to create target folder: {str(e)}")
                            return

                    try:
                        # First try to save with PIL
                        from PIL import Image
                        image = cv2.cvtColor(self.model.current_image, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(image)
                        pil_image.save(file_name)
                        pil_image.close()
                    except Exception as pil_error:
                        logging.warning(f"PIL save error: {str(pil_error)}, trying with cv2.imencode")
                        
                        try:
                            # If PIL doesn't work, try with cv2.imencode
                            ext = os.path.splitext(file_name)[1].lower()
                            if ext in ['.jpg', '.jpeg']:
                                _, buf = cv2.imencode('.jpg', self.model.current_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            elif ext == '.png':
                                _, buf = cv2.imencode('.png', self.model.current_image)
                            elif ext == '.bmp':
                                _, buf = cv2.imencode('.bmp', self.model.current_image)
                            else:
                                raise ValueError(f"Unsupported file format: {ext}")
                            
                            buf.tofile(file_name)
                        except Exception as cv_error:
                            raise Exception(f"Save failed with both methods: PIL: {str(pil_error)}, CV2: {str(cv_error)}")

                    logging.info(f"Image successfully saved: {file_name}")
                    self._show_info("Save Successful", "The image was successfully saved")

        except Exception as e:
            logging.error(f"Error saving image: {str(e)}")
            self._show_error(
                "Save Error",
                f"Failed to save image: {str(e)}\n"
                "Try saving to another folder or with a filename without accents."
            )

    def _on_function_selected(self, item) -> None:
        """
        Handle the selected transformation function.

        Args:
            item: The selected list item
        """
        if self.model.current_image is None:
            self._show_error("Warning", "Please load an image first!")
            return

        function_name = item.text()

        try:
            if function_name == "Style Transfer":
                self._handle_style_transfer()
            elif function_name == "Colorize Black and White":
                self._handle_colorization()
            elif function_name == "Super Resolution":
                self._handle_super_resolution()

        except Exception as e:
            logging.error(f"Error executing {function_name}: {str(e)}")
            self._show_error("Processing Error",
                             f"An error occurred during the operation: {str(e)}")

    def _handle_style_transfer(self) -> None:
        """Execute style transfer."""
        try:
            if self.model.current_image is None:
                self._show_error("Error", "No image loaded")
                return

            dialog = StyleTransferDialog(self.model.current_image, self.view)

            if dialog.exec_() == StyleTransferDialog.Accepted:
                dialog.show_progress(True)

                try:
                    content_image, style_image = dialog.get_images()

                    if content_image is not None and style_image is not None:
                        result = self.model.apply_style_transfer(
                            content_image, style_image)

                        if result is not None:
                            # Update image and history
                            self.model.add_to_history("Style Transfer", result)

                            # Update UI
                            self.display_image(self.model.current_image)
                            self.update_history_list()
                            self.update_button_states()
                            self.update_image_info()

                            self.fit_to_view()

                            logging.info("Style transfer successfully executed")
                        else:
                            raise ValueError("Style transfer result is None")

                except Exception as e:
                    logging.error(f"Error during style transfer: {str(e)}")
                    self._show_error("Processing Error", str(e))
                finally:
                    dialog.show_progress(False)

        except Exception as e:
            logging.error(f"Error handling style transfer: {str(e)}")
            self._show_error("Style Transfer Error", str(e))

    def _handle_super_resolution(self) -> None:
        """Handle super resolution."""
        try:
            if self.model.current_image is None:
                self._show_error("Error", "No image loaded")
                return
            
            if not self._check_network_connection():
                return

            dialog = SuperResolutionDialog(self.model.current_image, self.view)

            if dialog.exec_() == SuperResolutionDialog.Accepted:
                dialog.show_progress(True)  # This can stay if needed.

                try:
                    scale_factor = dialog.get_scale_factor()
                    result = self.model.super_resolution(
                        self.model.current_image, scale_factor)

                    if result is not None:
                        # Update image and history
                        self.model.add_to_history(
                            f"Super Resolution ({scale_factor}x)", result)

                        # Update UI
                        self.display_image(self.model.current_image)
                        self.update_history_list()
                        self.update_button_states()
                        self.update_image_info()

                        self.fit_to_view()

                        logging.info("Super resolution successfully executed")
                    else:
                        raise ValueError("Super resolution result is None")

                except Exception as e:
                    logging.error(f"Error during super resolution: {str(e)}")
                    self._show_error("Processing Error", str(e))
                finally:
                    dialog.show_progress(False)

        except Exception as e:
            logging.error(f"Error handling super resolution: {str(e)}")
            self._show_error("Super Resolution Error", str(e))

    def _handle_colorization(self) -> None:
        """Handle black and white image colorization."""
        try:
            if self.model.current_image is None:
                self._show_error("Error", "No image loaded")
                return

            if not self._check_network_connection():
                return

            print(
                f"Controller - Original image size: {self.model.current_image.shape}")
            dialog = ColorizationDialog(
                self.model.current_image, self.model, self.view)

            if dialog.exec_() == ColorizationDialog.Accepted:
                dialog.show_progress(True)  # This can stay if needed.

                try:
                    colorized_image = dialog.get_colorized_image()
                    print(
                        f"Controller - Colorized image size from dialog: {
                            colorized_image.shape if colorized_image is not None else 'None'}")

                    if colorized_image is not None:
                        # Update image and history
                        self.model.add_to_history("Colorization", colorized_image)

                        # Update UI
                        self.display_image(self.model.current_image)
                        self.update_history_list()
                        self.update_button_states()
                        self.update_image_info()

                        self.fit_to_view()

                        logging.info("Colorization successfully executed")
                    else:
                        raise ValueError("Colorization result is None")

                except Exception as e:
                    logging.error(f"Error during colorization: {str(e)}")
                    self._show_error(
                        "Processing Error",
                        f"Error during colorization: {
                            str(e)}")
                finally:
                    dialog.show_progress(False)

        except Exception as e:
            logging.error(f"Error handling colorization: {str(e)}")
            self._show_error(
                "Colorization Error",
                f"Failed to execute colorization: {
                    str(e)}")

    def _convert_cv_to_pixmap(self, image: np.ndarray) -> QPixmap:
        """
        Convert OpenCV image to QPixmap format.

        Args:
            image: The OpenCV image

        Returns:
            QPixmap: The converted image
        """
        try:
            height, width, channel = image.shape
            bytes_per_line = 3 * width

            # BGR -> RGB conversion
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create QImage
            q_image = QImage(
                rgb_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888
            )

            pixmap = QPixmap.fromImage(q_image)

            # Free memory
            del rgb_image

            return pixmap

        except Exception as e:
            logging.error(f"Error during image conversion: {str(e)}")
            return QPixmap()

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

    def update_image_info(self) -> None:
        """Update the display of image information."""
        if self.model.current_image is not None:
            height, width = self.model.current_image.shape[:2]
            # zoom_percentage = int(self.model.scale * 100)
            if self.update_status_bar_callback:
                self.update_status_bar_callback()

    def update_history_list(self) -> None:
        """Update the operation history list."""
        try:
            self.view.history_list.clear()
            if not self.model.history:
                return

            for i, (operation_name, _) in enumerate(self.model.history):
                item = self._create_history_item(operation_name, i)
                self.view.history_list.addItem(item)
        except Exception as e:
            logging.error(f"Error updating history list: {str(e)}")

    def _create_history_item(
            self,
            operation_name: str,
            index: int) -> QListWidgetItem:
        """
        Create a history list item.

        Args:
            operation_name: The operation name
            index: The operation index

        Returns:
            QListWidgetItem: The created list item
        """
        item = QListWidgetItem(operation_name)

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
        """Update button states."""
        has_image = self.model.current_image is not None
        self.view.undo_button.setEnabled(has_image and self.model.can_undo())
        self.view.redo_button.setEnabled(has_image and self.model.can_redo())
        self.view.reset_button.setEnabled(
            has_image and not np.array_equal(
                self.model.current_image,
                self.model.original_image
            )
        )

    def display_image(self, image: Optional[np.ndarray]) -> None:
        """Display the image on the interface."""
        try:
            if image is None:
                self.view.update_image_display(None)
                return

            # Apply scaling
            height, width = image.shape[:2]
            new_width = max(1, int(width * self.model.scale))
            new_height = max(1, int(height * self.model.scale))

            if new_width > 0 and new_height > 0:
                resized = cv2.resize(image, (new_width, new_height),
                                interpolation=cv2.INTER_AREA)

                # Convert to QImage
                if len(resized.shape) == 3:  # Color image
                    height, width, channel = resized.shape
                    bytes_per_line = 3 * width
                    
                    # BGR -> RGB conversion
                    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    
                    q_image = QImage(
                        rgb_image.data,
                        width, height,
                        bytes_per_line,
                        QImage.Format_RGB888
                    )
                else:  # Grayscale image
                    height, width = resized.shape
                    bytes_per_line = width
                    q_image = QImage(
                        resized.data,
                        width, height,
                        bytes_per_line,
                        QImage.Format_Grayscale8
                    )

                pixmap = QPixmap.fromImage(q_image)
                self.view.update_image_display(pixmap)
                self.view.image_label.setAlignment(Qt.AlignCenter)

                if self.update_status_bar_callback:
                    self.update_status_bar_callback()

                logging.debug(
                    f"Image displayed with {self.model.scale:.2f}x zoom")

        except Exception as e:
            logging.error(f"Error displaying image: {str(e)}")
            self.view.image_label.setText("Error displaying image")

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

    def _calculate_scale_for_viewport(self) -> float:
        """
        Calculate the appropriate scaling factor for the viewport size.

        Returns:
            float: The calculated scaling factor
        """
        if self.model.current_image is None:
            return 1.0

        img_height, img_width = self.model.current_image.shape[:2]
        scroll_area_size = self.view.scroll_area.size()

        view_width = scroll_area_size.width() * self.VIEWPORT_SCALE
        view_height = scroll_area_size.height() * self.VIEWPORT_SCALE

        width_ratio = view_width / img_width
        height_ratio = view_height / img_height

        return min(width_ratio, height_ratio)

    def undo(self) -> None:
        """Undo the last operation."""
        try:
            self.model.undo()
            self.display_image(self.model.current_image)
            self.update_history_list()
            self.update_button_states()
            self.update_image_info()
            self.fit_to_view()
            logging.info("Operation undone")
        except Exception as e:
            logging.error(f"Error undoing operation: {str(e)}")
            self._show_error("Undo Error", str(e))

    def redo(self) -> None:
        """Redo the last undone operation."""
        try:
            self.model.redo()
            self.display_image(self.model.current_image)
            self.update_history_list()
            self.update_button_states()
            self.update_image_info()
            self.fit_to_view()
            logging.info("Operation redone")
        except Exception as e:
            logging.error(f"Error redoing operation: {str(e)}")
            self._show_error("Redo Error", str(e))

    def reset_image(self) -> None:
        """Restore the original image."""
        try:
            reply = QMessageBox.question(
                self.view,
                'Reset',
                "Are you sure you want to restore the original image?\n"
                "All modifications will be lost!",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.model.reset_image()
                self.display_image(self.model.current_image)
                self.update_history_list()
                self.update_button_states()
                self.update_image_info()
                self.fit_to_view()
                logging.info("Image restored to original state")
        except Exception as e:
            logging.error(f"Error resetting image: {str(e)}")
            self._show_error("Reset Error", str(e))

    def _on_history_item_clicked(self, item: QListWidgetItem) -> None:
        """
        Handle navigation between operation history.

        Args:
            item: The selected history list item
        """
        try:
            index = self.view.history_list.row(item)
            if index != self.model.current_index:
                self.model.set_state_to_history_index(index)
                self.display_image(self.model.current_image)
                self.update_button_states()
                self.update_history_list()
                self.update_image_info()
                self.fit_to_view()
        except Exception as e:
            logging.error(f"Error restoring history state: {str(e)}")
            self._show_error("History Error", str(e))

    def fit_to_view(self) -> None:
        """Fit the image to the view."""
        if self.model.current_image is not None:
            self.model.scale = self._calculate_scale_for_viewport()
            self.display_image(self.model.current_image)

    def _check_network_connection(self) -> bool:
        """
        Check network connection for API access.
        
        Returns:
            bool: True if connected, False if not
        """
        try:
            import socket
            # Try to connect to the DeepAI API server
            socket.create_connection(("api.deepai.org", 443), timeout=3)
            return True
        except OSError:
            self._show_error(
                "Network Error",
                "No internet connection or API server is not accessible.\n"
                "An active internet connection is required for this operation."
            )
            return False
        
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