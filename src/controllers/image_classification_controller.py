"""
This module contains the controller class for the image classification application
and related background tasks (ThumbnailWorker, ClassificationWorker).
Its responsibility is to manage models, views, and user interactions.
"""

import logging
import os
import shutil
from typing import List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from src.models.image_classification_model import (ImageClassificationModel,
                                                   UnsupportedModelError)
from src.views.image_classification_view import ImageClassificationView


class ThumbnailWorker(QRunnable):
    """
    A worker thread derived from QRunnable responsible for
    generating an image thumbnail.
    """

    class Signals(QObject):
        """Signals for ThumbnailWorker indicating changes in work status."""
        finished = pyqtSignal(str)  # Signal emitted when work is completed.
        error = pyqtSignal(str)     # Signal emitted when an error occurs.

    def __init__(self, image_path):
        """
        Initialize the ThumbnailWorker instance.

        Args:
            image_path (str): Path to the image to be processed.
        """
        super().__init__()
        self.signals = self.Signals()
        self.image_path = image_path

    def run(self):
        """
        Method responsible for executing the work.

        Processes the image and sends signals about the result.
        """
        try:
            self.signals.finished.emit(self.image_path)
        except Exception as e:
            self.signals.error.emit(str(e))


class ClassificationWorker(QRunnable):
    """A QRunnable thread that runs an image classification model."""

    class Signals(QObject):
        """Signals for ClassificationWorker indicating changes in work status."""
        finished = pyqtSignal(tuple)  # Signal emitted when classification is complete.
        # Signal indicating classification progress.
        progress = pyqtSignal(int)
        error = pyqtSignal(str)       # Signal emitted when an error occurs.

    def __init__(self, model, image_path, top_n):
        """
        Initialize the classification job.

        Args:
            model: Instance of the classification model.
            image_path (str): Path to the image to be classified.
            top_n (int): Number of top results to display in classification results.
        """
        super().__init__()
        self.signals = self.Signals()
        self.model = model
        self.image_path = image_path
        self.top_n = top_n

    def run(self):
        """Method executing the image classification task."""
        try:
            image = cv2.imread(self.image_path)
            if image is None:
                self.signals.error.emit(
                    f"Failed to load: {self.image_path}")
                return
            predictions = self.model.classify_image(image, top_n=self.top_n)
            self.signals.finished.emit((self.image_path, predictions))
        except Exception as e:
            self.signals.error.emit(str(e))


class ImageClassificationController:
    """
    Controller for the image classification application.

    The controller is responsible for handling user interactions, coordinating
    communication between the model and view, and executing image processing
    operations.
    """

    SUPPORTED_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')
    VIEWPORT_SCALE = 0.95
    DEFAULT_RESULT_TEXT = "Classification result will appear here"

    def __init__(self, view: ImageClassificationView) -> None:
        """
        Initialize the controller and set up event handlers.

        Args:
            view: The view for the image classification application
        """
        self.view = view
        self.model = ImageClassificationModel()
        self.selected_image: Optional[str] = None
        self.selected_folder: Optional[str] = None
        self.current_image: Optional[np.ndarray] = None
        self.update_status_bar_callback = None
        self.batch_results: List[str] = []
        self.output_directory = None
        self.view.select_output_dir_button.clicked.connect(
            self.select_output_directory)

        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(4)  # Number of parallel threads
        self.processed_images = 0
        self.total_images = 0
        self.batch_results = []

        self._setup_logging()
        self._connect_signals()
        self._initialize_ui()
        self._update_available_models()

    def _setup_logging(self) -> None:
        """Set up the logging system."""
        logging.basicConfig(
            filename='image_classification.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _connect_signals(self) -> None:
        """Connect view events to appropriate callback functions."""
        self.view.model_selector.currentTextChanged.connect(
            self._on_model_changed)
        self.view.select_image_button.clicked.connect(self.select_image)
        self.view.select_folder_button.clicked.connect(self.select_folder)
        self.view.classify_button.clicked.connect(self.start_classification)
        self.view.auto_sort_checkbox.stateChanged.connect(
            self.toggle_move_copy_options)
        self.view.save_results_button.clicked.connect(
            self.save_results_to_file)
        self.view.image_radio.clicked.connect(self.enable_image_selection)
        self.view.folder_radio.clicked.connect(self.enable_folder_selection)
        # self.view.add_model_button.clicked.connect(self._on_add_model_clicked)

    def _initialize_ui(self) -> None:
        """Initialize the initial state of the user interface."""
        self.enable_image_selection()
        self._update_model_info(self.view.get_selected_model_name())

    def _on_model_changed(self, model_name: str) -> None:
        """
        Handle model change.

        Args:
            model_name: Name of the selected model
        """
        try:
            # Update model information in the view
            model_descriptions = self.view.MODEL_DESCRIPTIONS
            model_info = model_descriptions.get(model_name, {
                "description": f"Custom model: {model_name}",
                "type": "custom"
            })

            # Determine model type
            if model_name.lower() in self.model.SUPPORTED_MODELS:
                model_type = "built-in"
            else:
                model_type = "custom"

            # Update view
            self.view.model_type_label.setText(model_type)

            # Compose description and link
            description = model_info.get(
                "description", "No description available")
            if "link" in model_info:
                description += f"<br><a href='{
                    model_info['link']}'>More information</a>"

            # Update description
            self.view.model_description_label.setText(description)

            # Create new model instance
            self.model = ImageClassificationModel(model_name)

        except UnsupportedModelError as e:
            self.view.show_error_message("Model Error", str(e))

    def _update_model_info(self, model_name: str) -> None:
        """
        Update model description and type in the view.

        Args:
            model_name (str): Name of the selected model.
        """
        model_descriptions = self.view.MODEL_DESCRIPTIONS
        model_info = model_descriptions.get(model_name, {
            "description": "No description available",
            "type": "unknown"
        })

        if model_name.lower() in self.model.SUPPORTED_MODELS:
            model_type = "built-in"
        else:
            model_type = "custom"

        self.view.update_model_info(model_name)

    def _on_add_model_clicked(self) -> None:
        """Handle click on add custom model button."""
        paths = self.view.show_add_model_dialog()
        if paths:
            model_path, labels_path = paths
            try:
                self.model.add_custom_model(model_path, labels_path)
                self._update_available_models()
                self.view.show_info_message(
                    "Successful Addition",
                    "Custom model successfully added."
                )
            except ValueError as e:
                self.view.show_error_message(
                    "Error adding model",
                    str(e)
                )

    def _update_available_models(self) -> None:
        """Update the list of available models in the interface."""
        available_models = self.model.get_available_models()
        self.view.update_available_models(available_models)

    def enable_image_selection(self) -> None:
        """Enable image selection and disable folder selection."""
        self.view.select_image_button.setEnabled(True)
        self.view.select_folder_button.setEnabled(False)
        self.selected_folder = None
        self.view.update_folder_path_display("")

        self.current_image = None
        self.selected_image = None
        self.view.update_image_path_display("")
        self.view.image_label.clear()
        self.view.image_label.setText(self.view.UI_TEXTS['no_image'])

        self.view.auto_sort_checkbox.setEnabled(False)
        self.view.auto_sort_checkbox.setChecked(False)
        self.view.move_radio.setEnabled(False)
        self.view.copy_radio.setEnabled(False)
        self.view.image_radio.setChecked(True)
        self.view.display_container.setCurrentWidget(self.view.image_label)
        self.view.thumbnail_grid.clear()

        self.view.result_label.setText(
            self.view.UI_TEXTS['result_placeholder'])
        self.batch_results.clear()

    def enable_folder_selection(self) -> None:
        """Enable folder selection and disable image selection."""
        self.view.select_image_button.setEnabled(False)
        self.view.select_folder_button.setEnabled(True)
        self.selected_image = None
        self.current_image = None
        self.view.update_image_path_display("")

        # Clear image from display and switch to thumbnail view
        self.view.image_label.clear()
        self.view.image_label.setText(self.view.UI_TEXTS['no_image'])
        self.view.display_container.setCurrentWidget(self.view.thumbnail_grid)
        self.view.thumbnail_grid.clear()

        # Clear results
        self.view.result_label.setText(
            self.view.UI_TEXTS['result_placeholder'])
        self.batch_results.clear()

        self.view.auto_sort_checkbox.setEnabled(True)
        self.view.folder_radio.setChecked(True)

    def toggle_move_copy_options(self) -> None:
        """
        Enable or disable the automatic copying or moving of images
        based on classification results.
        """
        enabled = self.view.auto_sort_checkbox.isChecked()
        self.view.move_radio.setEnabled(enabled)
        self.view.copy_radio.setEnabled(enabled)
        self.view.select_output_dir_button.setEnabled(enabled)
        self.view.output_dir_display.setEnabled(enabled)

        if enabled and not (self.view.move_radio.isChecked()
                            or self.view.copy_radio.isChecked()):
            self.view.copy_radio.setChecked(True)

    def get_selected_model(self) -> None:
        """
        Initialize the selected model if it hasn't been done yet.

        Raises:
            UnsupportedModelError: If the selected model is not supported
        """
        try:
            current_model = self.view.model_selector.currentText().lower()
            if self.model is None or self.model.model_name != current_model:
                self.model = ImageClassificationModel(model_name=current_model)
                logging.info(f"New model initialized: {current_model}")
        except UnsupportedModelError as e:
            logging.error(f"Model loading error: {str(e)}")
            QMessageBox.critical(
                self.view,
                "Model Error",
                f"The selected model is not supported: {
                    str(e)}")
            raise
        except Exception as e:
            logging.error(
                f"Unknown error during model loading: {
                    str(e)}")
            QMessageBox.critical(
                self.view,
                "Model Error",
                f"An error occurred while loading the model: {
                    str(e)}")
            raise

    def select_image(self) -> None:
        """Select and load an image for processing."""
        file_name, _ = QFileDialog.getOpenFileName(
            self.view,
            "Select Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_name:
            return

        image = self._load_and_validate_image(file_name)
        if image is None:
            return

        self.selected_image = file_name
        self.selected_folder = None
        self.view.update_image_path_display(file_name)

        self.current_image = image
        self.model.set_current_image(image)

        # Calculate and set viewport scaling
        img_height, img_width = image.shape[:2]
        scroll_area_size = self.view.scroll_area.size()
        view_width = scroll_area_size.width() * self.VIEWPORT_SCALE
        view_height = scroll_area_size.height() * self.VIEWPORT_SCALE

        width_ratio = view_width / img_width
        height_ratio = view_height / img_height
        self.scale = min(width_ratio, height_ratio)

        # Display image
        self.display_image(image)
        logging.info(f"Image selected: {file_name}")

    def _load_and_validate_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and validate the image.

        Args:
            image_path: Path to the image to load

        Returns:
            Optional[np.ndarray]: The loaded image or None in case of error
        """
        try:
            # Normalize path
            normalized_path = self._normalize_path(image_path)
            
            # First try with cv2.imread
            image = cv2.imread(normalized_path)
            
            # If that failed, try with PIL
            if image is None:
                from PIL import Image
                pil_image = Image.open(normalized_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                pil_image.close()

            if image is None:
                raise ValueError(f"Failed to load image: {normalized_path}")

            return image

        except Exception as e:
            logging.error(f"Error loading image: {str(e)}")
            self._show_error("Loading Error", 
                            f"Failed to load the image. Check if the filename "
                            f"contains special characters, or try "
                            f"renaming the file without accents.")
            return None

    def select_folder(self, folder_path: Optional[str] = None) -> None:
        """Select a folder for batch processing."""
        folder_path = QFileDialog.getExistingDirectory(
            self.view, "Select Folder")
        if folder_path:
            # Check if the folder exists and is accessible
            if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
                self.view.show_error_message(
                    "Invalid folder", 
                    "The selected folder doesn't exist or is not accessible."
                )
                return
                
            self.selected_folder = folder_path
            self.selected_image = None
            self.view.update_folder_path_display(folder_path)
            self._display_folder_thumbnails(folder_path)
            logging.info(f"Folder selected: {folder_path}")

    def start_classification(self) -> None:
        """Start image classification in the selected mode."""
        top_n = self.view.top_n_spinbox.value()

        # If auto-sort is enabled, check the target folder
        if self.view.auto_sort_checkbox.isChecked() and not self.output_directory:
            QMessageBox.warning(
                self.view,
                "Missing target folder",
                "Please select a target folder for automatic sorting!"
            )
            return

        if self.selected_image:
            self.classify_image(self.selected_image, top_n)
        elif self.selected_folder:
            self.batch_classify_images(self.selected_folder, top_n)
        else:
            QMessageBox.warning(
                self.view,
                "Warning",
                "Please select an image or folder."
            )

    def classify_image(self, image_path: str, top_n: int) -> None:
        """
        Classify an image and display the result.

        Args:
            image_path: Path to the image to classify
            top_n: Number of top results to display
        """
        try:
            image = self._load_and_validate_image(image_path)
            if image is None:
                return

            self.get_selected_model()
            
            # Add debug log
            logging.debug(f"Image size: {image.shape}")
            logging.debug(f"Model used: {self.model.model_name}")
            
            top_n_predictions = self.model.classify_image(image, top_n=top_n)

            # Format results
            results_text = "Classification results:\n\n"
            for i, (class_name, confidence) in enumerate(top_n_predictions, 1):
                results_text += f"{i}. {class_name}\n"
                results_text += f"   Confidence: {confidence * 100:.1f}%\n\n"

            self.view.result_label.setText(results_text)
            logging.info(f"Image classified: {image_path}")

        except Exception as e:
            logging.error(f"Error during classification: {str(e)}")
            detailed_error = f"Detailed error: {str(e)}\n"
            detailed_error += f"Error type: {type(e)}\n"
            logging.error(detailed_error)
            QMessageBox.warning(
                self.view,
                "Classification Error",
                f"An error occurred during classification: {str(e)}"
            )

    def batch_classify_images(self, folder_path: str, top_n: int) -> None:
        """
        Perform batch image classification in a folder.

        Args:
            folder_path: Folder containing images to process
            top_n: Number of top results to display
        """
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS)]

        if not image_files:
            QMessageBox.warning(
                self.view, "Error", "No images in the folder")
            return

        self.processed_images = 0
        self.total_images = len(image_files)
        self.batch_results = []
        self.view.progress_bar.setValue(0)
        self.get_selected_model()

        for image_file in image_files:
            try:
                image_path = os.path.join(folder_path, image_file)
                print(f"Processing: {image_path}")  # Debug

                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(
                        f"Failed to load image: {image_path}")

                # Store current image path
                self.current_image_path = image_path

                # Classification
                predictions = self.model.classify_image(image, top_n)

                # Format and store result
                result_text = f"{image_file}:\n"
                for i, (class_name, confidence) in enumerate(predictions, 1):
                    result_text += f"   {i}. {class_name} ({
                        confidence * 100:.1f}%)\n"
                result_text += "\n"

                self.batch_results.append(result_text)

                if self.view.auto_sort_checkbox.isChecked():
                    self._handle_auto_sort(
                        image_path, predictions[0][0], folder_path)

            except Exception as e:
                print(f"Error classifying {image_file}: {str(e)}")
                continue

            finally:
                self.processed_images += 1
                progress = int(
                    (self.processed_images / self.total_images) * 100)
                self.view.progress_bar.setValue(progress)

        # Display results
        if self.batch_results:
            self.view.result_label.setText("".join(self.batch_results))

    def _process_single_image_in_batch(
            self,
            image_file: str,
            folder_path: str,
            top_n: int,
            idx: int,
            total_images: int) -> None:
        """
        Process a single image in batch processing.

        Args:
            image_file: Name of the image file to process
            folder_path: Path to the folder
            top_n: Number of top results to display
            idx: Index of the image in the processing order
            total_images: Total number of images to process
        """
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            logging.warning(f"Failed to load image: {image_path}")
            return

        try:
            predictions = self.model.classify_image(image, top_n=top_n)

            # Format result
            result_text = f"{image_file}:\n"
            for i, (class_name, confidence) in enumerate(predictions, 1):
                result_text += f"   {i}. {class_name} ({
                    confidence * 100:.1f}%)\n"
            result_text += "\n"  # Extra line break between results

            self.batch_results.append(result_text)

            if self.view.auto_sort_checkbox.isChecked():
                self._handle_auto_sort(
                    image_path, predictions[0][0], folder_path)

            self.view.progress_bar.setValue(
                int((idx + 1) / total_images * 100))

        except Exception as e:
            logging.error(
                f"An error occurred during classification of {image_file}: {
                    str(e)}")

    def select_output_directory(self) -> None:
        """
        Open a dialog for the user to select the target folder 
        for automatic sorting.
        """
        directory = QFileDialog.getExistingDirectory(
            self.view, "Select target folder")
        if directory:
            # Check if the folder exists and we have write permission
            if not os.path.exists(directory) or not os.path.isdir(directory):
                self.view.show_error_message(
                    "Invalid target folder",
                    "The selected target folder doesn't exist or is not accessible."
                )
                return
                
            try:
                # Test write permission
                test_file = os.path.join(directory, ".test_write_permission")
                with open(test_file, 'w') as f:
                    f.write("")
                os.remove(test_file)
            except (IOError, OSError):
                self.view.show_error_message(
                    "Permission error",
                    "No write permission for the selected folder."
                )
                return
                
            self.output_directory = directory
            self.view.output_dir_display.setText(directory)

    def _handle_auto_sort(
            self,
            image_path: str,
            class_name: str,
            base_directory: str) -> None:
        if self.output_directory is None:
            self.view.show_error_message(
                "Missing target folder",
                "Please select a target folder for automatic sorting!")
            return

        if self.view.move_radio.isChecked():
            self.move_image_to_class_folder(
                image_path, class_name, self.output_directory)
        elif self.view.copy_radio.isChecked():
            self.copy_image_to_class_folder(
                image_path, class_name, self.output_directory)

    def _create_class_directory(
            self,
            base_directory: str,
            class_name: str) -> str:
        """
        Create a directory for the classification result.

        Args:
            base_directory: Path to the base directory
            class_name: Name of the class

        Returns:
            str: Path to the created directory
        """
        class_directory = os.path.join(base_directory, class_name)
        if not os.path.exists(class_directory):
            os.makedirs(class_directory)
        return class_directory

    def move_image_to_class_folder(
            self,
            image_path: str,
            class_name: str,
            base_directory: str) -> None:
        """
        Move an image to the folder corresponding to its class.

        Args:
            image_path: Path to the image to move
            class_name: Name of the class
            base_directory: Path to the base directory
        """
        try:
            class_directory = self._create_class_directory(
                base_directory, class_name)
            shutil.move(image_path, class_directory)
            logging.info(f"Image moved: {image_path} -> {class_directory}")
        except Exception as e:
            logging.error(f"An error occurred while moving the image: {str(e)}")
            raise

    def copy_image_to_class_folder(
            self,
            image_path: str,
            class_name: str,
            base_directory: str) -> None:
        """
        Copy an image to the folder corresponding to its class.

        Args:
            image_path: Path to the image to copy
            class_name: Name of the class
            base_directory: Path to the base directory
        """
        try:
            class_directory = self._create_class_directory(
                base_directory, class_name)
            shutil.copy(image_path, class_directory)
            logging.info(f"Image copied: {image_path} -> {class_directory}")
        except Exception as e:
            logging.error(f"An error occurred while copying the image: {str(e)}")
            raise

    def save_results_to_file(self) -> None:
        """Save classification results to a file."""
        results_text = self._get_results_text()

        if not self._validate_results_text(results_text):
            QMessageBox.warning(
                self.view,
                "No data to save",
                "No results to save.")
            return

        file_path = self._get_save_file_path()
        if not file_path:
            return

        try:
            self._write_results_to_file(file_path, results_text)
            QMessageBox.information(self.view, "Save successful",
                                    "Results were successfully saved.")
            logging.info(f"Results saved: {file_path}")
        except Exception as e:
            error_msg = f"Failed to save results: {e}"
            logging.error(error_msg)
            QMessageBox.critical(self.view, "An error occurred", error_msg)

    def _get_results_text(self) -> str:
        """
        Compile the text of the results to save.

        Returns:
            str: The results text
        """
        return "\n".join(
            self.batch_results) if self.batch_results else self.view.result_label.text()

    def _validate_results_text(self, results_text: str) -> bool:
        """
        Check if there are results to save.

        Args:
            results_text: The text to check

        Returns:
            bool: True if there are results to save, False if not
        """
        return bool(results_text and results_text != self.DEFAULT_RESULT_TEXT)

    def _get_save_file_path(self) -> Optional[str]:
        """
        Get the save location from the user.

        Returns:
            Optional[str]: Path to the selected file or None if canceled
        """
        file_path, _ = QFileDialog.getSaveFileName(
            self.view,
            "Save Results",
            "",
            "Text Files (*.txt)"
        )
        return file_path

    def _write_results_to_file(
            self,
            file_path: str,
            results_text: str) -> None:
        """
        Write the results to a file.

        Args:
            file_path: Path to the target file
            results_text: Text to save
        """
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write("Classification Results:\n")
            file.write(results_text)

    def display_image(self, image: np.ndarray) -> None:
        """Display the image with proper scaling."""
        if image is None:
            self.view.image_label.setText(self.view.UI_TEXTS['no_image'])
            return

        try:
            # Apply scaling
            height, width = image.shape[:2]
            new_width = int(width * self.scale)
            new_height = int(height * self.scale)

            # Check that new dimensions are not zero
            if new_width > 0 and new_height > 0:
                resized = cv2.resize(image, (new_width, new_height),
                                     interpolation=cv2.INTER_AREA)

                # Convert to QImage
                if len(resized.shape) == 3:  # Color image
                    height, width, channel = resized.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(
                        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
                        width, height, bytes_per_line, QImage.Format_RGB888
                    )
                else:  # Grayscale image
                    height, width = resized.shape
                    bytes_per_line = width
                    q_image = QImage(
                        resized.data, width, height, bytes_per_line,
                        QImage.Format_Grayscale8
                    )

                pixmap = QPixmap.fromImage(q_image)
                self.view.image_label.setPixmap(pixmap)
                self.view.image_label.setAlignment(Qt.AlignCenter)

                if self.update_status_bar_callback:
                    self.update_status_bar_callback()

            logging.debug(f"Image displayed with {self.scale:.2f}x zoom")

        except Exception as e:
            logging.error(f"Error displaying image: {str(e)}")
            self.view.image_label.setText("Error displaying image")

    def update_image_info(self) -> None:
        """Update the status bar with image size and zoom level."""
        self.display_image_info()

    def fit_to_view(self) -> None:
        """Fit the image to the view."""
        if self.current_image is not None:
            img_height, img_width = self.current_image.shape[:2]
            scroll_area_size = self.view.scroll_area.size()
            view_width = scroll_area_size.width() * self.VIEWPORT_SCALE
            view_height = scroll_area_size.height() * self.VIEWPORT_SCALE

            width_ratio = view_width / img_width
            height_ratio = view_height / img_height
            self.scale = min(width_ratio, height_ratio)

            self.display_image(self.current_image)
            logging.debug(f"Image fitted to view, new scale: {self.scale}")

    def _handle_classification_result(self, result):
        """
        Handle individual classification results and update the view.

        Args:
            result (tuple): The result contains the image path and classification results.
        """
        image_path, predictions = result
        image_file = os.path.basename(image_path)

        result_text = f"{image_file}:\n"
        for i, (class_name, confidence) in enumerate(predictions, 1):
            result_text += f"   {i}. {class_name} ({confidence * 100:.1f}%)\n"
        result_text += "\n"

        self.batch_results.append(result_text)

        if self.view.auto_sort_checkbox.isChecked():
            self._handle_auto_sort(
                image_path,
                predictions[0][0],
                self.output_directory)

        self.processed_images += 1
        progress = int((self.processed_images / self.total_images) * 100)
        self.view.progress_bar.setValue(progress)

        if self.processed_images == self.total_images:
            self.view.result_label.setText("\n".join(self.batch_results))

    def _handle_classification_error(self, error_message):
        """
        Handle errors that occur during classification.

        Args:
            error_message (str): The error message text.
        """
        self.processed_images += 1
        progress = int((self.processed_images / self.total_images) * 100)
        self.view.progress_bar.setValue(progress)
        logging.error(f"Classification error: {error_message}")

    def _display_folder_thumbnails(self, folder_path: str) -> None:
        """
        Display thumbnails of all images in the selected folder.

        Args:
            folder_path: Path to the folder
        """
        try:
            normalized_path = self._normalize_path(folder_path)
            
            self.view.display_container.setCurrentWidget(self.view.thumbnail_grid)
            self.view.thumbnail_grid.clear()

            image_files = [
                os.path.join(normalized_path, f) for f in os.listdir(normalized_path)
                if f.lower().endswith(self.SUPPORTED_IMAGE_EXTENSIONS)
            ]

            if not image_files:
                self.view.thumbnail_grid.message_label.setText("The folder does not contain any images")
                return

            for image_path in image_files:
                try:
                    normalized_image_path = self._normalize_path(image_path)
                    worker = ThumbnailWorker(normalized_image_path)
                    worker.signals.finished.connect(self._handle_thumbnail_loaded)
                    worker.signals.error.connect(self._handle_thumbnail_error)
                    self.threadpool.start(worker)
                except Exception as e:
                    logging.error(f"Error loading thumbnail: {str(e)}")
                    continue

        except Exception as e:
            logging.error(f"Error during folder loading: {str(e)}")
            self._show_error(
                "Folder loading error",
                f"Failed to load folder: {str(e)}\n"
                f"Check that folder names do not contain special characters."
            )

    def _handle_thumbnail_loaded(self, image_path: str) -> None:
        """
        Handle completion of thumbnail loading.

        Args:
            image_path (str): Path to the loaded image.
        """
        self.view.thumbnail_grid.add_thumbnail(image_path)

    def _handle_thumbnail_error(self, error: str) -> None:
        """
        Handle errors that occur during thumbnail loading.

        Args:
            error (str): The error message text.
        """
        logging.error(f"Thumbnail loading error: {error}")

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
            
    def _show_error(self, title: str, message: str) -> None:
        """
        Display an error message to the user.
        
        Args:
            title: Title of the error dialog
            message: Error message to display
        """
        QMessageBox.critical(self.view, title, message)