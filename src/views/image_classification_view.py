"""
This module implements the image classification application view using PyQt5.

It contains classes and functions for managing custom models, displaying thumbnails,
and handling the image classification application's user interface.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QImage, QImageReader, QPixmap
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                             QDialog, QDialogButtonBox, QFileDialog, QFrame,
                             QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QProgressBar, QPushButton,
                             QRadioButton, QScrollArea, QSpinBox, QSplitter,
                             QStackedWidget, QVBoxLayout, QWidget)


class CustomModelDialog(QDialog):
    """Dialog for adding custom models."""

    def __init__(self, parent=None):
        """
        Initializes the custom model dialog class.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("Add Custom Model")
        self.setModal(True)
        self._setup_ui()

    def _setup_ui(self):
        """
        Creates the user interface needed to add custom models.

        The interface includes options for selecting the model and label files,
        as well as OK and Cancel buttons.
        """
        layout = QVBoxLayout(self)

        # Model file selection
        model_layout = QHBoxLayout()
        self.model_path = QLineEdit()
        self.model_path.setReadOnly(True)
        self.browse_model = QPushButton("Browse...")
        model_layout.addWidget(QLabel("Model file (.h5):"))
        model_layout.addWidget(self.model_path)
        model_layout.addWidget(self.browse_model)
        layout.addLayout(model_layout)

        # Labels file selection
        labels_layout = QHBoxLayout()
        self.labels_path = QLineEdit()
        self.labels_path.setReadOnly(True)
        self.browse_labels = QPushButton("Browse...")
        labels_layout.addWidget(QLabel("Labels file (optional):"))
        labels_layout.addWidget(self.labels_path)
        labels_layout.addWidget(self.browse_labels)
        layout.addLayout(labels_layout)

        # Information label
        info_label = QLabel(
            "Note:\n"
            "- The model file must be in .h5 format\n"
            "- The labels file should contain one label per line\n"
            "- The order of labels must match the model's output classes")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Connect events
        self.browse_model.clicked.connect(self._browse_model)
        self.browse_labels.clicked.connect(self._browse_labels)

    def _browse_model(self):
        """Opens a file dialog to select the model file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "Model Files (*.h5)"
        )
        if file_name:
            self.model_path.setText(file_name)

    def _browse_labels(self):
        """Opens a file dialog to select the labels file."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Labels File", "", "Text Files (*.txt)"
        )
        if file_name:
            self.labels_path.setText(file_name)

    def get_paths(self) -> Tuple[str, Optional[str]]:
        """
        Returns the paths of the selected model and label files.

        Returns:
            Tuple[str, Optional[str]]: The model file path and optionally the labels file path.
        """
        return (
            self.model_path.text(),
            self.labels_path.text() if self.labels_path.text() else None
        )


class ThumbnailGrid(QWidget):
    """
    Widget for displaying thumbnails.

    Displays images and their filenames in a grid layout,
    while supporting dynamic addition and removal.
    """

    def __init__(self, parent=None):
        """
        Initializes the Thumbnail Grid class.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.layout.setSpacing(10)
        self.thumbnail_size = 150
        self.items = []

        self.message_label = QLabel("No folder selected")
        self.message_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.message_label, 0, 0, 1, 4)

    def clear(self):
        """Clears all currently displayed thumbnails."""
        for item in self.items:
            self.layout.removeWidget(item)
            item.deleteLater()
        self.items.clear()
        self.message_label.show()
        self.message_label.setText("No folder selected")

    def add_thumbnail(self, image_path):
        """
        Adds a new thumbnail to the grid.

        Args:
            image_path (str): Path to the image to be displayed as a thumbnail.

        Raises:
            Exception: In case of error, for example if the image cannot be read.
        """
        self.message_label.hide()
        try:
            # Create QImage on a separate thread
            image_reader = QImageReader(image_path)
            if image_reader.canRead():
                # Scale to thumbnail size
                image_reader.setScaledSize(
                    QSize(self.thumbnail_size, self.thumbnail_size))
                pixmap = QPixmap.fromImage(image_reader.read())

                # Create thumbnail label
                thumb_label = QLabel()
                thumb_label.setPixmap(pixmap)
                thumb_label.setAlignment(Qt.AlignCenter)
                thumb_label.setStyleSheet("border: 1px solid gray;")

                # Display filename
                filename = os.path.basename(image_path)
                text_label = QLabel(filename)
                text_label.setAlignment(Qt.AlignCenter)
                text_label.setWordWrap(True)

                # Container widget
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.addWidget(thumb_label)
                container_layout.addWidget(text_label)

                # Add to grid
                row = len(self.items) // 4
                col = len(self.items) % 4
                self.layout.addWidget(container, row, col)
                self.items.append(container)

                # Immediate display
                container.show()
                QApplication.processEvents()

        except Exception as e:
            logging.error(f"Error creating thumbnail: {str(e)}")


class ImageClassificationView(QWidget):
    """
    The user interface for the image classification application.

    This class is responsible for displaying and managing the user interface.
    It contains all UI elements and their layout.
    """

    # UI text constants
    UI_TEXTS = {
        'select_image': "Select Image",
        'select_folder': "Select Folder",
        'auto_sort': "Automatic folder organization",
        'move': "Move",
        'copy': "Copy",
        'classify': "Start Classification",
        'result_placeholder': "Classification results will appear here",
        'image_radio': "Classify individual image",
        'folder_radio': "Batch classify folder",
        'save_results': "Save Results",
        'model_select': "Select model:",
        'top_n': "Number of Top-N predictions:",
        'no_image': "No image loaded",
        'add_custom_model': "Add custom model...",
        'model_info': "Model Information",
        'model_type': "Type:"
    }

    # Model descriptions and links
    MODEL_DESCRIPTIONS = {
        "mobilenetv2": {
            "description": "MobileNetV2: Lightweight, fast model for mobile devices.",
            "link": "https://keras.io/api/applications/mobilenet",
            "type": "built-in"
        },
        "resnet50": {
            "description": "ResNet50: High-performance model for detailed images.",
            "link": "https://keras.io/api/applications/resnet",
            "type": "built-in"
        },
        "inceptionv3": {
            "description": "InceptionV3: Medium-sized, general-purpose image classifier.",
            "link": "https://keras.io/api/applications/inceptionv3",
            "type": "built-in"
        }
    }

    # UI constants
    SPLITTER_RATIO = 0.5
    DEFAULT_TOP_N = 3
    MAX_TOP_N = 10
    MIN_TOP_N = 1
    SPACING = 10

    def __init__(self) -> None:
        """Initializes the view and sets up the basic UI elements."""
        super().__init__()
        self._init_ui()
        self._setup_tooltips()
        self.scale = 1.0  # Initialize zoom level
        self.current_image = None  # Initialize current image
        self.auto_sort_checkbox.stateChanged.connect(
            self._toggle_move_copy_options)

    def _init_ui(self) -> None:
        """Initializes and arranges the UI elements."""
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Create left and right widgets
        left_widget = self._create_left_panel()
        right_widget = self._create_right_panel()

        # Set up splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([
            int(self.width() * self.SPLITTER_RATIO),
            int(self.width() * (1 - self.SPLITTER_RATIO))
        ])

    def _create_left_panel(self) -> QWidget:
        """
        Creates the left panel with image display.

        Returns:
            QWidget: The left panel widget
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        layout.addWidget(self.scroll_area)

        # Container widget for image display and thumbnail grid
        self.display_container = QStackedWidget()

        # Original image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # Thumbnail grid
        self.thumbnail_grid = ThumbnailGrid()

        # Add both to the stack widget
        self.display_container.addWidget(self.image_label)
        self.display_container.addWidget(self.thumbnail_grid)

        self.scroll_area.setWidget(self.display_container)

        # Set default text
        self.image_label.setText(self.UI_TEXTS['no_image'])

        return widget

    def _create_right_panel(self) -> QWidget:
        """
        Creates the right panel with control elements.

        Returns:
            QWidget: The right panel widget
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Radio buttons section
        layout.addWidget(self._create_radio_section())

        # File selection section
        layout.addLayout(self._create_file_selection_section())

        # Automatic sorting section
        layout.addLayout(self._create_auto_sort_section())

        # Separator
        layout.addWidget(self._create_separator())

        # Model selection section
        layout.addLayout(self._create_model_section())

        # Classification controls
        layout.addLayout(self._create_classification_controls())

        # Results display
        layout.addLayout(self._create_results_section())

        return widget

    def _create_radio_section(self) -> QWidget:
        """
        Creates the mode selection radio buttons.

        Returns:
            QWidget: The widget containing the radio buttons
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        mode_label = QLabel("Input type:")
        layout.addWidget(mode_label)

        self.image_radio = QRadioButton(self.UI_TEXTS['image_radio'])
        self.folder_radio = QRadioButton(self.UI_TEXTS['folder_radio'])

        self.image_folder_group = QButtonGroup()
        self.image_folder_group.addButton(self.image_radio)
        self.image_folder_group.addButton(self.folder_radio)

        layout.addWidget(self.image_radio)
        layout.addWidget(self.folder_radio)

        return widget

    def _create_file_selection_section(self) -> QHBoxLayout:
        """
        Creates the file selection section.

        Returns:
            QHBoxLayout: The layout of the file selection elements
        """
        layout = QVBoxLayout()

        # Image selection
        image_layout = QHBoxLayout()
        self.select_image_button = QPushButton(self.UI_TEXTS['select_image'])
        self.image_path_display = QLineEdit()
        self.image_path_display.setReadOnly(True)
        image_layout.addWidget(self.select_image_button)
        image_layout.addWidget(self.image_path_display)
        layout.addLayout(image_layout)

        # Folder selection
        folder_layout = QHBoxLayout()
        self.select_folder_button = QPushButton(self.UI_TEXTS['select_folder'])
        self.folder_path_display = QLineEdit()
        self.folder_path_display.setReadOnly(True)
        folder_layout.addWidget(self.select_folder_button)
        folder_layout.addWidget(self.folder_path_display)
        layout.addLayout(folder_layout)

        return layout

    def _create_auto_sort_section(self) -> QVBoxLayout:
        layout = QVBoxLayout()

        self.auto_sort_checkbox = QCheckBox(self.UI_TEXTS['auto_sort'])
        layout.addWidget(self.auto_sort_checkbox)

        move_copy_layout = QHBoxLayout()
        self.copy_radio = QRadioButton(self.UI_TEXTS['copy'])
        self.move_radio = QRadioButton(self.UI_TEXTS['move'])

        self.move_copy_group = QButtonGroup()
        self.move_copy_group.addButton(self.copy_radio)
        self.move_copy_group.addButton(self.move_radio)

        move_copy_layout.addWidget(self.copy_radio)
        move_copy_layout.addWidget(self.move_radio)
        layout.addLayout(move_copy_layout)

        # Add output directory selection
        output_dir_layout = QHBoxLayout()
        self.select_output_dir_button = QPushButton("Select Target Folder")
        self.output_dir_display = QLineEdit()
        self.output_dir_display.setReadOnly(True)
        output_dir_layout.addWidget(self.select_output_dir_button)
        output_dir_layout.addWidget(self.output_dir_display)
        layout.addLayout(output_dir_layout)

        # Initially disable all controls
        self.move_radio.setEnabled(False)
        self.copy_radio.setEnabled(False)
        self.select_output_dir_button.setEnabled(False)
        self.output_dir_display.setEnabled(False)

        return layout

    def _create_separator(self) -> QFrame:
        """
        Creates a separator line.

        Returns:
            QFrame: The separator line widget
        """
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator

    def _create_model_section(self) -> QVBoxLayout:
        """
        Creates the model selection section.

        Returns:
            QVBoxLayout: The layout of the model selection elements
        """
        layout = QVBoxLayout()

        # Model selector label and ComboBox
        model_header = QHBoxLayout()
        model_header.addWidget(QLabel(self.UI_TEXTS['model_select']))

        # self.add_model_button = QPushButton(self.UI_TEXTS['add_custom_model'])
        # model_header.addWidget(self.add_model_button)

        layout.addLayout(model_header)

        self.model_selector = QComboBox()
        layout.addWidget(self.model_selector)

        # Model information panel
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.StyledPanel)
        info_layout = QVBoxLayout(info_frame)

        # Model type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel(self.UI_TEXTS['model_type']))
        self.model_type_label = QLabel()
        type_layout.addWidget(self.model_type_label)
        info_layout.addLayout(type_layout)

        # Model description
        self.model_description_label = QLabel()
        self.model_description_label.setWordWrap(True)
        self.model_description_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.model_description_label.setTextFormat(Qt.RichText)
        self.model_description_label.setOpenExternalLinks(True)
        info_layout.addWidget(self.model_description_label)

        layout.addWidget(info_frame)

        return layout

    def _create_classification_controls(self) -> QVBoxLayout:
        """
        Creates the classification controls.

        Returns:
            QVBoxLayout: The layout of the classification controls
        """
        layout = QVBoxLayout()

        self.top_n_label = QLabel(self.UI_TEXTS['top_n'])
        layout.addWidget(self.top_n_label)

        self.top_n_spinbox = QSpinBox()
        self.top_n_spinbox.setRange(self.MIN_TOP_N, self.MAX_TOP_N)
        self.top_n_spinbox.setValue(self.DEFAULT_TOP_N)
        layout.addWidget(self.top_n_spinbox)

        self.classify_button = QPushButton(self.UI_TEXTS['classify'])
        layout.addWidget(self.classify_button)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        return layout

    def _create_results_section(self) -> QVBoxLayout:
        """
        Creates the results display section.

        Returns:
            QVBoxLayout: The complete layout of the results section
        """
        layout = QVBoxLayout()

        # Scrollable results area
        self.result_scroll_area = QScrollArea()
        self.result_scroll_area.setWidgetResizable(True)

        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout(self.result_widget)

        self.result_label = QLabel(self.UI_TEXTS['result_placeholder'])
        self.result_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.result_label.setWordWrap(True)
        self.result_layout.addWidget(self.result_label)

        self.result_scroll_area.setWidget(self.result_widget)
        layout.addWidget(self.result_scroll_area)

        # Save results button
        self.save_results_button = QPushButton(self.UI_TEXTS['save_results'])
        layout.addWidget(self.save_results_button)

        return layout

    def update_available_models(self, models: Dict[str, str]) -> None:
        """
        Updates the list of available models.

        Args:
            models: The models' names and types (built-in/custom)
        """
        self.model_selector.clear()
        for model_name, model_type in models.items():
            self.model_selector.addItem(model_name)
            if model_type == "custom":
                # For custom models, add the description
                self.MODEL_DESCRIPTIONS[model_name] = {
                    "description": f"Custom model: {model_name}",
                    "type": "custom"
                }

    def show_add_model_dialog(self) -> Optional[Tuple[str, Optional[str]]]:
        """
        Displays the add custom model dialog.

        Returns:
            Optional[Tuple[str, Optional[str]]]: The paths of the selected model and labels files,
            or None if canceled
        """
        dialog = CustomModelDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.get_paths()
        return None

    def update_model_info(self, model_name: str) -> None:
        """
        Updates the model information.

        Args:
            model_name: The name of the selected model
        """
        model_info = self.MODEL_DESCRIPTIONS.get(model_name.lower(), {
            "description": f"Custom model: {model_name}",
            "type": "custom"
        })

        self.model_type_label.setText(model_info.get("type", "unknown"))
        description = model_info.get("description", "No description")

        if "link" in model_info:
            description += f"<br><a href='{
                model_info['link']}'>More information</a>"

        self.model_description_label.setText(description)

    def _setup_tooltips(self) -> None:
        """Sets tooltips for UI elements."""
        self.select_image_button.setToolTip(
            "Select an image file (.jpg, .png, .bmp)")
        self.select_folder_button.setToolTip(
            "Select a folder for batch processing")
        self.auto_sort_checkbox.setToolTip(
            "Automatically organize images into subfolders by class")
        self.model_selector.setToolTip(
            "Select the classifier model to use")
        self.top_n_spinbox.setToolTip(
            "Specify how many top matches you want to see (1-10)")
        self.classify_button.setToolTip(
            "Start classification of selected image(s)")
        self.save_results_button.setToolTip(
            "Save classification results to a file")

    def update_image_path_display(self, path: str) -> None:
        """
        Updates the path display of the selected image.

        Args:
            path: The path of the selected image
        """
        self.image_path_display.setText(path)

    def update_folder_path_display(self, path: str) -> None:
        """
        Updates the path display of the selected folder.

        Args:
            path: The path of the selected folder
        """
        self.folder_path_display.setText(path)

    def update_progress(self, value: int) -> None:
        """
        Updates the progress bar value.

        Args:
            value: The new value (0-100)
        """
        self.progress_bar.setValue(value)

    def display_batch_results(
            self, results: List[Tuple[str, str, float]]) -> None:
        """
        Displays the results of batch processing.

        Args:
            results: List of (filename, class, confidence) tuples
        """
        self._clear_results()

        for img, class_name, confidence in results:
            result_text = f"{img}: {class_name} ({confidence:.2f}%)"
            label = QLabel(result_text)
            self.result_layout.addWidget(label)

    def _clear_results(self) -> None:
        """Clears previous results."""
        for i in reversed(range(self.result_layout.count())):
            widget = self.result_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def display_image(self, image: np.ndarray) -> None:
        """
        Displays the given image on the interface with appropriate scaling.

        Args:
            image: The image to display as a numpy array.

        Raises:
            ValueError: If the image format is incorrect or of invalid size.
        """
        if image is None:
            self.image_label.setText("No image loaded")
            return

        if not isinstance(image, np.ndarray):
            raise ValueError(
                "Incorrect image format: numpy.ndarray required")

        try:
            # Apply scaling
            height, width = image.shape[:2]
            if height == 0 or width == 0:
                raise ValueError(
                    "Invalid image size: 0 width or height")

            new_width = max(1, int(width * self.scale))
            new_height = max(1, int(height * self.scale))
            resized = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Convert to QImage
            if len(resized.shape) == 3:  # Color image
                if resized.shape[2] == 3:  # BGR format
                    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                elif resized.shape[2] == 4:  # BGRA format
                    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
                else:
                    raise ValueError(
                        f"Unsupported channel count: {
                            resized.shape[2]}")

                height, width, channel = rgb_image.shape
                bytes_per_line = channel * width
                q_image = QImage(
                    rgb_image.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_RGB888
                )

            elif len(resized.shape) == 2:  # Grayscale image
                height, width = resized.shape
                q_image = QImage(
                    resized.data,
                    width,
                    height,
                    width,
                    QImage.Format_Grayscale8
                )
            else:
                raise ValueError(
                    f"Unsupported image format: {len(resized.shape)} dimensions")

            # Create and display pixmap
            if q_image.isNull():
                raise ValueError("Failed to create QImage object")

            pixmap = QPixmap.fromImage(q_image)
            if pixmap.isNull():
                raise ValueError(
                    "Failed to create QPixmap object")

            self.image_label.setPixmap(pixmap)
            self.image_label.resize(pixmap.size())

            # Free QImage and pixmap
            del q_image
            del pixmap

        except Exception as e:
            logging.error(f"Error displaying image: {str(e)}")
            raise ValueError(f"Error displaying image: {str(e)}")

    def update_model_description(self) -> None:
        """
        Updates the description and link of the selected model.
        The description appears in HTML format with an embedded link.
        """
        try:
            selected_model = self.model_selector.currentText()
            model_info = self.MODEL_DESCRIPTIONS[selected_model]
            description = model_info["description"]
            link = model_info["link"]

            html_text = (
                f"{description}<br>"
                f"<a href='{link}'>More information: {link}</a>"
            )
            self.model_description_label.setText(html_text)
        except KeyError:
            self.model_description_label.setText(
                "No description found for the selected model."
            )
        except Exception as e:
            self.model_description_label.setText(
                f"Error loading model description: {str(e)}"
            )

    def show_error_message(self, title: str, message: str) -> None:
        """
        Displays an error message in a popup window.

        Args:
            title: The message title
            message: The detailed error message
        """
        QMessageBox.critical(self, title, message)

    def show_warning_message(self, title: str, message: str) -> None:
        """
        Displays a warning message in a popup window.

        Args:
            title: The message title
            message: The warning message
        """
        QMessageBox.warning(self, title, message)

    def show_info_message(self, title: str, message: str) -> None:
        """
        Displays an information message in a popup window.

        Args:
            title: The message title
            message: The information message
        """
        QMessageBox.information(self, title, message)

    def get_selected_model_name(self) -> str:
        """
        Returns the name of the selected model.

        Returns:
            str: The name of the selected model
        """
        return self.model_selector.currentText()

    def get_top_n_value(self) -> int:
        """
        Returns the set top-N value.

        Returns:
            int: The set top-N value
        """
        return self.top_n_spinbox.value()

    def is_auto_sort_enabled(self) -> bool:
        """
        Checks if automatic sorting is enabled.

        Returns:
            bool: True if automatic sorting is active, False if not
        """
        return self.auto_sort_checkbox.isChecked()

    def is_move_selected(self) -> bool:
        """
        Checks if move is selected for automatic sorting.

        Returns:
            bool: True if move is selected, False if copy is selected
        """
        return self.move_radio.isChecked()

    def _toggle_move_copy_options(self, state):
        """
        Enables or disables the move and copy options depending on automatic sorting.

        Args:
            state: The automatic sorting checkbox state (Qt.Checked or Qt.Unchecked).
        """
        is_enabled = state == Qt.Checked
        self.move_radio.setEnabled(is_enabled)
        self.copy_radio.setEnabled(is_enabled)
        self.select_output_dir_button.setEnabled(is_enabled)
        self.output_dir_display.setEnabled(is_enabled)