"""
This module contains classes and functions for displaying and managing the PyQt5-based main application window.
"""

import logging
from typing import Optional

from PyQt5.QtWidgets import (QFrame, QHBoxLayout, QLabel, QMainWindow,
                             QMessageBox, QPushButton, QTabWidget, QVBoxLayout,
                             QWidget)


class MainWindowView(QMainWindow):
    """
    The main window of the application.

    This class is responsible for displaying and managing the main application window.
    It contains the tab widget for different modules (image classification,
    image transformation, image editing), as well as general control elements.

    Attributes:
        DEFAULT_WINDOW_SIZE (tuple): Default window size
        MIN_WINDOW_SIZE (tuple): Minimum window size
        SPACING (int): General spacing size
        STATUS_BAR_TIMEOUT (int): Display time for status messages (ms)
    """

    # UI constants
    DEFAULT_WINDOW_SIZE = (1200, 800)
    MIN_WINDOW_SIZE = (800, 600)
    SPACING = 10
    STATUS_BAR_TIMEOUT = 3000

    def __init__(self) -> None:
        """Initializes the main window."""
        super().__init__()
        self._setup_logging()
        self.initUI()
        logging.info("MainWindowView initialized")

    def _setup_logging(self) -> None:
        """Sets up the logging system."""
        logging.basicConfig(
            filename='main_window.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def initUI(self) -> None:
        """Initializes and sets up the UI elements."""
        self._setup_window_properties()
        self._create_central_widget()
        self._create_tab_widget()
        self._create_status_bar()
        self._setup_zoom_controls()
        self.showMaximized()

        logging.debug("UI elements initialized")

    def _setup_window_properties(self) -> None:
        """Sets up the window properties."""
        self.setWindowTitle("Image Processing Application")
        self.setGeometry(100, 100, *self.DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(*self.MIN_WINDOW_SIZE)

    def _create_central_widget(self) -> None:
        """Creates and sets up the central widget."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(self.SPACING)

    def _create_tab_widget(self) -> None:
        """Creates the tab widget."""
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)
        self.tab_widget.setDocumentMode(True)
        self.main_layout.addWidget(self.tab_widget)

    def _create_status_bar(self) -> None:
        """Creates and sets up the status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet(
            "QStatusBar { border-top: 1px solid gray; }")

    def _setup_zoom_controls(self) -> None:
        """Sets up the zoom controls in the status bar."""
        # Zoom controls widget
        zoom_widget = QWidget()
        zoom_layout = QHBoxLayout(zoom_widget)
        zoom_layout.setContentsMargins(0, 0, 0, 0)

        # Separator line before zoom controls
        self.zoom_separator = QFrame()
        self.zoom_separator.setFrameShape(QFrame.VLine)
        self.zoom_separator.setFrameShadow(QFrame.Sunken)

        # Create zoom controls
        self.zoom_label = QLabel("Zoom:")
        self.zoom_in_button = self._create_zoom_button(
            "Zoom In +", "Zoom in on image")
        self.zoom_out_button = self._create_zoom_button(
            "Zoom Out -", "Zoom out on image")
        self.reset_zoom_button = self._create_zoom_button(
            "Original Size", "Reset to original size")
        self.fit_to_view_button = self._create_zoom_button(
            "Fit to Screen", "Fit image to view")

        # Add controls to the layout
        zoom_layout.addWidget(self.zoom_separator)
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(self.zoom_in_button)
        zoom_layout.addWidget(self.zoom_out_button)
        zoom_layout.addWidget(self.reset_zoom_button)
        zoom_layout.addWidget(self.fit_to_view_button)

        # Add zoom widget to the status bar
        self.status_bar.addPermanentWidget(zoom_widget)

        # Hide zoom controls by default
        self.zoom_separator.hide()
        self.zoom_label.hide()
        self.zoom_in_button.hide()
        self.zoom_out_button.hide()
        self.reset_zoom_button.hide()
        self.fit_to_view_button.hide()

    def _create_zoom_button(self, text: str, tooltip: str) -> QPushButton:
        """
        Creates a zoom control button.

        Args:
            text: Button text
            tooltip: Button tooltip text

        Returns:
            QPushButton: The created button
        """
        button = QPushButton(text)
        button.setToolTip(tooltip)
        button.setMaximumHeight(25)
        return button

    def add_tab(self, widget: QWidget, title: str) -> None:
        """
        Adds a new tab to the tab widget.

        Args:
            widget: The widget for the tab
            title: The tab title
        """
        try:
            self.tab_widget.addTab(widget, title)
            logging.debug(f"New tab added: {title}")
        except Exception as e:
            logging.error(f"Error adding tab: {str(e)}")
            self.show_error(
                "Tab Addition Error",
                f"Failed to add tab: {str(e)}")

    def show_status_message(
            self,
            message: str,
            timeout: Optional[int] = None) -> None:
        """
        Displays a message in the status bar.

        Args:
            message: The message to display
            timeout: Message disappearance time (ms), None for permanent
        """
        self.status_bar.showMessage(
            message, timeout or self.STATUS_BAR_TIMEOUT)
        logging.debug(f"Status message displayed: {message}")

    def update_zoom_display(self, zoom_level: float) -> None:
        """
        Updates the zoom level display.

        Args:
            zoom_level: The zoom level (1.0 = 100%)
        """
        self.zoom_label.setText(f"Zoom: {zoom_level:.0%}")

    def show_error(self, title: str, message: str) -> None:
        """
        Displays an error message.

        Args:
            title: Message title
            message: Error message text
        """
        QMessageBox.critical(self, title, message)
        logging.error(f"Error message displayed - {title}: {message}")

    def closeEvent(self, event) -> None:
        """
        Handles the window close event.

        Args:
            event: The close event
        """
        reply = QMessageBox.question(
            self,
            'Close Application',
            "Are you sure you want to close the application?\nAny unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No)

        if reply == QMessageBox.Yes:
            logging.info("Application close confirmed")
            event.accept()
        else:
            logging.info("Application close canceled")
            event.ignore()

    def get_current_tab(self) -> Optional[QWidget]:
        """
        Returns the currently selected tab.

        Returns:
            Optional[QWidget]: The active tab widget or None
        """
        return self.tab_widget.currentWidget()