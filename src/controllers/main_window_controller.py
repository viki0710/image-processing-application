"""This file implements the controller for the image transformation module, which handles various AI-based image transformation operations."""

import logging
from typing import Any, Dict, Optional

from PyQt5.QtWidgets import QMessageBox, QWidget

from src.controllers.image_classification_controller import \
    ImageClassificationController
from src.controllers.image_editing_controller import ImageEditingController
from src.controllers.image_transformation_controller import \
    ImageTransformationController
from src.models.image_transformation_model import ImageTransformationModel
from src.views.image_classification_view import ImageClassificationView
from src.views.image_editing_view import ImageEditingView
from src.views.image_transformation_view import ImageTransformationView
from src.views.main_window_view import MainWindowView


class MainWindowController:
    """
    Controller for the application's main window.

    This class is responsible for coordinating communication between the main window
    and the various modules, as well as handling global operations.

    Attributes:
        MAX_SCALE (float): Maximum allowed zoom level
        MIN_SCALE (float): Minimum allowed zoom level
        DEFAULT_SCALE (float): Default zoom level
        ZOOM_STEP (float): Zoom in/out step size
    """

    # Zoom constants
    MAX_SCALE = 5.0
    MIN_SCALE = 0.2
    DEFAULT_SCALE = 1.0
    ZOOM_STEP = 1.2

    def __init__(self) -> None:
        """Initialize the main window controller."""
        self._setup_logging()
        self.view = MainWindowView()
        self.controllers = self._initialize_controllers()
        self._setup_tabs()
        self._connect_signals()
        self._update_zoom_controls_visibility(self.view.get_current_tab())
        self.view.show()

        logging.info("MainWindowController initialized")

    def _setup_logging(self) -> None:
        """Set up the logging system."""
        logging.basicConfig(
            filename='main_controller.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _initialize_controllers(self) -> Dict[str, Any]:
        """
        Initialize the module controllers.

        Returns:
            Dict[str, Any]: Dictionary of created controllers
        """
        try:
            # Create views
            classification_view = ImageClassificationView()
            transformation_view = ImageTransformationView()
            editing_view = ImageEditingView()

            # Create model for the transformation controller
            transformation_model = ImageTransformationModel()

            controllers = {
                'classification': ImageClassificationController(classification_view),
                'transformation': ImageTransformationController(
                    transformation_model,
                    transformation_view),
                'editing': ImageEditingController(editing_view)}

            # Set up callbacks
            for controller in controllers.values():
                if hasattr(controller, 'update_status_bar_callback'):
                    controller.update_status_bar_callback = self.update_status_bar

            logging.info("Module controllers successfully initialized")
            return controllers

        except Exception as e:
            logging.error(f"Error initializing controllers: {str(e)}")
            raise

    def _setup_tabs(self) -> None:
        """Set up and add module tabs."""
        try:
            tab_configs = [
                ('classification', "Image Classification"),
                ('transformation', "Image Transformation"),
                ('editing', "Image Editing")
            ]

            for module_id, title in tab_configs:
                controller = self.controllers[module_id]
                self.view.add_tab(controller.view, title)

            logging.info("Tabs successfully set up")

        except Exception as e:
            logging.error(f"Error setting up tabs: {str(e)}")
            self.view.show_error(
                "Initialization Error",
                f"Failed to create modules: {
                    str(e)}")

    def _connect_signals(self) -> None:
        """Connect events to appropriate handler functions."""
        # Zoom controls
        self.view.zoom_in_button.clicked.connect(self.zoom_in)
        self.view.zoom_out_button.clicked.connect(self.zoom_out)
        self.view.reset_zoom_button.clicked.connect(self.zoom_reset)
        self.view.fit_to_view_button.clicked.connect(self.fit_to_view)

        # Tab change event
        self.view.tab_widget.currentChanged.connect(self._on_tab_changed)

        logging.debug("Events successfully connected")

    def _on_tab_changed(self, index: int) -> None:
        """
        Handle tab change event.

        Args:
            index: Index of the newly selected tab
        """
        try:
            current_tab = self.view.tab_widget.widget(index)
            # Show/hide zoom buttons based on tab type
            self._update_zoom_controls_visibility(current_tab)
            self.update_status_bar()
            logging.debug(f"Tab change: {index}")
        except Exception as e:
            logging.error(f"Error during tab change: {str(e)}")

    def _update_zoom_controls_visibility(self, current_tab: QWidget) -> None:
        """
        Update zoom controls visibility based on the current tab.

        Args:
            current_tab: The current tab widget
        """
        # Zoom controls should only appear in the image editing module
        is_editing_tab = isinstance(current_tab, ImageEditingView)

        self.view.zoom_in_button.setVisible(is_editing_tab)
        self.view.zoom_out_button.setVisible(is_editing_tab)
        self.view.reset_zoom_button.setVisible(is_editing_tab)
        self.view.fit_to_view_button.setVisible(is_editing_tab)

        # Handle separator line (if exists)
        if hasattr(self.view, 'zoom_separator'):
            self.view.zoom_separator.setVisible(is_editing_tab)

    def update_status_bar(self) -> None:
        """Update status bar content based on the current module state."""
        try:
            current_tab = self.view.get_current_tab()

            if isinstance(current_tab, ImageEditingView):
                controller = self.controllers['editing']
                if controller.model.current_image is not None:
                    self._update_status_bar_info(controller)

            elif isinstance(current_tab, ImageClassificationView):
                controller = self.controllers['classification']
                if controller.current_image is not None:
                    self._update_status_bar_info(controller)

            elif isinstance(current_tab, ImageTransformationView):
                controller = self.controllers['transformation']
                if hasattr(
                        controller,
                        'current_image') and controller.current_image is not None:
                    self._update_status_bar_info(controller)

        except Exception as e:
            logging.error(f"Error updating status bar: {str(e)}")

    def _update_status_bar_info(self, controller: Any) -> None:
        """
        Update status bar information based on a given controller.

        Args:
            controller: The module controller
        """
        try:
            if hasattr(controller, 'model'):
                image = controller.model.current_image
                scale = controller.model.scale  # Read scale from the model
            else:
                image = controller.current_image
                scale = getattr(controller, 'scale', 1.0)

            if image is not None:
                height, width = image.shape[:2]
                zoom_percentage = int(scale * 100)

                message = f"Size: {width}x{height} px | Zoom: {zoom_percentage}%"
                self.view.show_status_message(message)
                self.view.update_zoom_display(scale)

        except Exception as e:
            logging.error(
                f"Error updating status information: {
                    str(e)}")

    def zoom_in(self) -> None:
        """Zoom in the current image."""
        try:
            controller = self._get_current_controller()
            if controller and isinstance(controller, ImageEditingController):
                if self._can_zoom_in(controller):
                    controller.zoom_in()  # Redirect to the controller
                    self.update_status_bar()
                    logging.debug("Zoom in executed")

        except Exception as e:
            logging.error(f"Error during zoom in: {str(e)}")
            self.view.show_error("Zoom In Error", str(e))

    def zoom_out(self) -> None:
        """Zoom out the current image."""
        try:
            controller = self._get_current_controller()
            if controller and isinstance(controller, ImageEditingController):
                if self._can_zoom_out(controller):
                    controller.zoom_out()  # Redirect to the controller
                    self.update_status_bar()
                    logging.debug("Zoom out executed")

        except Exception as e:
            logging.error(f"Error during zoom out: {str(e)}")
            self.view.show_error("Zoom Out Error", str(e))

    def zoom_reset(self) -> None:
        """Reset the image to original size."""
        try:
            controller = self._get_current_controller()
            if controller and isinstance(controller, ImageEditingController):
                controller.zoom_reset()  # Redirect to the controller
                self.update_status_bar()
                logging.debug("Zoom reset executed")

        except Exception as e:
            logging.error(f"Error resetting zoom: {str(e)}")
            self.view.show_error("Reset Error", str(e))

    def fit_to_view(self) -> None:
        """Fit the image to the view."""
        try:
            controller = self._get_current_controller()
            if controller and isinstance(controller, ImageEditingController):
                controller.fit_to_view()
                self.update_status_bar()
                logging.debug("Image fitted to view")

        except Exception as e:
            logging.error(f"Error fitting to view: {str(e)}")
            self.view.show_error("Fit Error", str(e))

    def _get_current_controller(self) -> Optional[Any]:
        """
        Return the controller for the current tab.

        Returns:
            Optional[Any]: The current controller or None
        """
        current_tab = self.view.get_current_tab()

        if isinstance(current_tab, ImageEditingView):
            return self.controllers['editing']
        elif isinstance(current_tab, ImageClassificationView):
            return self.controllers['classification']
        elif isinstance(current_tab, ImageTransformationView):
            return self.controllers['transformation']

        return None

    def _can_zoom_in(self, controller: Any) -> bool:
        """
        Check if zoom in is possible.
        """
        # This can be moved to the ImageEditingController
        if isinstance(controller, ImageEditingController):
            return controller.can_zoom_in()
        return False

    def _can_zoom_out(self, controller: Any) -> bool:
        """
        Check if zoom out is possible.
        """
        # This can be moved to the ImageEditingController
        if isinstance(controller, ImageEditingController):
            return controller.can_zoom_out()
        return False

    def show_about_dialog(self) -> None:
        """Display the about dialog."""
        QMessageBox.about(
            self.view,
            "About",
            "Image Processing Application\n\n"
            "Version: 1.0\n"
            "Created by: Viktória Balla\n"
            "© 2024 All rights reserved"
        )