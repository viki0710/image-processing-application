"""
This module provides image editing functions and state management.
It enables loading, editing, undoing, and reapplying changes to images.
"""

import logging
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from src.image_editing_functions import IMAGE_EDITING_FUNCTIONS


class ImageEditingError(Exception):
    """Base exception class for image editing errors."""


class ImageLoadError(ImageEditingError):
    """Exception for image loading errors."""


class ImageProcessingError(ImageEditingError):
    """Exception for image processing errors."""


class FunctionNotFoundError(ImageEditingError):
    """Exception for calling a non-existent function."""


class ImageEditingModel:
    """
    Image editing model class that handles image manipulation operations and state management.

    Attributes:
        current_image (Optional[np.ndarray]): The currently edited image
        original_image (Optional[np.ndarray]): The original, unmodified image
        scale (float): The display scale ratio
        history (List[Tuple[str, Tuple, np.ndarray]]): History of operations (function name, parameters, result image)
        current_index (int): Current position in the history
        image_path (Optional[str]): Path to the loaded image
    """

    # Constants
    DEFAULT_SCALE = 1.0
    MIN_SCALE = 0.1
    MAX_SCALE = 10.0

    def __init__(self) -> None:
        """Initialize the image editing model."""
        self.current_image: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None
        self.scale: float = self.DEFAULT_SCALE
        self.history: List[Tuple[str, Tuple, np.ndarray]] = []
        self.current_index: int = -1
        self.image_path: Optional[str] = None

        logging.debug("ImageEditingModel initialized")

    def load_image(self, image_path: str) -> None:
        """
        Load an image from the specified path.

        Args:
            image_path: Path to the image to be loaded

        Raises:
            ImageLoadError: If the image loading fails
        """
        try:
            # Try to load the image using cv2.imdecode
            image = cv2.imdecode(
                np.fromfile(image_path, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED
            )
            
            if image is None:
                # If it failed, try with PIL
                try:
                    from PIL import Image
                    with Image.open(image_path) as pil_image:
                        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as pil_error:
                    logging.error(f"PIL loading error: {str(pil_error)}")
                    raise ImageLoadError(
                        f"Failed to load image using PIL as well: {str(pil_error)}"
                    )

            if image is None:
                raise ImageLoadError(f"Failed to load image: {image_path}")

            self.current_image = image
            self.original_image = image.copy()
            self.image_path = image_path
            self.clear_history()
            self.scale = self.DEFAULT_SCALE

            logging.info(f"Image successfully loaded: {image_path}")

        except Exception as e:
            error_msg = f"Image loading error: {str(e)}"
            logging.error(error_msg)
            raise ImageLoadError(error_msg)

    def is_grayscale(self) -> bool:
        """
        Check if the image is grayscale.

        Returns:
            bool: True if the image is grayscale, False if it's colored or no image is loaded

        Note:
            An image is grayscale if it's a 2D array or a 3D array with 1 channel
        """
        if self.current_image is None:
            return False
        return len(self.current_image.shape) == 2 or (
            len(self.current_image.shape) == 3 and self.current_image.shape[2] == 1
        )

    def clear_history(self) -> None:
        """Clear the operation history."""
        self.history.clear()
        self.current_index = -1
        logging.debug("History cleared")

    def apply_function(self, function_name: str, *
                       args: Any) -> Optional[np.ndarray]:
        """
        Apply an image processing function to the current image.

        Args:
            function_name: Name of the function to apply
            *args: Parameters to pass to the function

        Returns:
            Optional[np.ndarray]: The modified image or None in case of error

        Raises:
            FunctionNotFoundError: If the specified function doesn't exist
            ImageProcessingError: If an error occurs during image processing
        """
        try:
            if self.current_image is None:
                raise ImageProcessingError("No image loaded")

            selected_function = next(
                (func for func in IMAGE_EDITING_FUNCTIONS if func['name'] == function_name),
                None)

            if selected_function is None:
                raise FunctionNotFoundError(
                    f"The specified function was not found: {function_name}")

            if function_name == "Grayscale" and self.is_grayscale():
                logging.info(
                    "The image is already grayscale, no need for conversion.")
                return self.current_image

            new_image = selected_function['function'](
                self.current_image, *args)
            if new_image is not None:
                self.add_to_history(function_name, args, new_image)
                return new_image
            return None

        except Exception as e:
            error_msg = f"Error while applying function: {str(e)}"
            logging.error(error_msg)
            raise ImageProcessingError(error_msg)

    def add_to_history(
            self,
            function_name: str,
            args: Tuple,
            image: np.ndarray) -> None:
        """
        Add a new operation to the history.

        Args:
            function_name: Name of the applied function
            args: Parameters passed to the function
            image: The resulting image from the operation

        Note:
            When adding an operation, all operations after the current position are deleted
        """
        if not self.history:
            self.history.append(
                ("Original image", (), self.original_image.copy()))
            self.current_index = 0

        self.history = self.history[:self.current_index + 1]

        if not np.array_equal(self.current_image, image):
            self.history.append((function_name, args, image.copy()))
            self.current_index = len(self.history) - 1
            self.current_image = image.copy()

            logging.debug(
                f"New operation added to history: {function_name}")

    def undo(self) -> None:
        """Undo the last operation."""
        if self.current_index > 0:
            self.current_index -= 1
            _, _, image = self.history[self.current_index]
            self.current_image = image.copy()
            logging.info("Operation undone")

    def redo(self) -> None:
        """Redo the last undone operation."""
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            _, _, image = self.history[self.current_index]
            self.current_image = image.copy()
            logging.info("Operation reapplied")

    def reset_image(self) -> None:
        """
        Reset the image to its original state.
        """
        if self.original_image is None:
            raise ImageProcessingError("No original image to reset to")

        if self.history:
            self.current_index = 0
            self.current_image = self.history[0][2].copy()
            logging.info("Image reset to original state")

    def set_state_to_history_index(self, index: int) -> None:
        """
        Restore the image to a previous state.

        Args:
            index: The desired state index in the history

        Raises:
            ValueError: If the index is invalid
        """
        if not 0 <= index < len(self.history):
            raise ValueError("Invalid history index")

        self.current_index = index
        _, _, image = self.history[index]
        self.current_image = image.copy()
        logging.info(f"State restored to history element {index}")

    def get_history(self) -> List[Tuple[str, Tuple, np.ndarray]]:
        """
        Return the operation history.

        Returns:
            List[Tuple[str, Tuple, np.ndarray]]: List of operations (name, parameters, image)
        """
        return self.history

    def get_current_image(self) -> Optional[np.ndarray]:
        """
        Return the current image according to the display scale.

        Returns:
            Optional[np.ndarray]: The resized image or None if no image is loaded
        """
        if self.current_image is not None:
            height, width = self.current_image.shape[:2]
            new_size = (int(width * self.scale), int(height * self.scale))
            return cv2.resize(
                self.current_image,
                new_size,
                interpolation=cv2.INTER_AREA)
        return None

    def zoom(self, factor: float) -> None:
        """
        Modify the display scale.

        Args:
            factor: The scaling factor

        Note:
            The final scale remains between MIN_SCALE and MAX_SCALE
        """
        new_scale = self.scale * factor
        self.scale = max(min(new_scale, self.MAX_SCALE), self.MIN_SCALE)
        logging.debug(f"Zoom modified, new scale: {self.scale}")

    def zoom_reset(self) -> None:
        """Reset the display scale to default."""
        self.scale = self.DEFAULT_SCALE
        logging.debug("Zoom reset to default value")

    def can_undo(self) -> bool:
        """
        Check if there is an operation that can be undone.

        Returns:
            bool: True if there is an operation that can be undone
        """
        return self.current_index > 0

    def can_redo(self) -> bool:
        """
        Check if there is an operation that can be redone.

        Returns:
            bool: True if there is an operation that can be redone
        """
        return self.current_index < len(self.history) - 1