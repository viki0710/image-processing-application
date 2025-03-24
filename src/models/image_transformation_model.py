"""
This module is used for executing image transformation operations, including style transfer,
super-resolution, image colorization, and managing states through history tracking.
"""

import base64
import json
import logging
import os
import sys
import time
from io import BytesIO
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

# Get the application path
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle (exe)
    application_path = sys._MEIPASS
else:
    # If the application is run from a Python interpreter
    application_path = os.path.dirname(os.path.abspath(__file__))

# Use application_path to open config.json
config_path = os.path.join(application_path, '..', '..', 'config.json')
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

API_KEY = config['DEEPAI_API_KEY']


class ImageTransformationModel:
    """
    Image transformation model class.
    """

    def __init__(self) -> None:
        self.current_image: Optional[np.ndarray] = None
        self.scale: float = 1.0
        self._load_style_transfer_model()
        self.api_key = API_KEY
        self.history: List[Tuple[str, np.ndarray]] = []
        self.current_index: int = -1

    def _load_style_transfer_model(self) -> None:
        """Loads the style transfer model with multiple attempts."""
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            try:
                cache_dir = os.path.join(os.path.expanduser("~"), ".keras")
                if os.path.exists(cache_dir):
                    for root, dirs, files in os.walk(cache_dir):
                        for file in files:
                            if file.endswith('.pb'):
                                os.remove(os.path.join(root, file))

                model_dir = "model_cache"
                os.makedirs(model_dir, exist_ok=True)
                os.environ['TFHUB_CACHE_DIR'] = os.path.abspath(model_dir)

                self.style_transfer_model = hub.load(
                    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
                )
                logging.info("Style transfer model successfully loaded")
                return

            except Exception as e:
                attempt += 1
                logging.warning(
                    f"Attempt {attempt}/{max_attempts}: Error loading model: {str(e)}")
                if attempt < max_attempts:
                    time.sleep(2)
                else:
                    logging.error(
                        "Failed to load style transfer model")
                    self.style_transfer_model = None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses the image."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image.astype(np.float32)

    def apply_style_transfer(self,
                             content_image: np.ndarray,
                             style_image: np.ndarray) -> Optional[np.ndarray]:
        """Applies style transfer."""
        try:
            if self.style_transfer_model is None:
                self._load_style_transfer_model()
                if self.style_transfer_model is None:
                    raise ValueError("Style transfer model is not available")

            content = self.preprocess_image(content_image)
            style = self.preprocess_image(style_image)

            outputs = self.style_transfer_model(
                tf.constant(content), tf.constant(style))
            stylized_image = outputs[0]

            result = stylized_image.numpy()
            result = np.squeeze(result, axis=0)
            result = (result * 255).astype(np.uint8)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            logging.info("Style transfer successfully executed")
            return result

        except Exception as e:
            logging.error(f"Error during style transfer: {str(e)}")
            return None

    def get_current_image(self) -> Optional[np.ndarray]:
        """
        Returns the current image according to the display scale.

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

    def set_current_image(self, image: np.ndarray) -> None:
        """Sets the current and original image."""
        self.current_image = image.copy()
        self.original_image = image.copy()
        self.clear_history()
        self.add_to_history("Original image", self.original_image)

    def super_resolution(self, image: np.ndarray,
                         scale_factor: int = 2) -> Optional[np.ndarray]:
        """
        Applies super-resolution technique to an image.

        Args:
            image: The input image.
            scale_factor: The magnification level (default: 2).

        Returns:
            Optional[np.ndarray]: The magnified image or None in case of error.
        """
        temp_file = 'temp_sr_image.jpg'
        try:
            cv2.imwrite(temp_file, image)

            with open(temp_file, 'rb') as f:
                response = requests.post(
                    "https://api.deepai.org/api/torch-srgan",
                    headers={'api-key': API_KEY},
                    files={'image': f}
                )
                f.close()

            os.remove(temp_file)

            if response.status_code == 200:
                result = response.json()
                sr_response = requests.get(result['output_url'])
                sr_image = np.asarray(
                    bytearray(
                        sr_response.content),
                    dtype="uint8")
                return cv2.imdecode(sr_image, cv2.IMREAD_COLOR)

        except Exception as e:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except BaseException:
                    pass
            logging.error(f"Error during super-resolution: {str(e)}")
            return None

        return None

    def _image_to_base64(self, image: np.ndarray) -> str:
        """
        Converts the image to base64 format.

        Args:
            image: The image to convert.

        Returns:
            str: The base64 encoded image in string format.
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image)

        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def colorize_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Colorizes a black and white image using an external API.

        Args:
            image: The image to colorize.

        Returns:
            Optional[np.ndarray]: The colorized image or None in case of error.
        """
        temp_file = 'temp_image.jpg'
        try:
            # Save the original, large image here
            print(f"Model - Input image size: {image.shape}")
            print(
                f"Model - self.current_image size: {self.current_image.shape}")
            cv2.imwrite(temp_file, self.current_image)

            with open(temp_file, 'rb') as f:
                response = requests.post(
                    "https://api.deepai.org/api/colorizer",
                    headers={'api-key': API_KEY},
                    files={'image': f}
                )
                f.close()

            os.remove(temp_file)

            if response.status_code == 200:
                result = response.json()
                colored_response = requests.get(result['output_url'])
                colored_image = np.asarray(
                    bytearray(
                        colored_response.content),
                    dtype="uint8")
                return cv2.imdecode(colored_image, cv2.IMREAD_COLOR)

        except Exception as e:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except BaseException:
                    pass
            logging.error(f"Error during colorization: {str(e)}")
            return None

        return None

    def clear_history(self) -> None:
        """Clears the history."""
        self.history.clear()
        self.current_index = -1

    def add_to_history(self, operation_name: str, image: np.ndarray) -> None:
        """
        Adds a new operation to the history.

        Args:
            operation_name: The name of the operation
            image: The image resulting from the operation
        """
        self.history = self.history[:self.current_index + 1]

        self.history.append((operation_name, image.copy()))
        self.current_index = len(self.history) - 1
        self.current_image = image.copy()

    def can_undo(self) -> bool:
        """Checks if there is an operation that can be undone."""
        return self.current_index > 0

    def can_redo(self) -> bool:
        """Checks if there is an operation that can be redone."""
        return self.current_index < len(self.history) - 1

    def undo(self) -> None:
        """Undoes the last operation."""
        if self.can_undo():
            self.current_index -= 1
            _, image = self.history[self.current_index]
            self.current_image = image.copy()

    def redo(self) -> None:
        """Redoes the last undone operation."""
        if self.can_redo():
            self.current_index += 1
            _, image = self.history[self.current_index]
            self.current_image = image.copy()

    def reset_image(self) -> None:
        """Restores the original image."""
        if self.original_image is not None:
            # Instead of clearing the entire history, just return to the first element
            if len(self.history) > 0:
                self.current_index = 0
                self.current_image = self.history[0][1].copy()
                logging.info("Image restored to original state")
            else:
                # If there's no history yet, add the original image now
                self.current_image = self.original_image.copy()
                self.add_to_history("Original image", self.current_image)

    def set_state_to_history_index(self, index: int) -> None:
        """
        Restores the image to a previous state.

        Args:
            index: The desired state index in the history

        Raises:
            ValueError: If the index is invalid
        """
        if not 0 <= index < len(self.history):
            raise ValueError("Invalid history index")

        self.current_index = index
        _, image = self.history[index]
        self.current_image = image.copy()
        logging.info(f"State restored to history element {index}")