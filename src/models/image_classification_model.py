"""
This module manages image classification models, including pre-trained and custom models.
It enables image classification as well as adding and managing custom models.
"""

import logging
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf


class UnsupportedModelError(Exception):
    """Custom exception for unsupported models."""

    def __init__(self, model_name: str):
        self.message = f"Unsupported model: {model_name}"
        super().__init__(self.message)


class ImageClassificationModel:
    """
    Image classification model class that uses various pre-trained neural networks for image classification.

    Attributes:
        SUPPORTED_MODELS (dict): Supported models and their associated utility functions
        MIN_TOP_N (int): Minimum allowed value for the top_n parameter
        MAX_TOP_N (int): Maximum allowed value for the top_n parameter
        model_name (str): Name of the currently used model
        model (tf.keras.Model): The loaded Keras model
        preprocess_input (Callable): Preprocessing function for the model
        decode_predictions (Callable): Prediction decoding function for the model
    """

    SUPPORTED_MODELS = {
        "mobilenetv2": (
            tf.keras.applications.MobileNetV2,
            tf.keras.applications.mobilenet_v2),
        "resnet50": (
            tf.keras.applications.ResNet50,
            tf.keras.applications.resnet50),
        "inceptionv3": (
            tf.keras.applications.InceptionV3,
            tf.keras.applications.inception_v3)}

    CUSTOM_MODELS_DIR = "custom_models"

    MIN_TOP_N = 1
    MAX_TOP_N = 10

    def __init__(self, model_name: str = "mobilenetv2") -> None:
        """
        Initialize the image classification model.

        Args:
            model_name: Name or path of the model to use

        Raises:
            UnsupportedModelError: If the specified model is not supported
        """
        self.model_name = model_name.lower()
        self.model: tf.keras.Model
        self.preprocess_input: Callable
        self.decode_predictions: Callable
        self.custom_labels: Optional[List[str]] = None

        os.makedirs(self.CUSTOM_MODELS_DIR, exist_ok=True)

        if os.path.exists(os.path.join(self.CUSTOM_MODELS_DIR, model_name)):
            self._load_custom_model(model_name)
        else:
            self.model, self.preprocess_input, self.decode_predictions = self._load_model()

        self.current_image = None
        self.scale = 1.0

    # src/models/image_classification_model.py

    def _load_model(self) -> Tuple[tf.keras.Model, Callable, Callable]:
        """
        Load the selected model and its associated functions.
        """
        if self.model_name not in self.SUPPORTED_MODELS:
            raise UnsupportedModelError(self.model_name)

        try:
            # Determine base path
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))

            # Set cache directory
            cache_dir = os.path.join(base_path, 'model_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Set TensorFlow cache
            os.environ['TFHUB_CACHE_DIR'] = cache_dir
            os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

            # Load model
            model_class, utils = self.SUPPORTED_MODELS[self.model_name]
            
            # Explicitly set the image loading mode
            tf.keras.backend.clear_session()
            
            model = model_class(weights='imagenet')
            
            logging.info(f"Model successfully loaded: {self.model_name}")
            
            return model, utils.preprocess_input, utils.decode_predictions

        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def _load_custom_model(self, model_path: str) -> None:
        """
        Load a custom model and its associated labels.

        Args:
            model_path: Path to the model

        Raises:
            ValueError: If loading the model fails
        """
        try:
            model_dir = os.path.join(self.CUSTOM_MODELS_DIR, model_path)

            self.model = tf.keras.models.load_model(
                os.path.join(model_dir, 'model.h5'))

            labels_path = os.path.join(model_dir, 'labels.txt')
            if os.path.exists(labels_path):
                with open(labels_path, 'r', encoding='utf-8') as f:
                    self.custom_labels = [line.strip()
                                          for line in f.readlines()]

            self.preprocess_input = lambda x: x / 255.0
            self.decode_predictions = self._decode_custom_predictions

            logging.info(f"Custom model successfully loaded: {model_path}")

        except Exception as e:
            logging.error(f"Error loading custom model: {str(e)}")
            raise ValueError(
                f"Failed to load custom model: {
                    str(e)}")

    def _decode_custom_predictions(
            self, predictions: np.ndarray, top: int = 5) -> List[List[Tuple[str, str, float]]]:
        """
        Decode predictions from a custom model.

        Args:
            predictions: Predictions made by the model
            top: Number of top results to return

        Returns:
            List[List[Tuple[str, str, float]]]: List of decoded predictions
        """
        results = []
        for pred in predictions:
            top_indices = pred.argsort()[-top:][::-1]
            result = []
            for idx in top_indices:
                label = self.custom_labels[idx] if self.custom_labels else f"class_{idx}"
                result.append(('custom', label, float(pred[idx])))
            results.append(result)
        return results

    def add_custom_model(
            self,
            model_path: str,
            labels_path: Optional[str] = None) -> None:
        """
        Add a new custom model to the supported models.

        Args:
            model_path: Path to the .h5 model file
            labels_path: Path to the label file (optional)

        Raises:
            ValueError: If the files are invalid or copying fails
        """
        try:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            model_dir = os.path.join(self.CUSTOM_MODELS_DIR, model_name)

            os.makedirs(model_dir, exist_ok=True)

            import shutil
            shutil.copy2(model_path, os.path.join(model_dir, 'model.h5'))

            if labels_path and os.path.exists(labels_path):
                shutil.copy2(
                    labels_path, os.path.join(
                        model_dir, 'labels.txt'))

            logging.info(f"Custom model successfully added: {model_name}")

        except Exception as e:
            logging.error(f"Error adding custom model: {str(e)}")
            raise ValueError(
                f"Failed to add custom model: {
                    str(e)}")

    def get_available_models(self) -> Dict[str, str]:
        """
        Return all available models.

        Returns:
            Dict[str, str]: Names and types of models (built-in/custom)
        """
        models = {}

        for model_name in self.SUPPORTED_MODELS.keys():
            models[model_name] = "built-in"

        if os.path.exists(self.CUSTOM_MODELS_DIR):
            for model_name in os.listdir(self.CUSTOM_MODELS_DIR):
                if os.path.exists(
                    os.path.join(
                        self.CUSTOM_MODELS_DIR,
                        model_name,
                        'model.h5')):
                    models[model_name] = "custom"

        return models

    def _validate_input(self, image: np.ndarray, top_n: int) -> None:
        """
        Verify the correctness of input parameters.

        Args:
            image: The image to validate
            top_n: The number of requested predictions

        Raises:
            ValueError: If any input parameter is invalid
        """
        if image is None:
            raise ValueError("Input image cannot be None")

        if len(image.shape) != 3:
            raise ValueError(
                "Input image must be 3-dimensional (height, width, channels)")

        if image.shape[2] != 3:
            raise ValueError(
                "Input image must have 3 channels (BGR)")

        if not self.MIN_TOP_N <= top_n <= self.MAX_TOP_N:
            raise ValueError(
                f"The value of top_n must be between {
                    self.MIN_TOP_N} and {
                    self.MAX_TOP_N}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for the model.

        Args:
            image: Input image in BGR format

        Returns:
            np.ndarray: The preprocessed image

        Raises:
            RuntimeError: If an error occurs during preprocessing
        """
        input_size = (
            299,
            299) if self.model_name == "inceptionv3" else (
            224,
            224)
        try:
            image_resized = cv2.resize(image, input_size)
            image_array = np.expand_dims(image_resized, axis=0)
            return self.preprocess_input(image_array)
        except Exception as e:
            raise RuntimeError(f"Error during preprocessing: {str(e)}")

    def classify_image(self, image: np.ndarray,
                       top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Classify the given image and return the top N predictions.

        Args:
            image: The image to classify in BGR format
            top_n: Number of top results to return

        Returns:
            List[Tuple[str, float]]: List of top_n predictions in (class_name, probability) format

        Raises:
            ValueError: If the input parameters are invalid
            RuntimeError: If an error occurs during prediction
        """
        self._validate_input(image, top_n)

        try:
            processed_image = self.preprocess_image(image)
            predictions = self.model.predict(processed_image)
            decoded_predictions = self.decode_predictions(
                predictions, top=top_n)
            return [(pred[1], pred[2]) for pred in decoded_predictions[0]]
        except RuntimeError as e:
            raise e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error occurred during image classification: {
                    str(e)}")

    def get_current_image(self) -> Optional[np.ndarray]:
        """
        Return the current image.

        Returns:
            Optional[np.ndarray]: The current image or None if no image is loaded
        """
        return self.current_image

    def set_current_image(self, image: np.ndarray) -> None:
        """
        Set the current image.

        Args:
            image: The image to set
        """
        self.current_image = image