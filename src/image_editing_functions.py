"""
Image manipulation functions using OpenCV.

This module implements various image processing operations, such as rotation, resizing,
flipping, blurring, sharpening, grayscale conversion, contrast and brightness adjustment,
saturation modification, and gamma correction.
"""

import cv2
import numpy as np


def rotate_image(image, angle):
    """Rotates the image by the specified angle."""
    if angle == 0:
        return image.copy()
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width,
         new_height),
        flags=cv2.INTER_LINEAR)

    return rotated_image


def resize_image(image, dimensions: dict):
    """
    Resizes the image based on the provided dimensions.

    Args:
        image: The image to be resized
        dimensions: Dictionary with sizing settings
            {
                'type': 'percentage' or 'absolute',
                'percentage': percentage value (if type='percentage'),
                'width': new width (if type='absolute'),
                'height': new height (if type='absolute'),
                'interpolation': interpolation method
            }
    """
    current_height, current_width = image.shape[:2]

    if dimensions['type'] == 'percentage':
        percentage = dimensions['percentage']
        new_width = int(current_width * (percentage / 100))
        new_height = int(current_height * (percentage / 100))
    else:  # absolute
        new_width = dimensions['width']
        new_height = dimensions['height']

    interpolation_method = cv2.INTER_AREA  # default
    if 'interpolation' in dimensions:
        if dimensions['interpolation'] == 'bicubic':
            interpolation_method = cv2.INTER_CUBIC
        elif dimensions['interpolation'] == 'nearest':
            interpolation_method = cv2.INTER_NEAREST
        elif dimensions['interpolation'] == 'bilinear':
            interpolation_method = cv2.INTER_LINEAR

    return cv2.resize(image, (new_width, new_height),
                      interpolation=interpolation_method)


def flip_image(image, flip_code):
    """
    Flips the image with the specified flip code.

    Args:
        image: The original image
        flip_code: The type of flip or None if no flipping is needed

    Returns:
        The flipped image or the original image if flip_code is None
    """
    if flip_code is None:
        return image.copy()
    return cv2.flip(image, flip_code)


def apply_blur(image, kernel_size):
    """Applies blur to the image with the specified kernel size."""
    # If kernel_size is even, increase it by one to make it odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_sharpen(image, intensity=1):
    """Sharpens the image with the specified intensity."""
    kernel = np.array([[-1, -1, -1], [-1, 8 + intensity, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def convert_to_grayscale(image):
    """Converts the image to grayscale."""
    if len(image.shape) == 2:  # already grayscale
        return image
    else:  # convert color image
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return grayscale_image


def adjust_contrast_brightness(image, contrast=1.0, brightness=0):
    """
    Adjusts the image contrast and brightness.

    :param image: The input image
    :param contrast: Contrast strength (1.0 is default, 2.0 means double contrast)
    :param brightness: Brightness adjustment (between -100 and 100)
    :return: The modified image
    """
    # Ensure contrast and brightness are within reasonable ranges
    # Contrast should be between 0.1 and 3.0
    contrast = max(0.1, min(contrast, 3.0))
    # Brightness should be between -100 and 100
    brightness = max(-100, min(brightness, 100))

    # Apply the contrast and brightness adjustment manually
    new_image = np.clip(
        (image * contrast + brightness),
        0,
        255).astype(
        np.uint8)
    return new_image


def adjust_saturation(image, saturation_scale=1.0):
    """
    Adjusts the image color saturation.

    :param image: The input image
    :param saturation_scale: Saturation scale (1.0 is default, 0.5 is half saturation)
    :return: The modified image
    """
    if len(image.shape) == 2:  # already grayscale
        return image
    if saturation_scale == 1.0:
        return image.copy()
    else:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = cv2.multiply(hsv_image[:, :, 1], saturation_scale)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def adjust_gamma(image, gamma=1.0):
    """
    Applies gamma correction to the image.

    :param image: The input image
    :param gamma: Gamma value (1.0 is default, 2.0 is brighter, 0.5 is darker)
    :return: The gamma-corrected image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma *
                     255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


"""List of image processing functions and their parameters."""
IMAGE_EDITING_FUNCTIONS = [
    {"name": "Rotation", "function": rotate_image, "params": ["angle"]},
    {"name": "Resize", "function": resize_image, "params": ["dimensions"]},
    {"name": "Flip", "function": flip_image, "params": ["flip_code"]},
    {"name": "Blur", "function": apply_blur, "params": ["kernel_size"]},
    {"name": "Sharpen", "function": apply_sharpen, "params": ["intensity"]},
    {"name": "Grayscale", "function": convert_to_grayscale, "params": []},
    {"name": "Contrast and brightness", "function": adjust_contrast_brightness, "params": ["contrast", "brightness"]},
    {"name": "Saturation", "function": adjust_saturation, "params": ["saturation_scale"]},
    {"name": "Gamma correction", "function": adjust_gamma, "params": ["gamma"]}
]