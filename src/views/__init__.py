# src/views/__init__.py
"""
User interface modules for the image processing application.

This module contains the view classes and dialogs responsible for
the application's display.
"""

from .dialogs import (AdjustmentsDialog, BlurDialog, FlipDialog, GammaDialog,
                      ResizeDialog, RotateDialog, SaturationDialog,
                      SharpenDialog, StyleTransferDialog)
from .image_classification_view import ImageClassificationView
from .image_editing_view import ImageEditingView
from .image_transformation_view import ImageTransformationView
from .main_window_view import MainWindowView

__all__ = [
    'MainWindowView',
    'ImageClassificationView',
    'ImageEditingView',
    'ImageTransformationView',
    'BlurDialog',
    'FlipDialog',
    'GammaDialog',
    'ResizeDialog',
    'RotateDialog',
    'SaturationDialog',
    'SharpenDialog',
    'AdjustmentsDialog',
    'StyleTransferDialog'
]