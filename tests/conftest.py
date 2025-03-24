import os
import sys

import pytest
from PyQt5.QtWidgets import QApplication

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


@pytest.fixture(scope="session")
def qapp():
    """Shared QApplication instance for tests."""
    return QApplication(sys.argv)


@pytest.fixture
def test_image():
    """Generate test image."""
    import numpy as np
    return np.zeros((224, 224, 3), dtype=np.uint8)
