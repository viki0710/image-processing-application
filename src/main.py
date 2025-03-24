"""
This module contains the entry point of the application. It starts the PyQt5-based graphical user interface.
"""

import logging
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from src.controllers.main_window_controller import MainWindowController

sys.path.append(str(Path(__file__).parent.parent))


def main():
   """The application entry point. Initializes logging, starts the Qt application, and opens the main window."""
   logging.basicConfig(filename='image_processor.log', level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')

   app = QApplication(sys.argv)
   controller = MainWindowController()
   controller.view.show()
   logging.info("Application started")
   sys.exit(app.exec_())


if __name__ == "__main__":
   main()