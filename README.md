# Image Processing Application

## Project Overview
This application is a Python-based desktop solution for image processing that integrates machine learning models to provide three main functionalities:

1. **Image Classification**: Automatically categorize images using pre-trained convolutional neural networks
2. **Image Transformation**: Apply AI-based transformations like style transfer, black-and-white colorization, and super-resolution
3. **Image Editing**: Perform basic and advanced image editing operations with real-time preview

The project was developed as a BSc thesis at Eötvös Loránd University, Faculty of Informatics, Department of Information Systems.

## Features

### Image Classification Module
- Classify individual images or batch process entire folders
- Choose from multiple pre-trained models (MobileNetV2, ResNet50, InceptionV3)
- Adjust classification confidence with Top-N predictions parameter
- Automatically organize images into folders based on classification results
- Export classification results to text files

### Image Transformation Module
- **Style Transfer**: Apply artistic styles from one image to another
- **Black-and-White Colorization**: Automatically add realistic colors to grayscale images
- **Super-Resolution**: Enhance image resolution using AI techniques (2x or 4x)
- Track transformation history with undo/redo functionality

### Image Editing Module
- Basic editing operations (rotate, resize, mirror)
- Apply filters (blur, sharpen)
- Adjust color properties (contrast, brightness, saturation, gamma)
- Convert to grayscale
- Track edit history with non-destructive editing workflow
- Zoom and pan controls for detailed editing

## System Requirements

### Software Requirements
- Python 3.12.5
- pip 24.2 package manager

### Python Dependencies
- NumPy 1.26.0
- OpenCV Python 4.10.0.84
- Pillow 10.4.0
- PyQt5 5.15.11
- TensorFlow 2.17.0
- TensorFlow Hub 0.16.1
- Additional dependencies listed in requirements.txt

### Hardware Requirements
- Minimum 8GB RAM
- Minimum 1GB free storage
- Dedicated or integrated graphics card
- Processor: Intel Core i5 / AMD Ryzen 5 or better
- Internet connection for API services

## Installation

1. Install Python 3.12.5 from the [official website](https://www.python.org/downloads/release/python-3125/)
2. Verify Python installation:
   ```
   python --version
   ```
3. Verify pip installation:
   ```
   pip --version
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Set up the development environment:
   ```
   python setup.py develop
   ```
6. Run the application:
   ```
   python src/main.py
   ```

**Note**: It's recommended to use a virtual environment (venv or conda) to avoid dependency conflicts.

## Architecture
The application is built using the Model-View-Controller (MVC) architectural pattern for modularity and maintainability:

- **Model**: Handles business logic and data processing
- **View**: Manages the user interface and user interaction
- **Controller**: Coordinates between the model and view components

This separation of concerns allows for easier maintenance and future extension of the application's functionality.

## Testing
The application has been thoroughly tested with:
- Automated unit tests for individual components
- Integration tests for component interactions
- Manual testing for user interface and workflows
- User testing with participants of varying technical backgrounds
- Performance testing for processor and memory usage

## Future Development Opportunities
- Enhanced user interface with drag & drop support
- Customizable keyboard shortcuts
- Dark/light theme switching
- GPU acceleration for machine learning models
- Intelligent caching system for improved memory management
- Support for user-provided custom models
- Additional AI-based transformation types

## Author
- Viktória Balla, Computer Science BSc

## Supervisor
- Dániel Varga, Assistant Professor

## License
MIT License

## Acknowledgements
This project utilizes several open-source libraries and pre-trained models:
- TensorFlow and TensorFlow Hub for machine learning functionality
- OpenCV for image processing operations
- PyQt5 for the graphical user interface
- DeepAI API for certain transformation operations
