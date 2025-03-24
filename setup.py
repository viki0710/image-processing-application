from setuptools import find_packages, setup

setup(
    name="image_processor",
    packages=find_packages(),
    version="1.0.0",
    install_requires=[
        "numpy==1.26.0",
        "opencv_python==4.10.0.84", 
        "Pillow==10.4.0",
        "PyQt5==5.15.11",
        "PyQt5_sip==12.15.0",
        "pytest==8.3.3",
        "Requests==2.32.3",
        "setuptools==72.2.0",
        "tensorflow==2.17.0",
        "tensorflow_hub==0.16.1",
        "tensorflow_intel==2.17.0"
    ],
    python_requires=">=3.8,<3.12",
    author="Balla Viktoria",
    description="Image processing application using pre-trained neural networks",
)