# IMAGE_PROCESSING_INFRARED
This Python script is designed for advanced image processing and analysis, utilizing the OpenCV library to perform various operations such as blurring, edge detection, contour detection, and thresholding. The primary focus is on manipulating and analyzing 2D numpy arrays that represent images, with a special emphasis on greyscale image processing for applications like object detection, shape analysis, and feature quantification in images. The script includes functions for:

Rescaling images to the 0-255 range and converting them to uint8 data type, which is a prerequisite for many OpenCV functions.
Applying Gaussian and median blurring to smooth images, which helps in noise reduction and edge preservation.
Detecting edges using the Laplacian method and Canny edge detection, crucial for object and feature detection.
Finding and drawing contours based on single and double threshold algorithms, including advanced techniques combining Otsu's method and adaptive thresholding for more accurate contour detection.
Counting significant contours to quantify features within an image.
Demonstrating the usage of Maximally Stable Extremal Regions (MSER) for stable feature detection across various scales.
The script is versatile and can be adapted for a wide range of image processing tasks, particularly in fields such as computer vision, digital image analysis, and automated inspection systems.

README
Getting Started
Before running the script, ensure you have Python installed on your system. This code is compatible with Python 3.x versions. You will also need to install OpenCV, a powerful open-source computer vision and machine learning software library.

Prerequisites
Python 3.x
pip (Python package manager)
OpenCV library
Installation
Install Python 3.x: Download and install Python from python.org. Make sure to add Python to your system's PATH.
Install OpenCV: Run the following command in your terminal or command prompt to install OpenCV:
bash
Copy code
pip install opencv-python
Running the Script
Save the script in a .py file, for example, image_processing.py.
Open your terminal or command prompt.
Navigate to the directory where image_processing.py is saved.
Run the script using Python:
bash
Copy code
python image_processing.py
Usage
The script is modular, with each function designed to perform a specific image processing task. You can call these functions with your image data (as 2D numpy arrays) to apply various processing techniques. Here's a quick guide to using some of the key functions:

Greyscale and Rescale Image:
python
Copy code
grey_scaled_image, original_max, original_min = greyscale_int_array(your_image_array)
Apply Gaussian Blur:
python
Copy code
blurred_image = array_blur_G(grey_scaled_image, kernel_size, standard_deviation)
Detect Edges:
python
Copy code
edges = laplacian(grey_scaled_image)
Find and Draw Contours:
python
Copy code
draw_contours(your_image, grey_scaled_image, threshold_value, minimum_contour_area)
Replace your_image_array, kernel_size, standard_deviation, threshold_value, and minimum_contour_area with your actual data and parameters.


