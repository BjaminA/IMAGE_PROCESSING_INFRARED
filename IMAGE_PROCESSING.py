import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import logging

# Note: Assumes input arrays are 2D numpy arrays with x-axis as axis 0 and y-axis as axis 1.

# For visualization, two options are provided: OpenCV and Matplotlib. Depending on the user's preference or the specific needs
# of the visualization task, one can choose between them. OpenCV displays the image in a new window, while Matplotlib shows it inline
# (useful in Jupyter notebooks or similar environments).

# Converts the input array to greyscale and rescales to 0-255 range. This is essential because many OpenCV functions 
# require the array to be in uint8 format.
def greyscale_int_array(array):
    h_val = np.max(array)
    l_val = np.min(array)
    norm_array = 255 * (array - l_val) / (h_val - l_val)
    norm_array = norm_array.astype(np.uint8)
    return norm_array, h_val, l_val

# Applies Gaussian blurring to smooth the image. This can help reduce noise and detail. The kernel size (side) must be an odd integer,
# and std deviates the standard deviation in the X and Y directions.
def array_blur_G(array, side, std):
    blurred_array = cv2.GaussianBlur(array, (side, side), std)
    return blurred_array

# Applies Median blurring, which is particularly effective at removing noise while preserving edges. The kernel size (side) 
# must be an odd integer.
def array_blur_M(array, side):
    norm_array, h_val, l_val = greyscale_int_array(array)
    blurred_array = cv2.medianBlur(norm_array, side)
    renorm_array = ((h_val - l_val) / 255) * blurred_array + l_val
    return renorm_array

# Calculates the Laplacian of the array, which is a measure of the rate at which the gradient of the array values changes.
# This can be used to detect edges.
def laplacian(array):
    laplacian_array = cv2.Laplacian(array, cv2.CV_64F)
    return laplacian_array

# Returns a list of contours found in the array after applying a binary threshold. Contours are useful for object detection
# and shape analysis.
def contours_list(array, threshold):
    _, binary_array = cv2.threshold(array, threshold, 255, cv2.THRESH_BINARY)
    binary_array = binary_array.astype(np.uint8)  # Necessary for findContours to work
    contours, _ = cv2.findContours(binary_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Draws contours on the image where the area of the contour is above a minimum threshold. Useful for highlighting
# significant features in an image.
def draw_contours(image, array, threshold, min_area):
    contours = contours_list(array, threshold)
    significant_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    bgr_image = cv2.cvtColor(np.transpose(np.array(image), (1, 0, 2)), cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr_image, significant_contours, -1, (0, 0, 0), 1)
    cv2.imshow('RGB Image', np.transpose(bgr_image, (1, 0, 2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Counts the number of contours with an area above the specified minimum. This can be used to quantify features or objects in an image.
def count_contours(array, threshold, min_area):
    contours = contours_list(array, threshold)
    significant_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    return len(significant_contours)

# The following functions (e.g., `draw_contours_mod2`, `otsu_draw_contours`, `adapt_otsu_draw_contours2`, etc.) extend the basic
# contour detection and drawing functionality by incorporating more advanced thresholding techniques like Otsu's method or
# adaptive thresholding, or by applying different processing based on contour sizes. These methods allow for more nuanced
# and effective image analysis, especially in complex or variable lighting conditions.

# It's important to note that while these functions are tailored for specific use cases, the principles behind them are widely
# applicable in image processing
