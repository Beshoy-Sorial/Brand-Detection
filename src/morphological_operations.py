import cv2
from matplotlib import image


def dilate(image ,kernel_size=(5, 5), iterations=2):
    """implement your solution and documnet this functoin"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closed = close(image, kernel)
    return cv2.dilate(closed, kernel, iterations=iterations)


def close(image, kernel):
    """implement your solution and documnet this functoin"""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)



# not used currently
def open(image, kernel):
    """implement your solution and documnet this functoin"""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# not used currently
def erode(image, kernel_size=(5, 5), iterations=2):
    """implement your solution and documnet this functoin"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened = open(image, kernel)
    return cv2.erode(opened, kernel, iterations=iterations)