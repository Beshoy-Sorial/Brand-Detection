import cv2
from matplotlib import image

def bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
    """implement your solution and documnet this functoin"""
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)

def gaussian_blur(image, kernel_size=(5, 5), sigmaX=0):
    """implement your solution and documnet this functoin"""
    return cv2.GaussianBlur(image, kernel_size, sigmaX)