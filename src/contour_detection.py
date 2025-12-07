import cv2

def get_contours(image):
    """implement your solution and document this function"""
    return cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def get_area(contour):
    """implement your solution and document this function"""
    return cv2.contourArea(contour)

def get_bounding_rects(contour):
    """implement your solution and document this function"""
    return cv2.boundingRect(contour)