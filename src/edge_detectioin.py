import cv2

def canny_edge_detector(image, threshold1=50, threshold2=150):
        """implement your solution and documnet this functoin"""
        return cv2.Canny(image, threshold1, threshold2)