import cv2

sift = None

def init(   
            layer_count : int = 7,
            contrast_threshold: float = 0.01,
            sigma: float = 1.414
        ):
    """implement your solution and documnet this functoin"""
    global sift
    sift = cv2.SIFT_create(
            nOctaveLayers=layer_count,
            contrastThreshold=contrast_threshold,
            sigma=sigma
        )

def calc_sift(image):
    """implement your solution and documnet this functoin"""
    sift.detectAndCompute(image, None)