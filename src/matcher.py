import cv2

def knn_matches(des_1, des_2):
    """implement your solution and documnet this functoin"""
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    return bf.knnMatch(des_1, des_2, k=2)
     