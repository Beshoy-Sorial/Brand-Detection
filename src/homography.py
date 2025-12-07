import cv2


def ran_sac(src_pts, dst_pts):
    """implement your solution and documnet this functoin"""
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

