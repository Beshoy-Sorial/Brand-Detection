import cv2
from matplotlib import image
import numpy as np


def dilate(image, kernel_size=(5, 5), iterations=2):
    """
    Perform binary dilation manually (without cv2.dilate).
    """
    img = image.copy()
    kh, kw = kernel_size
    pad_h, pad_w = kh // 2, kw // 2

    for _ in range(iterations):
        padded = np.pad(
            img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
        )

        dilated = np.zeros_like(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i : i + kh, j : j + kw]
                if np.any(region == 255):
                    dilated[i, j] = 255

        img = dilated

    return img


# not used currently
def open(image, kernel):
    """implement your solution and documnet this functoin"""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def erode(image, kernel_size=(5, 5), iterations=2):
    """
    Perform erosion manually without using cv2.erode.
    """
    img = image.copy()
    kh, kw = kernel_size
    pad_h, pad_w = kh // 2, kw // 2

    for _ in range(iterations):
        padded = np.pad(
            img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=255
        )

        eroded = np.zeros_like(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i : i + kh, j : j + kw]
                if np.all(region == 255):
                    eroded[i, j] = 255

        img = eroded

    return img
