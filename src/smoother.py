import numpy as np
from scipy.ndimage import convolve


def bilateral_filter(image, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Apply bilateral filter for edge-preserving smoothing.
    
    Parameters:
    -----------
    image : numpy.ndarray - Input grayscale image
    d : int - Diameter of pixel neighborhood
    sigmaColor : float - Filter sigma in color space
    sigmaSpace : float - Filter sigma in coordinate space
    
    Returns: numpy.ndarray - Filtered image
    """
    img = image.astype(np.float64)
    output = np.zeros_like(img)
    rows, cols = img.shape
    radius = d // 2
    
    # Pre-compute spatial Gaussian weights
    spatial_gauss = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            x = i - radius
            y = j - radius
            spatial_gauss[i, j] = np.exp(-(x**2 + y**2) / (2 * sigmaSpace**2))
    
    # Apply bilateral filter
    for i in range(rows):
        for j in range(cols):
            i_min = max(0, i - radius)
            i_max = min(rows, i + radius + 1)
            j_min = max(0, j - radius)
            j_max = min(cols, j + radius + 1)
            
            neighborhood = img[i_min:i_max, j_min:j_max]
            center_pixel = img[i, j]
            intensity_diff = neighborhood - center_pixel
            intensity_gauss = np.exp(-(intensity_diff**2) / (2 * sigmaColor**2))
            
            si_min = i_min - (i - radius)
            si_max = si_min + (i_max - i_min)
            sj_min = j_min - (j - radius)
            sj_max = sj_min + (j_max - j_min)
            spatial_weights = spatial_gauss[si_min:si_max, sj_min:sj_max]
            
            weights = spatial_weights * intensity_gauss
            weight_sum = np.sum(weights)
            
            if weight_sum > 0:
                output[i, j] = np.sum(weights * neighborhood) / weight_sum
            else:
                output[i, j] = center_pixel
    
    return output.astype(image.dtype)


def gaussian_blur(image, kernel_size=(5, 5), sigmaX=0):
    """
    Apply Gaussian blur for smoothing and noise reduction.
    
    Parameters:
    -----------
    image : numpy.ndarray - Input image
    kernel_size : tuple - Size of Gaussian kernel
    sigmaX : float - Standard deviation in X direction
    
    Returns: numpy.ndarray - Blurred image
    """
    ksize_y, ksize_x = kernel_size
    
    if sigmaX == 0:
        sigmaX = 0.3 * ((ksize_x - 1) * 0.5 - 1) + 0.8
    sigmaY = 0.3 * ((ksize_y - 1) * 0.5 - 1) + 0.8
    
    def gaussian_kernel_1d(size, sigma):
        kernel = np.zeros(size)
        center = size // 2
        for i in range(size):
            x = i - center
            kernel[i] = np.exp(-(x**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return kernel
    
    kernel_x = gaussian_kernel_1d(ksize_x, sigmaX)
    kernel_y = gaussian_kernel_1d(ksize_y, sigmaY)
    kernel_2d = np.outer(kernel_y, kernel_x)
    
    if len(image.shape) == 3:
        output = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            output[:, :, c] = convolve(image[:, :, c], kernel_2d, mode='reflect')
        return output.astype(image.dtype)
    else:
        output = convolve(image, kernel_2d, mode='reflect')
        return output.astype(image.dtype)

