import numpy as np
from scipy.ndimage import convolve


def gaussian_blur_internal(image, kernel_size=(5, 5), sigmaX=1.4):
    """Internal gaussian blur for edge detection"""
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
    
    output = convolve(image, kernel_2d, mode='reflect')
    return output


def canny_edge_detector(image, threshold1=50, threshold2=150):
    """
    Detect edges using the Canny edge detection algorithm.
    
    Parameters:
    -----------
    image : numpy.ndarray - Input grayscale image
    threshold1 : float - Lower threshold for hysteresis
    threshold2 : float - Upper threshold for hysteresis
    
    Returns: numpy.ndarray - Binary edge map
    """
    # Step 1: Gaussian blur
    img = image.astype(np.float64)
    blurred = gaussian_blur_internal(img, kernel_size=(5, 5), sigmaX=1.4)
    
    # Step 2: Sobel gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    
    gradient_x = convolve(blurred, sobel_x, mode='reflect')
    gradient_y = convolve(blurred, sobel_y, mode='reflect')
    
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x) * 180 / np.pi
    gradient_direction[gradient_direction < 0] += 180
    
    # Step 3: Non-maximum suppression
    rows, cols = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = gradient_direction[i, j]
            
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [gradient_magnitude[i, j-1], gradient_magnitude[i, j+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]]
            else:
                neighbors = [gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]]
            
            if gradient_magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = gradient_magnitude[i, j]
    
    # Step 4: Double threshold
    strong_edges = suppressed >= threshold2
    weak_edges = (suppressed >= threshold1) & (suppressed < threshold2)
    
    # Step 5: Edge tracking by hysteresis
    output = strong_edges.astype(np.uint8) * 255
    
    def trace_edge(i, j, visited):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= rows or cj < 0 or cj >= cols:
                continue
            if visited[ci, cj] or not weak_edges[ci, cj]:
                continue
            
            visited[ci, cj] = True
            output[ci, cj] = 255
            
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    stack.append((ci + di, cj + dj))
    
    visited = np.zeros_like(weak_edges, dtype=bool)
    for i in range(rows):
        for j in range(cols):
            if strong_edges[i, j]:
                trace_edge(i, j, visited)
    
    return output