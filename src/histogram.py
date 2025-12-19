import numpy as np


def hist(image, nbins):
    """
    Compute the histogram of an image.
    
    Parameters:
    -----------
    image : numpy.ndarray - Input image
    nbins : int - Number of histogram bins
    
    Returns: tuple - (hist_values, bin_centers)
    """
    img_flat = image.flatten().astype(np.float64)
    img_min, img_max = img_flat.min(), img_flat.max()
    
    if img_max > img_min:
        img_normalized = (img_flat - img_min) / (img_max - img_min)
    else:
        img_normalized = np.zeros_like(img_flat)
    
    hist_values, bin_edges = np.histogram(img_normalized, bins=nbins, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers = bin_centers * (img_max - img_min) + img_min
    
    return hist_values.astype(np.int64), bin_centers


def equalize_hist(image, nbins=256):
    """
    Perform histogram equalization to enhance image contrast.
    
    Parameters:
    -----------
    image : numpy.ndarray - Input image
    nbins : int - Number of histogram bins
    
    Returns: numpy.ndarray - Equalized image (float64, range [0, 1])
    """
    img = image.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    
    if img_max > img_min:
        img_normalized = (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img, dtype=np.float64)
    
    hist_values, bin_edges = np.histogram(img_normalized.flatten(), bins=nbins, range=(0, 1))
    
    cdf = hist_values.cumsum()
    cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
    cdf_normalized = (cdf - cdf_min) / (cdf[-1] - cdf_min + 1e-10)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    img_flat = img_normalized.flatten()
    equalized_flat = np.interp(img_flat, bin_centers, cdf_normalized)
    equalized = equalized_flat.reshape(img.shape)
    
    return equalized
