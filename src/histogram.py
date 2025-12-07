import skimage


def hist(image, nbins):
    """implement your solution and documnet this functoin"""
    return skimage.exposure.histogram(image, nbins=nbins)

def equalize_hist(image, nbins):
    """implement your solution and documnet this functoin"""
    return skimage.exposure.equalize_hist(image, nbins=nbins)

