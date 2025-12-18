import numpy as np

def knn_matches(des_1, des_2, k=2):
    """
    Find k-nearest neighbors for each descriptor in des_1 from des_2.
    
    This function implements a brute-force k-NN matcher that finds the k closest
    descriptors in des_2 for each descriptor in des_1 using Euclidean distance (L2 norm).
    
    Args:
        des_1: query descriptors
        des_2: query descriptors
        k: int 
    
    Returns:
        matches: list of lists
    
    Note:
        - Returns cv2.DMatch-like objects with queryIdx, trainIdx, and distance attributes
        - If des_2 has fewer than k descriptors, returns all available matches
    """
    if des_1 is None or des_2 is None:
        return []
    
    n_query = des_1.shape[0]
    n_train = des_2.shape[0]
    
    # Adjust k if there are fewer train descriptors than k
    k_actual = min(k, n_train)
    
    matches = []
    
    for i in range(n_query):
        # Compute L2 (Euclidean) distances from query descriptor i to all train descriptors
        # Using vectorized computation for efficiency
        diffs = des_2 - des_1[i]
        distances = np.sqrt(np.sum(diffs ** 2, axis=1))
        
        # Find indices of k smallest distances
        # np.argpartition is more efficient than full sort for small k
        if k_actual < n_train:
            k_indices = np.argpartition(distances, k_actual - 1)[:k_actual]
            # Sort the k indices by their distances
            k_indices = k_indices[np.argsort(distances[k_indices])]
        else:
            # If k >= n_train, sort all distances
            k_indices = np.argsort(distances)
        
        # Create DMatch-like objects for the k nearest neighbors
        query_matches = []
        for j in k_indices:
            match = DMatch(queryIdx=i, trainIdx=int(j), distance=float(distances[j]))
            query_matches.append(match)
        
        matches.append(query_matches)
    
    return matches


class DMatch:
    """
    Simple class to mimic cv2.DMatch structure.
    """
    def __init__(self, queryIdx, trainIdx, distance, imgIdx=-1):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance
        self.imgIdx = imgIdx  # Added for full cv2.DMatch compatibility
    
    def __repr__(self):
        return f"DMatch(queryIdx={self.queryIdx}, trainIdx={self.trainIdx}, distance={self.distance:.4f})"

