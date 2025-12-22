import numpy as np
from numba import jit, prange


@jit(nopython=True, parallel=True, fastmath=True)
def compute_distances_and_knn(des_1, des_2, k):
    """
    Compute k-nearest neighbors using Numba for speed.
    Returns indices and distances for k nearest neighbors.
    """
    n_query = des_1.shape[0]
    n_train = des_2.shape[0]
    k_actual = min(k, n_train)

    # Preallocate output arrays
    indices = np.zeros((n_query, k_actual), dtype=np.int32)
    distances = np.zeros((n_query, k_actual), dtype=np.float32)

    # Parallel loop over query descriptors
    for i in prange(n_query):
        # Compute distances from query descriptor i to all train descriptors
        dists = np.zeros(n_train, dtype=np.float32)
        for j in range(n_train):
            dist_sq = 0.0
            for d in range(des_1.shape[1]):
                diff = des_2[j, d] - des_1[i, d]
                dist_sq += diff * diff
            dists[j] = np.sqrt(dist_sq)

        # Find k smallest distances using partial sort
        # Simple selection for k smallest elements
        temp_indices = np.arange(n_train, dtype=np.int32)

        # Partial sort to get k smallest
        for ki in range(k_actual):
            min_idx = ki
            for j in range(ki + 1, n_train):
                if dists[temp_indices[j]] < dists[temp_indices[min_idx]]:
                    min_idx = j
            # Swap
            temp_indices[ki], temp_indices[min_idx] = (
                temp_indices[min_idx],
                temp_indices[ki],
            )

        # Store results
        for ki in range(k_actual):
            idx = temp_indices[ki]
            indices[i, ki] = idx
            distances[i, ki] = dists[idx]

    return indices, distances


def knn_matches(des_1, des_2, k=2):
    """
    Find k-nearest neighbors for each descriptor in des_1 from des_2.

    This function implements a brute-force k-NN matcher that finds the k closest
    descriptors in des_2 for each descriptor in des_1 using Euclidean distance (L2 norm).

    Args:
        des_1: query descriptors (numpy array)
        des_2: train descriptors (numpy array)
        k: int - number of nearest neighbors

    Returns:
        matches: list of lists of DMatch objects
    """
    if des_1 is None or des_2 is None:
        return []

    # Ensure float32 for best Numba performance
    des_1 = des_1.astype(np.float32)
    des_2 = des_2.astype(np.float32)

    # Compute k-NN using Numba
    indices, distances = compute_distances_and_knn(des_1, des_2, k)

    # Convert to DMatch objects
    matches = []
    for i in range(indices.shape[0]):
        query_matches = []
        for j in range(indices.shape[1]):
            match = DMatch(
                queryIdx=i, trainIdx=int(indices[i, j]), distance=float(distances[i, j])
            )
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
        self.imgIdx = imgIdx

    def __repr__(self):
        return f"DMatch(queryIdx={self.queryIdx}, trainIdx={self.trainIdx}, distance={self.distance:.4f})"
