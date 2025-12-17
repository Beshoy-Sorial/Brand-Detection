import cv2
import numpy as np

def normalize_points(pts):
    """Normalize points for numerical stability."""
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    
    # Handle degenerate case where all points are identical
    if mean_dist < 1e-8:
        return np.eye(3)
    
    scale = np.sqrt(2) / mean_dist
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    return T

def compute_homography_4pts(src, dst):
    """Compute homography from 4 point correspondences using DLT."""
    A = []
    for i in range(4):
        x, y = src[i]
        xp, yp = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
    
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def apply_homography(H, pts):
    """Apply homography to points."""
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    transformed = (H @ pts_h.T).T
    return transformed[:, :2] / transformed[:, 2:3]

def compute_reprojection_error(H, src, dst):
    """Compute reprojection error for all points."""
    projected = apply_homography(H, src)
    errors = np.sqrt(np.sum((projected - dst)**2, axis=1))
    return errors

def ran_sac(src_pts, dst_pts):
    """
    Find homography using RANSAC algorithm.
    
    Args:
        src_pts: Source points (Nx2)
        dst_pts: Destination points (Nx2)
        threshold: RANSAC inlier threshold in pixels
        max_iters: Maximum RANSAC iterations
        confidence: Desired confidence level
    
    Returns:
        H: 3x3 homography matrix
        mask: Boolean array indicating inliers
    """
    src_pts = src_pts.reshape(-1, 2)
    dst_pts = dst_pts.reshape(-1, 2)

    
    n_pts = len(src_pts)
    if n_pts < 4:
        raise ValueError("Need at least 4 points")
    
    # Normalize points for numerical stability
    T_src = normalize_points(src_pts)
    T_dst = normalize_points(dst_pts)
    
    src_norm = apply_homography(T_src, src_pts)
    dst_norm = apply_homography(T_dst, dst_pts)
    
    best_H = None
    best_inliers = None
    best_count = 0
    
    # Adaptive RANSAC
    n_iters = 2000
    threshold = 5.0
    confidence = 0.99
    iter_count = 0
    
    while iter_count < n_iters:
        # Randomly sample 4 points
        idx = np.random.choice(n_pts, 4, replace=False)
        src_sample = src_norm[idx]
        dst_sample = dst_norm[idx]
        
        # Compute homography
        try:
            H_norm = compute_homography_4pts(src_sample, dst_sample)
        except:
            iter_count += 1
            continue
        
        # Find inliers
        errors = compute_reprojection_error(H_norm, src_norm, dst_norm)
        inliers = errors < threshold
        n_inliers = np.sum(inliers)
        
        # Update best model
        if n_inliers > best_count:
            best_count = n_inliers
            best_inliers = inliers
            best_H = H_norm
            print(f"Iteration {iter_count}: New best model with {best_count} inliers")
            # Update number of iterations adaptively
            inlier_ratio = n_inliers / n_pts
            if inlier_ratio > 0 and inlier_ratio < 1:
                n_iters = min(n_iters, 
                             int(np.log(1 - confidence) / 
                                 np.log(1 - inlier_ratio**4)))
        
        iter_count += 1
    
    if best_H is None:
        return None, np.zeros(n_pts, dtype=bool)
    
    # Refine using all inliers
    if best_count >= 4:
        src_inliers = src_norm[best_inliers]
        dst_inliers = dst_norm[best_inliers]
        
        # Use all inliers to compute final homography
        A = []
        for i in range(len(src_inliers)):
            x, y = src_inliers[i]
            xp, yp = dst_inliers[i]
            A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
        
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H_refined = Vt[-1].reshape(3, 3)
        H_refined = H_refined / H_refined[2, 2]
        
        # Denormalize
        H_final = np.linalg.inv(T_dst) @ H_refined @ T_src
        H_final = H_final / H_final[2, 2]
    else:
        H_final = np.linalg.inv(T_dst) @ best_H @ T_src
        H_final = H_final / H_final[2, 2]
    
    return H_final, best_inliers


