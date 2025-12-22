import numpy as np


def normalize_points(pts):
    """Normalize points for numerical stability."""
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))

    # Handle degenerate case (all points at same location)
    if mean_dist < 1e-8:
        # Return identity-like transform
        return np.eye(3)

    scale = np.sqrt(2) / mean_dist
    T = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )
    return T


def compute_homography_4pts(src, dst):
    """Compute homography from 4 point correspondences using DLT."""
    A = []
    for i in range(4):
        x, y = src[i]
        xp, yp = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_homography(H, pts):
    """Apply homography to points."""
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    transformed = (H @ pts_h.T).T

    # Avoid division by zero
    w = transformed[:, 2:3]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)  # Replace near-zero with small value

    return transformed[:, :2] / w


def compute_reprojection_error(H, src, dst):
    """Compute reprojection error for all points."""
    projected = apply_homography(H, src)
    errors = np.sqrt(np.sum((projected - dst) ** 2, axis=1))
    return errors


def is_valid_homography(H, max_scale=3.5):
    """
    Check if homography is geometrically reasonable.
    Rejects degenerate or unrealistic transformations.
    """
    if H is None:
        return False

    # Normalize
    H = H / H[2, 2]

    # Check if bottom-right is positive (proper normalization)
    if H[2, 2] <= 0:
        return False

    # Extract affine part (top-left 2x2)
    A = H[:2, :2]

    # Compute SVD to get scales
    try:
        U, s, Vt = np.linalg.svd(A)
    except:
        return False

    # Check for degenerate transformation (zero singular values)
    if s[0] < 1e-6 or s[1] < 1e-6:
        return False  # Degenerate transformation

    # Check scales (singular values)
    if s[0] / s[1] > max_scale or s[1] / s[0] > max_scale:
        return False  # Too much anisotropic scaling

    if s[0] > max_scale or s[0] < 1.0 / max_scale:
        return False  # Scale too extreme

    if s[1] > max_scale or s[1] < 1.0 / max_scale:
        return False  # Scale too extreme

    # Check determinant (negative = reflection, not allowed for logos)
    det = np.linalg.det(A)
    if det < 0.025:
        return False  # Reflection/mirroring

    # Check perspective distortion (bottom row should be small)
    if abs(H[2, 0]) > 0.025 or abs(H[2, 1]) > 0.025:
        return False  # Too much perspective

    return True


def ran_sac(src_pts, dst_pts, threshold=5.0, max_iters=2000, confidence=0.99):
    """
    Find homography using RANSAC algorithm.

    Args:
        src_pts: Source points (Nx2) or (Nx1x2) - OpenCV format compatible
        dst_pts: Destination points (Nx2) or (Nx1x2) - OpenCV format compatible
        threshold: RANSAC inlier threshold in pixels
        max_iters: Maximum RANSAC iterations
        confidence: Desired confidence level

    Returns:
        H: 3x3 homography matrix
        mask: Boolean array indicating inliers
    """
    # Handle OpenCV format (N, 1, 2) -> reshape to (N, 2)
    if src_pts.ndim == 3 and src_pts.shape[1] == 1:
        src_pts = src_pts.reshape(-1, 2)
    if dst_pts.ndim == 3 and dst_pts.shape[1] == 1:
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
    n_iters = max_iters
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

        # Denormalize homography to test in original space
        H_test = np.linalg.inv(T_dst) @ H_norm @ T_src
        H_test = H_test / H_test[2, 2]

        # Validate homography before testing (reject degenerate cases)
        if not is_valid_homography(H_test):
            iter_count += 1
            continue

        # Find inliers in ORIGINAL space (not normalized)
        errors = compute_reprojection_error(H_test, src_pts, dst_pts)
        inliers = errors < threshold
        n_inliers = np.sum(inliers)

        # Update best model
        if n_inliers > best_count:
            best_count = n_inliers
            best_inliers = inliers
            best_H = H_test  # Store the denormalized homography!

            # Update number of iterations adaptively
            inlier_ratio = n_inliers / n_pts
            # Avoid log(0) when inlier_ratio is very close to 1
            if inlier_ratio > 0 and inlier_ratio < 0.9999:
                n_iters = min(
                    n_iters, int(np.log(1 - confidence) / np.log(1 - inlier_ratio**4))
                )

        iter_count += 1

    if best_H is None:
        return None, np.zeros(n_pts, dtype=bool)

    # Final validation of best homography
    if not is_valid_homography(best_H):
        return None, np.zeros(n_pts, dtype=bool)

    # Refine using all inliers
    if best_count >= 4:
        # Use inliers in NORMALIZED space for refinement
        src_inliers = src_norm[best_inliers]
        dst_inliers = dst_norm[best_inliers]

        # Use all inliers to compute final homography
        A = []
        for i in range(len(src_inliers)):
            x, y = src_inliers[i]
            xp, yp = dst_inliers[i]
            A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H_refined = Vt[-1].reshape(3, 3)

        if abs(H_refined[2, 2]) > 1e-8:
            H_refined = H_refined / H_refined[2, 2]

            # Denormalize
            H_final = np.linalg.inv(T_dst) @ H_refined @ T_src
            H_final = H_final / H_final[2, 2]
        else:
            # Refinement failed, use best_H from RANSAC
            H_final = best_H
    else:
        # Not enough inliers, use best_H from RANSAC
        H_final = best_H

    return H_final, best_inliers
