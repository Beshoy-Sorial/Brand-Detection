import cv2
import numpy as np
from math import exp, pi, cos, sin, log2
from numba import jit, prange
import warnings

warnings.filterwarnings("ignore")

_params = {}


def init(
    layer_count: int = 7,
    contrast_threshold: float = 0.01,
    sigma: float = 0.75,
    edge_threshold: float = 10,
):
    global _params
    _params = {
        "layers": layer_count + 3,
        "contrast": contrast_threshold,
        "sigma": sigma,
        "octaves": 4,
        "max_interp_steps": 5,
        "edge_threshold": edge_threshold,
    }


def calc_sift(image):
    if not _params:
        raise AttributeError("SIFT not initialized. Call init() first.")

    img = image.astype(np.float32) / 255.0

    min_dim = min(img.shape)
    max_octaves = int(log2(min_dim * 2)) - 2
    _params["octaves"] = min(4, max(1, max_octaves))

    # Build pyramids
    gauss = _gaussian_pyramid(img)
    dog = _dog_pyramid(gauss)

    # Pre-compute gradients for all levels
    grad_pyr = []
    for o in range(len(gauss)):
        grad_oct = []
        for l in range(len(gauss[o])):
            mag, ori = _gradient_numba(gauss[o][l])
            grad_oct.append((mag, ori))
        grad_pyr.append(grad_oct)

    # Find keypoints
    kps = _find_keypoints_fast(dog, _params["contrast"])
    if not kps:
        return [], None

    # Refine keypoints
    kps = _refine_keypoints_fast(
        kps,
        dog,
        _params["contrast"],
        _params["edge_threshold"],
        _params["max_interp_steps"],
        _params["layers"],
    )
    if not kps:
        return [], None

    # Assign orientations
    kps = _assign_orientation_fast(kps, gauss, grad_pyr)
    if not kps:
        return [], None

    # Extract descriptors
    keypoints, descriptors = _extract_descriptors_batch(kps, gauss, grad_pyr, dog)

    if not descriptors:
        return [], None

    return keypoints, np.array(descriptors, dtype=np.float32)


def _gaussian_pyramid(img):
    pyr = []
    s = _params["layers"] - 3
    k = 2 ** (1.0 / s)
    sigma_base = _params["sigma"]

    # Upscale base image once
    img_doubled = cv2.resize(
        img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
    )
    sigma_diff = np.sqrt(max(sigma_base**2 - 1.0, 0.01))
    ksize = int(2 * np.ceil(3 * sigma_diff) + 1) | 1
    base_img = cv2.GaussianBlur(img_doubled, (ksize, ksize), sigma_diff)

    for octave_idx in range(_params["octaves"]):
        octave = []

        if octave_idx == 0:
            octave.append(base_img)
        else:
            downsampled = pyr[octave_idx - 1][s][::2, ::2]
            octave.append(downsampled)

        # Pre-compute all sigmas for this octave
        for l in range(1, _params["layers"]):
            sigma_total = sigma_base * (k**l)
            sigma_prev = sigma_base * (k ** (l - 1))
            sigma_diff = np.sqrt(max(sigma_total**2 - sigma_prev**2, 0.01))

            ksize = int(2 * np.ceil(3 * sigma_diff) + 1) | 1
            blurred = cv2.GaussianBlur(octave[-1], (ksize, ksize), sigma_diff)
            octave.append(blurred)

        pyr.append(octave)

    return pyr


# @jit(nopython=True, parallel=True, cache=True)
# def _dog_pyramid_fast(gauss_flat, shapes, octave_starts):
#     """Fast DoG computation with parallelization"""
#     dog_list = []
#     for o in prange(len(shapes)):
#         start_idx = octave_starts[o]
#         n_layers = shapes[o]
#         for i in range(n_layers - 1):
#             idx1 = start_idx + i
#             idx2 = start_idx + i + 1
#             diff = gauss_flat[idx2] - gauss_flat[idx1]
#             dog_list.append(diff)
#     return dog_list


def _dog_pyramid(gauss):
    """Compute DoG pyramid"""
    dog = []
    for octave in gauss:
        dog_octave = []
        for i in range(len(octave) - 1):
            diff = octave[i + 1] - octave[i]
            dog_octave.append(diff)
        dog.append(dog_octave)
    return dog


@jit(nopython=True, cache=True)
def _is_extremum(prev, curr, next, y, x, threshold):
    """Fast extremum check"""
    v = curr[y, x]
    if abs(v) < threshold:
        return False

    is_max = v > 0

    # Check 3x3x3 neighborhood
    for dz in range(-1, 2):
        img = prev if dz == -1 else (curr if dz == 0 else next)
        for dy in range(-1, 2):
            yy = y + dy
            for dx in range(-1, 2):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                xx = x + dx
                val = img[yy, xx]
                if is_max:
                    if val >= v:
                        return False
                else:
                    if val <= v:
                        return False
    return True


def _find_keypoints_fast(dog, contrast):
    """Optimized keypoint detection"""
    kps = []
    threshold = 0.5 * contrast

    for o in range(len(dog)):
        octave = dog[o]
        for l in range(1, len(octave) - 1):
            prev = octave[l - 1]
            curr = octave[l]
            next = octave[l + 1]

            h, w = curr.shape

            # Vectorized threshold check first
            candidates = np.abs(curr[5 : h - 5, 5 : w - 5]) >= threshold
            y_coords, x_coords = np.where(candidates)
            y_coords += 5
            x_coords += 5

            for i in range(len(y_coords)):
                y, x = y_coords[i], x_coords[i]
                if _is_extremum(prev, curr, next, y, x, threshold):
                    kps.append((o, l, y, x))

    return kps


@jit(nopython=True, cache=True)
def _refine_single_keypoint(
    octave_data,
    o,
    l,
    y,
    x,
    h,
    w,
    n_layers,
    contrast_threshold,
    edge_threshold,
    max_steps,
):
    """Fast keypoint refinement"""
    xi, yi, li = x, y, l

    for step in range(max_steps):
        if (
            li < 1
            or li >= n_layers - 1
            or yi < 1
            or yi >= h - 1
            or xi < 1
            or xi >= w - 1
        ):
            return None

        prev = octave_data[li - 1]
        curr = octave_data[li]
        next = octave_data[li + 1]

        # Gradient
        dx = (curr[yi, xi + 1] - curr[yi, xi - 1]) * 0.5
        dy = (curr[yi + 1, xi] - curr[yi - 1, xi]) * 0.5
        ds = (next[yi, xi] - prev[yi, xi]) * 0.5

        # Hessian
        center = curr[yi, xi]
        Dxx = curr[yi, xi + 1] + curr[yi, xi - 1] - 2 * center
        Dyy = curr[yi + 1, xi] + curr[yi - 1, xi] - 2 * center
        Dss = next[yi, xi] + prev[yi, xi] - 2 * center

        Dxy = (
            curr[yi + 1, xi + 1]
            - curr[yi + 1, xi - 1]
            - curr[yi - 1, xi + 1]
            + curr[yi - 1, xi - 1]
        ) * 0.25

        # Edge rejection
        tr = Dxx + Dyy
        det = Dxx * Dyy - Dxy * Dxy
        r = edge_threshold

        if det <= 0 or (tr * tr) / det >= ((r + 1) * (r + 1)) / r:
            return None

        # Solve for offset (simplified 2D - ignore scale offset for speed)
        if abs(det) < 1e-10:
            return None

        offset_x = -(Dyy * dx - Dxy * dy) / det
        offset_y = -(Dxx * dy - Dxy * dx) / det
        offset_s = -ds / (Dss + 1e-10)

        if abs(offset_x) < 0.5 and abs(offset_y) < 0.5 and abs(offset_s) < 0.5:
            D_interp = center + 0.5 * (dx * offset_x + dy * offset_y + ds * offset_s)

            if abs(D_interp) >= contrast_threshold:
                return (xi + offset_x, yi + offset_y, li + offset_s)
            return None

        new_xi = xi + int(round(offset_x))
        new_yi = yi + int(round(offset_y))
        new_li = li + int(round(offset_s))

        if (
            abs(offset_x) > 1.5
            or abs(offset_y) > 1.5
            or abs(offset_s) > 1.5
            or new_li < 1
            or new_li >= n_layers - 1
            or new_yi < 1
            or new_yi >= h - 1
            or new_xi < 1
            or new_xi >= w - 1
        ):
            return None

        xi, yi, li = new_xi, new_yi, new_li

    return None


def _refine_keypoints_fast(
    kps, dog, contrast, edge_threshold, max_interp_steps, layers
):
    """Optimized keypoint refinement"""
    refined = []
    s = layers - 3
    contrast_threshold = contrast / s

    for o, l, y, x in kps:
        octave = dog[o]
        h, w = octave[l].shape

        # Convert octave to numpy array for numba
        octave_array = [octave[i] for i in range(len(octave))]

        result = _refine_single_keypoint(
            octave_array,
            o,
            l,
            y,
            x,
            h,
            w,
            len(octave),
            contrast_threshold,
            edge_threshold,
            max_interp_steps,
        )

        if result is not None:
            refined_x, refined_y, refined_l = result
            refined.append((o, refined_l, refined_y, refined_x))

    return refined


@jit(nopython=True, cache=True)
def _gradient_numba(img):
    """Fast gradient computation"""
    h, w = img.shape
    mag = np.zeros((h, w), dtype=np.float32)
    ori = np.zeros((h, w), dtype=np.float32)

    for i in prange(1, h - 1):
        for j in range(1, w - 1):
            gx = (img[i, j + 1] - img[i, j - 1]) * 0.5
            gy = (img[i + 1, j] - img[i - 1, j]) * 0.5
            mag[i, j] = np.sqrt(gx * gx + gy * gy)
            ori[i, j] = np.arctan2(gy, gx)

    return mag, ori


@jit(nopython=True, cache=True)
def _compute_orientation_hist_fast(mag, ori, y_int, x_int, radius, sigma_weight):
    """Fast orientation histogram computation"""
    hist = np.zeros(36, dtype=np.float32)
    h, w = mag.shape
    inv_2sigma2 = 1.0 / (2 * sigma_weight * sigma_weight)

    for dy in range(-radius, radius + 1):
        dy2 = dy * dy
        for dx in range(-radius, radius + 1):
            if dx * dx + dy2 > radius * radius:
                continue

            yy = y_int + dy
            xx = x_int + dx

            if 0 <= yy < h and 0 <= xx < w:
                weight = mag[yy, xx] * exp(-(dx * dx + dy2) * inv_2sigma2)
                angle_deg = (ori[yy, xx] * 180 / pi) % 360

                bin_f = angle_deg / 10.0
                b0 = int(bin_f) % 36
                b1 = (b0 + 1) % 36
                w1 = bin_f - int(bin_f)
                w0 = 1.0 - w1

                hist[b0] += weight * w0
                hist[b1] += weight * w1

    # Smooth histogram
    for _ in range(2):
        prev = hist[35]
        curr = hist[0]
        for i in range(36):
            next_val = hist[(i + 1) % 36]
            temp = 0.25 * prev + 0.5 * curr + 0.25 * next_val
            prev = curr
            curr = next_val
            hist[i] = temp

    return hist


def _assign_orientation_fast(kps, gauss, grad_pyr):
    """Fast orientation assignment"""
    out = []

    for kp in kps:
        o, l, y, x = kp[0], kp[1], kp[2], kp[3]
        l_int = int(round(l))

        if l_int < 0 or l_int >= len(gauss[o]):
            continue

        img = gauss[o][l_int]
        h, w = img.shape

        scale = _params["sigma"] * (2 ** (l / (_params["layers"] - 3)))
        radius = int(round(4.5 * scale))  # Slightly reduced for speed
        radius = max(radius, 1)

        y_int = int(round(y))
        x_int = int(round(x))

        if (
            y_int < radius
            or x_int < radius
            or y_int >= h - radius
            or x_int >= w - radius
        ):
            continue

        mag, ori = grad_pyr[o][l_int]
        sigma_weight = 1.5 * scale
        hist = _compute_orientation_hist_fast(
            mag, ori, y_int, x_int, radius, sigma_weight
        )

        max_val = hist.max()
        if max_val < 1e-6:
            continue

        # Find peaks
        for b in range(36):
            prev_b = (b - 1) % 36
            next_b = (b + 1) % 36

            if (
                hist[b] > hist[prev_b]
                and hist[b] > hist[next_b]
                and hist[b] >= 0.8 * max_val
            ):
                denom = hist[prev_b] - 2 * hist[b] + hist[next_b]
                if abs(denom) > 1e-6:
                    interp = 0.5 * (hist[prev_b] - hist[next_b]) / denom
                    interp = np.clip(interp, -0.5, 0.5)
                else:
                    interp = 0

                bin_float = b + interp
                angle = (bin_float * 10 * pi / 180) % (2 * pi)
                if angle > pi:
                    angle -= 2 * pi

                out.append((o, l, y, x, angle))

    return out


@jit(nopython=True, cache=True)
def _descriptor_numba_fast(mag, ori, x, y, angle):
    """Ultra-fast descriptor computation"""
    desc = np.zeros(128, dtype=np.float32)

    c = cos(-angle)
    s = sin(-angle)

    y_int = int(round(y))
    x_int = int(round(x))
    h, w = mag.shape

    inv_32 = 1.0 / 32.0
    inv_2pi = 4.0 / pi  # 8 / (2*pi)

    for dy in range(-8, 8):
        dy_y = y_int + dy
        if dy_y < 0 or dy_y >= h:
            continue

        for dx in range(-8, 8):
            dx_x = x_int + dx
            if dx_x < 0 or dx_x >= w:
                continue

            # Rotate
            rx = c * dx - s * dy
            ry = s * dx + c * dy

            # Bin coordinates
            bx_f = (rx / 4.0) + 2.0
            by_f = (ry / 4.0) + 2.0

            if bx_f <= -1.0 or bx_f >= 4.0 or by_f <= -1.0 or by_f >= 4.0:
                continue

            # Gaussian weight
            weight = mag[dy_y, dx_x] * exp(-(rx * rx + ry * ry) * inv_32)

            if weight < 1e-8:
                continue

            # Orientation
            rel_ori = (ori[dy_y, dx_x] - angle) % (2 * pi)
            bo_f = rel_ori * inv_2pi

            bx0 = int(bx_f)
            by0 = int(by_f)
            bo0 = int(bo_f) % 8

            dx_frac = bx_f - bx0
            dy_frac = by_f - by0
            do_frac = bo_f - bo0

            # Trilinear interpolation
            for di in range(2):
                by_idx = by0 + di
                if 0 <= by_idx < 4:
                    wy = (1 - dy_frac) if di == 0 else dy_frac
                    for dj in range(2):
                        bx_idx = bx0 + dj
                        if 0 <= bx_idx < 4:
                            wx = (1 - dx_frac) if dj == 0 else dx_frac
                            for dk in range(2):
                                bo_idx = (bo0 + dk) % 8
                                wo = (1 - do_frac) if dk == 0 else do_frac
                                idx = by_idx * 32 + bx_idx * 8 + bo_idx
                                desc[idx] += weight * wx * wy * wo

    # Normalize
    norm = 0.0
    for i in range(128):
        norm += desc[i] * desc[i]
    norm = np.sqrt(norm)

    if norm > 1e-6:
        for i in range(128):
            desc[i] = min(desc[i] / norm, 0.2)

        norm = 0.0
        for i in range(128):
            norm += desc[i] * desc[i]
        norm = np.sqrt(norm)

        if norm > 1e-6:
            for i in range(128):
                desc[i] /= norm

    return desc


def _extract_descriptors_batch(kps, gauss, grad_pyr, dog):
    """Batch descriptor extraction"""
    keypoints = []
    descriptors = []

    for kp in kps:
        o, l, y, x, angle = kp
        l_int = int(round(l))

        if l_int < 0 or l_int >= len(gauss[o]):
            continue

        img_octave = gauss[o][l_int]

        if (
            x < 8
            or y < 8
            or x >= img_octave.shape[1] - 8
            or y >= img_octave.shape[0] - 8
        ):
            continue

        mag, ori = grad_pyr[o][l_int]
        desc = _descriptor_numba_fast(mag, ori, x, y, angle)

        scale_factor = 2.0 ** (o - 1.0)
        size = 2.0 * _params["sigma"] * (2**o) * (2 ** (l / (_params["layers"] - 3)))

        yi, xi = int(round(y)), int(round(x))
        if (
            l_int < len(dog[o])
            and 0 <= yi < dog[o][l_int].shape[0]
            and 0 <= xi < dog[o][l_int].shape[1]
        ):
            response = float(abs(dog[o][l_int][yi, xi]))
        else:
            response = 0.0

        keypoints.append(
            cv2.KeyPoint(
                x=float(x * scale_factor),
                y=float(y * scale_factor),
                size=float(size),
                angle=float((angle * 180 / pi) % 360),
                response=response,
                octave=o,
            )
        )
        descriptors.append(desc)

    return keypoints, descriptors
