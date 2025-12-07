import contour_detection
import edge_detectioin
import histogram
import homography
import matcher
import morphological_operations
import sift
import smoother


import cv2
import numpy as np
from pathlib import Path



# Threshold configuration
MIN_MATCH_COUNT = 4  # Minimum matches needed for homography
BRAND_CONFIDENCE_THRESHOLD = 18  # Minimum good matches to classify as a known brand

# Contour extraction parameters
MIN_CONTOUR_AREA = 500  # Minimum area for a contour to be considered
MAX_CONTOUR_AREA = 50000  # Maximum area for a contour to be considered
MIN_ASPECT_RATIO = 0.3  # Minimum width/height ratio
MAX_ASPECT_RATIO = 3.0  # Maximum width/height ratio



def verify_logo(image, descriptores):
    """Decide which method to use for logo verification"""
    res1 = method_one(image, descriptores)
    res2 = method_two(image, descriptores)

    if res1 == res2:
        return res1
    else:
        if res1 == "UNKNOWN BRAND":
            return res2
        else:
            return res1

def method_one(image, descriptores):
    cloth = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if cloth is None:
        print(f"Error loading image")
        exit()
        
    # Compute SIFT features directly on the shirt image
    cloth = smoother.bilateral_filter(cloth, 0, 1, 1)
    kp_cloth, des_cloth = sift.calc_sift(cloth)
    
    if des_cloth is None:
        print("No descriptors found in shirt")
        return "UNKNOWN BRAND"
        
    print(f"Found {len(kp_cloth)} SIFT keypoints")
        
    best_brand = None
    best_match_count = 0
    
        
    # Match against all logos from all brands
    for brand, logo_data in descriptores.items():
        for idx, (kp_logo, des_logo) in enumerate(logo_data):
            if des_logo is None:
                continue
                
            # Match with KNN
            knn_matches = matcher.knn_matches(des_logo, des_cloth)
        
            # Lowe's ratio test
            good = []
            ratio_thresh = 0.6
            
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_thresh * n.distance:
                        good.append(m)
            
            # Apply RANSAC to filter outliers if we have enough matches
            inlier_count = len(good)
            if len(good) >= MIN_MATCH_COUNT:
                src_pts = np.float32([kp_logo[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_cloth[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = homography.ran_sac(src_pts, dst_pts)
                
                # Count only inliers (matches that fit the homography model)
                if mask is not None:
                    inlier_count = mask.sum()
            
            # Check if this is the best match so far (using inlier count)
            if inlier_count > best_match_count:
                best_match_count = inlier_count
                best_brand = brand
    
        
    if best_match_count < BRAND_CONFIDENCE_THRESHOLD:
        print(f"RESULT: UNKNOWN BRAND (only {best_match_count} matches found, threshold is {BRAND_CONFIDENCE_THRESHOLD})")
        is_known_brand = False
    else:
        print(f"RESULT: {best_brand} with {best_match_count} good matches ✓")
        is_known_brand = True
        
    return best_brand if is_known_brand else "UNKNOWN BRAND" 

def method_two(image, descriptores):
    def extract_logo_regions(image):
        # Apply Gaussian blur to reduce noise
        blurred = smoother.gaussian_blur(image, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = edge_detectioin.canny_edge_detector(blurred, 50, 150)
        
        # Morphological operations to close gaps in edges
        dilated = morphological_operations.dilate(edges, iterations= 2, kernel_size=(5,5))
        
        # Find contours
        contours, _ = contour_detection.get_contours(dilated)
        
        logo_regions = []
        
        for contour in contours:
            # Calculate contour properties
            area = contour_detection.get_area(contour)
            
            # Filter by area
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue
            
            # Get bounding rectangle
            x, y, w, h = contour_detection.get_bounding_rects(contour)
            
            # Filter by aspect ratio (reject very elongated shapes)
            aspect_ratio = w / float(h)
            if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                continue
            
            # Add padding around the region
            padding = 10
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(image.shape[1] - x_padded, w + 2 * padding)
            h_padded = min(image.shape[0] - y_padded, h + 2 * padding)
            
            # Extract the region of interest
            roi = image[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]
            
            if roi.size > 0:
                logo_regions.append((x_padded, y_padded, w_padded, h_padded, roi))
        
        return logo_regions
    
    cloth = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if cloth is None:
        print(f"Error loading image")
        return "UNKNOWN BRAND"
        
    # Extract suspected logo regions using edge detection and contours
    print("Extracting logo regions...")
    logo_regions = extract_logo_regions(cloth)
    print(f"Found {len(logo_regions)} suspected logo regions")
    
        
    if len(logo_regions) == 0:
        print("No logo regions detected - trying full image instead")
        logo_regions = [(0, 0, cloth.shape[1], cloth.shape[0], cloth)]
    
    best_brand = None
    best_match_count = 0
    best_region_coords = None
    
    # Process each suspected logo region
    for region_idx, (x, y, w, h, roi) in enumerate(logo_regions):
        print(f"\n  Analyzing region {region_idx + 1}/{len(logo_regions)} at ({x}, {y}, {w}, {h})")
        
        # Compute SIFT features on this region
        kp_roi, des_roi = sift.detectAndCompute(roi, None)
        
        if des_roi is None:
            print(f"    No descriptors found in region {region_idx + 1}")
            continue
        
        print(f"    Found {len(kp_roi)} SIFT keypoints in this region")
        
        # Match against all logos from all brands
        for brand, logo_data in descriptores.items():
            for idx, (kp_logo, des_logo) in enumerate(logo_data):
                if des_logo is None:
                    continue
                
                # Match with KNN
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                knn_matches = bf.knnMatch(des_logo, des_roi, k=2)
                
                # Lowe's ratio test
                good = []
                ratio_thresh = 0.6
                
                for match_pair in knn_matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < ratio_thresh * n.distance:
                            good.append(m)
                
                # Apply RANSAC to filter outliers if we have enough matches
                inlier_count = len(good)
                H = None
                mask = None
                
                if len(good) >= MIN_MATCH_COUNT:
                    src_pts = np.float32([kp_logo[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_roi[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    
                    # Adjust destination points to global coordinates
                    dst_pts_global = dst_pts.copy()
                    dst_pts_global[:, 0, 0] += x
                    dst_pts_global[:, 0, 1] += y
                    
                    H, mask = cv2.findHomography(src_pts, dst_pts_global, cv2.RANSAC, 5.0)
                    
                    # Count only inliers (matches that fit the homography model)
                    if mask is not None:
                        inlier_count = int(mask.sum())
                
                # Check if this is the best match so far (using inlier count)
                if inlier_count > best_match_count:
                    best_match_count = inlier_count
                    best_brand = brand
                    best_region_coords = (x, y, w, h)
                    
                    
                    # Store keypoints in global coordinates for visualization
                    kp_roi_global = []
                    for kp in kp_roi:
                        kp_copy = cv2.KeyPoint(kp.pt[0] + x, kp.pt[1] + y, kp.size, 
                                            kp.angle, kp.response, kp.octave, kp.class_id)
                        kp_roi_global.append(kp_copy)
                    
    
    # Determine if it's a known brand or unknown based on threshold
    if best_match_count < BRAND_CONFIDENCE_THRESHOLD:
        print(f"\nRESULT: UNKNOWN BRAND (only {best_match_count} matches found, threshold is {BRAND_CONFIDENCE_THRESHOLD})")
        is_known_brand = False
    else:
        print(f"\nRESULT: {best_brand} with {best_match_count} good matches ✓")
        if best_region_coords:
            print(f"Logo detected in region: {best_region_coords}")
        is_known_brand = True
    
    return best_brand if is_known_brand else "UNKNOWN BRAND"



    




   




    





    

    



        

    