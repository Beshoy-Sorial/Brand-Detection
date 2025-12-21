import cv2
import numpy as np
def get_contours(image):

    """
    Extract contours from a binary image using the Moore-Neighbor tracing algorithm."""
    h, w = image.shape
    
    # Pad image to simplify boundary checking
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = image
    
    # Track which pixels have been assigned to a contour
    visited = np.zeros_like(padded, dtype=bool)
    contours = []
    
    # 8-connected neighbors in clockwise order starting from top
    directions = [
        (-1, 0),   # 0: North
        (-1, 1),   # 1: North-East
        (0, 1),    # 2: East
        (1, 1),    # 3: South-East
        (1, 0),    # 4: South
        (1, -1),   # 5: South-West
        (0, -1),   # 6: West
        (-1, -1)   # 7: North-West
    ]
    
    def trace_contour(start_y, start_x, is_external):
        """
        Trace a contour using the Moore-Neighbor algorithm.
        """
        contour = []
        
        # For external contours, we entered from the left (background was west)
        # For holes, we entered from the right (background was east)
        if is_external:
            prev_dir = 6  # Came from West, so search starts from South (6+2)%8
        else:
            prev_dir = 2  # Came from East, so search starts from North (2+2)%8
        
        curr_y, curr_x = start_y, start_x
        start_pos = (start_y, start_x)
        
        # Trace the contour
        first_iteration = True
        
        while True:
            # Add current position to contour (convert to original coordinates)
            contour.append((curr_x - 1, curr_y - 1))
            
            # Mark as visited
            visited[curr_y, curr_x] = True
            
            # Find next boundary pixel using Moore-Neighbor search
            # Start searching from 2 positions clockwise from where we came from
            search_start = (prev_dir + 2) % 8 if not first_iteration else prev_dir
            first_iteration = False
            
            found = False
            for i in range(8):
                check_dir = (search_start + i) % 8
                dy, dx = directions[check_dir]
                next_y, next_x = curr_y + dy, curr_x + dx
                
                # Check bounds and if next pixel is foreground
                if (0 <= next_y < padded.shape[0] and 
                    0 <= next_x < padded.shape[1] and 
                    padded[next_y, next_x] > 0):
                    
                    # Found next boundary pixel
                    # The direction we came FROM is opposite to where we're going
                    prev_dir = (check_dir + 4) % 8
                    curr_y, curr_x = next_y, next_x
                    found = True
                    break
            
            if not found:
                # Isolated pixel or end of contour
                break
            
            # Check if we've completed the loop
            if (curr_y, curr_x) == start_pos and len(contour) > 1:
                break
            
            # Safety check
            if len(contour) > h * w * 2:
                break
        
        return contour
    
    # Scan for contours using raster scan
    for y in range(1, h + 1):
        for x in range(1, w + 1):
            if padded[y, x] > 0 and not visited[y, x]:
                # Check if this is a contour starting point
                
                # External contour: foreground pixel with background to the left
                if padded[y, x - 1] == 0:
                    contour = trace_contour(y, x, is_external=True)
                    if len(contour) > 0:
                        contours.append(contour)
                
                # Internal contour (hole): foreground pixel with background to the right
                elif padded[y, x + 1] == 0 and not visited[y, x]:
                    contour = trace_contour(y, x, is_external=False)
                    if len(contour) > 0:
                        contours.append(contour)
    
    return contours, None    # return cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
def get_area(contour):
    """implement your solution and document this function"""
    area = 0.0
    n = len(contour)

    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]
        area += x1 * y2 - x2 * y1 

    return abs(area) / 2.0
    # return cv2.contourArea(contour)

def get_bounding_rects(contour):
    """implement your solution and document this function"""
    xs = [p[0] for p in contour]
    ys = [p[1] for p in contour]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    # return cv2.boundingRect(contour)