import cv2
import numpy as np

def find_line_through_point(x0, y0, binary_mask, y_loc=0, min_line_length = 200, max_line_gap = 10):
    # Apply HoughLinesP to find lines in the binary mask
    lines = cv2.HoughLinesP(binary_mask, 1, np.pi/180, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)

    if lines is not None:
        min_distance = float('inf')
        best_line = None

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate the distance between the line segment and the given point (x0, y0)
            px = x2 - x1
            py = y2 - y1
            norm = px * px + py * py
            u = ((x0 - x1) * px + (y0 - y1) * py) / float(norm)

            if u > 1:
                u = 1
            elif u < 0:
                u = 0

            x = x1 + u * px
            y = y1 + u * py

            dx = x - x0
            dy = y - y0
            distance = np.sqrt(dx * dx + dy * dy)

            if distance < min_distance:
                min_distance = distance
                best_line = (x1, y1, x2, y2)

        if best_line is not None:
            x1, y1, x2, y2 = best_line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Calculate the cross point with y = y1
            x_cross = (y_loc - intercept) / slope
            cross_point = (x_cross, y_loc)

            return best_line, cross_point

    return None, None

def line_params(x1, y1, x2, y2):
    # Calculate the slope and intercept of the line
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def intersection_point(line1, line2):
    if isinstance(line1, np.ndarray):
        line1 = line1.tolist()
    if isinstance(line2, np.ndarray):
        line2 = line2.tolist()
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    m1, b1 = line_params(x1, y1, x2, y2)
    m2, b2 = line_params(x3, y3, x4, y4)

    # Check if lines are parallel
    if m1 == m2:
        return None

    # Calculate the intersection point
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y


def homography_from_vanishing_point(x1=None, y1=None, x2=None, y2=None, x3=None, y3=None, x4=None, y4=None, vp=None):
    if vp is None:
        vp = intersection_point((x1, y1, x2, y2), (x3, y3, x4, y4))
    if vp is None:
        return None

    px, py = vp
    H = np.array([
        [1, 0, -px],
        [0, 1, -py],
        [0, 0, 1]
    ])

    return H

def apply_homography(H, img):
    h, w = img.shape[:2]
    transformed_img = cv2.warpPerspective(img, H, (w, h))
    return transformed_img

def find_homography(pts_src, pts_dst):
    H, _ = cv2.findHomography(pts_src, pts_dst)
    return H

def project2d(points, img):
    print(len(points))
    if len(points) == 4:
        x1, y1 = points[0][0],points[0][1]
        x2, y2 = points[1][0],points[1][1]
        x3, y3 = points[2][0],points[2][1]
        x4, y4 = points[3][0],points[3][1]
        H = homography_from_vanishing_point(x1, y1, x2, y2, x3, y3, x4, y4)
    elif len(points) == 1:
        H = homography_from_vanishing_point(vp=points[0])
    else:
        x1, y1, x2, y2, x3, y3, x4, y4 = points
        #H = homography_from_vanishing_point(x1, y1, x2, y2, x3, y3, x4, y4)
        src_pts = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], dtype=np.float32)
        dst_pts = np.array([(x1, y1), (x1, y2), (x3, y3), (x3, y4)], dtype=np.float32)
        H = find_homography(src_pts, dst_pts)
        
    return apply_homography(H, img) 