import cv2
import math
import numpy as np
import cv2.aruco as aruco # This import is not used in the provided functions but kept for context.
from itertools import combinations
from typing import Tuple, List
from Functions.Library.planning import trace_targets


def detect_center_line_and_save(image_path="Data/frame_img.png", num_points=40.0):

    image = cv2.imread(image_path)
    print(image_path)

    orange_hsv_lower: Tuple[int, int, int] = (10, 115, 110)
    orange_hsv_upper: Tuple[int, int, int] = (25, 255, 255)
    morph_kernel_size: Tuple[int, int] = (5, 5)
    # save_path: str = "line_coordinates.txt"

    # 1) HSV-threshold for orange color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    orange_mask = cv2.inRange(
        hsv,
        np.array(orange_hsv_lower),
        np.array(orange_hsv_upper)
    )

    # 2) Apply morphological closing to fill small gaps in the orange line
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    closed = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

    # 3) Skeletonize the closed mask to get a single-pixel wide line
    # cv2.ximgproc.thinning is part of opencv-contrib-python
    skel = cv2.ximgproc.thinning(closed)
    # cv2.imshow("Skeletonized Line", skel) # Display the skeletonized line

    # 4) Extract all skeleton pixels and prepare for path tracing
    h, w = skel.shape
    skel_pixels = set() # Using a set for efficient lookup
    ys, xs = np.nonzero(skel) # Get coordinates of all non-zero pixels (skeleton)
    for y, x in zip(ys, xs):
        skel_pixels.add((x, y)) # Store as (x, y) for consistency with OpenCV drawing

    if not skel_pixels:
        raise RuntimeError("No centerline pixels found after orange masking and skeletonization.")

    # Define 8-directional neighbors for tracing
    neighbors_offsets = [(-1,-1), (0,-1), (1,-1),
                         (-1, 0),         (1, 0),
                         (-1, 1), (0, 1), (1, 1)]

    # Find a suitable starting point for tracing the line.
    # Prioritize endpoints (pixels with only one skeleton neighbor)
    endpoints = []
    for px, py in skel_pixels:
        neighbor_count = 0
        for dx, dy in neighbors_offsets:
            nx, ny = px + dx, py + dy
            # Check if neighbor is within bounds and is also a skeleton pixel
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) in skel_pixels:
                neighbor_count += 1
        # A pixel with 1 neighbor is an endpoint of a line segment
        if neighbor_count == 1:
            endpoints.append((px, py))

    start_point = None
    if endpoints:
        # If endpoints are found, pick the one that is top-most then left-most
        # This provides a consistent starting point for tracing.
        endpoints.sort(key=lambda p: (p[1], p[0])) # Sort by y-coordinate then x-coordinate
        start_point = endpoints[0]
    else:
        # Fallback: If no clear endpoints (e.g., a closed loop or very complex skeleton),
        # pick the top-leftmost pixel as a starting heuristic.
        # This might not be ideal for complex structures but works for simple lines.
        start_point = min(skel_pixels, key=lambda p: (p[1], p[0]))

    # 5) Trace the path sequentially from the start_point
    ordered_points_xy = []
    visited_pixels = set() # Keep track of visited pixels to avoid loops and re-visiting
    current_point = start_point

    while current_point and current_point not in visited_pixels:
        ordered_points_xy.append(current_point)
        visited_pixels.add(current_point)

        next_point = None
        unvisited_neighbors = []

        # Find unvisited skeleton neighbors of the current point
        for dx, dy in neighbors_offsets:
            nx, ny = current_point[0] + dx, current_point[1] + dy
            if (nx, ny) in skel_pixels and (nx, ny) not in visited_pixels:
                unvisited_neighbors.append((nx, ny))

        if len(unvisited_neighbors) == 1:
            # Ideal case for a line: exactly one unvisited neighbor
            next_point = unvisited_neighbors[0]
        elif len(unvisited_neighbors) > 1:
            # This can happen at junctions or if thinning isn't perfect.
            # For a simple line, we expect only one. If multiple, pick the first one.
            # For more complex skeletons, a more sophisticated logic (e.g., based on
            # previous direction) might be needed.
            next_point = unvisited_neighbors[0]
        else:
            # No unvisited neighbors, the path has ended
            break
        current_point = next_point

    if not ordered_points_xy:
        raise RuntimeError("Failed to trace any path from the starting point. Check mask and skeletonization.")

    # Calculate theta (orientation) for each point based on the next point in the sequence.
    # This provides the local direction of the line at each sampled point.
    sampled_points_with_theta = []
    total_traced_length = len(ordered_points_xy)
    num_points = int(num_points)
    # Determine indices for uniform sampling from the ordered points
    if total_traced_length < num_points:
        # If fewer points traced than requested, take all traced points
        sampled_indices = range(total_traced_length)
    else:
        # Uniformly sample 'num_points' from the traced path
        sampled_indices = [int(round(i * (total_traced_length - 1) / (num_points - 1))) for i in range(num_points)]

    for i in sampled_indices:
        x, y = ordered_points_xy[i]
        theta = 0.0 # Default theta

        if i < total_traced_length - 1:
            # Calculate theta using the vector from current point to the next point
            next_x, next_y = ordered_points_xy[i+1]
            theta = math.atan2(next_y - y, next_x - x)
        elif i > 0:
            # For the very last sampled point, use the theta from the previous segment
            # to maintain a consistent direction.
            prev_x, prev_y = ordered_points_xy[i-1]
            theta = math.atan2(y - prev_y, x - prev_x)
        # If it's the only point (total_traced_length == 1), theta remains 0.0
        sampled_points_with_theta.append((x, y))
    return trace_targets(input_target_list=sampled_points_with_theta, output_target_path="Targets/path.txt")

if __name__ == "__main__":
    detect_center_line_and_save(image_path="Data/frame_img.png", num_points=40.0)

