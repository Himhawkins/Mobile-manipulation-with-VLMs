#!/usr/bin/env python3
# modules/geometry_utils.py

import math
import numpy as np
from typing import List, Tuple

Point = Tuple[int, int]
Polyline = List[Point]
Mask = np.ndarray

def _densify_segment(p0: Point, p1: Point, step_px: int = 12) -> Polyline:
    """
    Generate intermediate points along a line segment.

    Args:
        p0: The starting point (x, y).
        p1: The ending point (x, y).
        step_px: The approximate distance between points.

    Returns:
        A list of points, including the start and end points.
    """
    x0, y0 = p0
    x1, y1 = p1
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1e-6:
        return [p0]
    
    num_steps = max(1, int(dist // step_px))
    points = []
    for i in range(num_steps + 1):
        t = i / num_steps
        px = int(round(x0 + t * (x1 - x0)))
        py = int(round(y0 + t * (y1 - y0)))
        # Avoid adding consecutive duplicate points from rounding
        if not points or (px, py) != points[-1]:
            points.append((px, py))
    return points

def _densify_polyline(path: Polyline, step_px: int = 12) -> Polyline:
    """
    Generate intermediate points along a multi-segment path.

    Args:
        path: A list of points defining the polyline.
        step_px: The approximate distance between points.

    Returns:
        A densified list of points.
    """
    if not path or len(path) < 2:
        return path
    
    densified_path = []
    for i in range(len(path) - 1):
        segment = _densify_segment(path[i], path[i+1], step_px=step_px)
        # Avoid duplicating the connection point between segments
        if densified_path:
            densified_path.extend(segment[1:])
        else:
            densified_path.extend(segment)
    return densified_path

def _bresenham_blocked(mask: Mask, p0: Point, p1: Point) -> bool:
    """
    Check if a line between two points is blocked by an obstacle in a mask.
    Uses the Bresenham's line algorithm for efficient grid traversal.

    Args:
        mask: A NumPy array where non-zero values represent obstacles.
        p0: The starting point (x, y).
        p1: The ending point (x, y).

    Returns:
        True if the line is blocked, False otherwise.
    """
    if mask is None:
        return False

    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    H, W = mask.shape[:2]

    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if 0 <= x0 < W and 0 <= y0 < H and mask[y0, x0] != 0:
            return True
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return False

def _line_blocked_multi(masks: List[Mask], p0: Point, p1: Point) -> bool:
    """Check if a line is blocked in any of a list of masks."""
    for m in masks:
        if m is not None and _bresenham_blocked(m, p0, p1):
            return True
    return False

def _min_clearance_along_line(dt: np.ndarray, p0: Point, p1: Point, stride: int = 3) -> float:
    """
    Find the minimum distance to an obstacle along a line segment.

    Args:
        dt: The distance transform of the obstacle map.
        p0: The starting point (x, y).
        p1: The ending point (x, y).
        stride: How many pixels to step when sampling along the line.

    Returns:
        The minimum clearance in pixels.
    """
    if dt is None:
        return 0.0
    
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    H, W = dt.shape[:2]

    num_points = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, num_points + 1, dtype=int)[::max(1, stride)]
    ys = np.linspace(y0, y1, num_points + 1, dtype=int)[::max(1, stride)]
    
    # Clip coordinates to be within the bounds of the distance transform
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)
    
    return float(dt[ys, xs].min())

def _line_ok(masks: List[Mask], dt: np.ndarray, p0: Point, p1: Point, clearance_req_px: float) -> bool:
    """
    Check if a line has sufficient clearance and is not blocked.

    Args:
        masks: A list of binary obstacle masks.
        dt: The distance transform of the obstacle map.
        p0: The starting point (x, y).
        p1: The ending point (x, y).
        clearance_req_px: The minimum required clearance in pixels.

    Returns:
        True if the line is clear, False otherwise.
    """
    if _line_blocked_multi(masks, p0, p1):
        return False
    if clearance_req_px <= 0:
        return True
    return _min_clearance_along_line(dt, p0, p1) >= clearance_req_px

def _polyline_ok(dt: np.ndarray, polyline: Polyline, clearance_req_px: float, stride: int = 3) -> bool:
    """
    Check if an entire polyline maintains a minimum clearance from obstacles.

    Args:
        dt: The distance transform of the obstacle map.
        polyline: A list of points defining the path.
        clearance_req_px: The minimum required clearance in pixels.
        stride: The sampling stride for checking each segment.

    Returns:
        True if the entire path is clear, False otherwise.
    """
    if clearance_req_px <= 0:
        return True
    
    for i in range(len(polyline) - 1):
        if _min_clearance_along_line(dt, polyline[i], polyline[i+1], stride=stride) < clearance_req_px:
            return False
    return True