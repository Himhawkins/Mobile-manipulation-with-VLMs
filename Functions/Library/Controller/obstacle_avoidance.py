# obstacle_avoidance.py

import os
import math
import cv2
import numpy as np
from astar import PathPlanner
import read_data

class ObstacleManager:
    """Manages obstacle data, masks, and the A* planner."""

    def __init__(self, data_folder, frame_size, spacing_px, robot_padding, robot_id=None):
        self.data_folder = data_folder
        self.frame_size = frame_size
        self.spacing_px = int(spacing_px)
        self.robot_padding = int(robot_padding)
        self.robot_id = robot_id

        self.static_path = os.path.join(data_folder, "obstacles.txt")
        self.real_path = os.path.join(data_folder, "realtime_obstacles.txt")
        self.arena_path = os.path.join(data_folder, "arena_corners.txt")
        self.robot_path = os.path.join(data_folder, "robot_pos.txt")

        self._mtimes = {p: 0 for p in [self.static_path, self.real_path, self.arena_path, self.robot_path]}
        self.planner = None
        self.mask_raw = None
        self.mask_dilated = None
        self.clearance_dt = None
        
        self.rebuild_if_needed()

    def _files_changed(self):
        """Check if any obstacle-related files have been modified."""
        for path in self._mtimes:
            try:
                mtime = os.path.getmtime(path)
                if mtime != self._mtimes[path]:
                    self._mtimes[path] = mtime
                    return True
            except FileNotFoundError:
                continue
        return False

    def rebuild_if_needed(self):
        """Rebuilds obstacle masks and planner if files have changed."""
        if self.planner and not self._files_changed():
            return

        print("[INFO] Obstacle data changed, rebuilding planner...")
        static_polys = read_data.read_obstacle_polygons(self.static_path)
        realtime_polys = read_data.read_obstacle_polygons(self.real_path)
        arena_corners = read_data.read_arena_corners(self.arena_path)
        
        obstacles = [{"corners": p} for p in (static_polys + realtime_polys)]
        self.planner = PathPlanner(obstacles, self.frame_size, arena_corners=arena_corners)

        self.mask_raw = self.planner.mask.copy()

        # Add other robots as circular obstacles
        other_robots = read_data.read_all_robot_positions(self.robot_path)
        for rid, x, y in other_robots:
            if self.robot_id is not None and rid == self.robot_id:
                continue
            cv2.circle(self.mask_raw, (int(x), int(y)), self.robot_padding, 255, -1)
        
        # Create inflated mask for collision checking
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.spacing_px + 1, 2 * self.spacing_px + 1))
        self.mask_dilated = cv2.dilate(self.mask_raw, kernel)
        
        # Create distance transform for proximity slowdown
        free_space = np.where(self.mask_raw == 0, 255, 0).astype(np.uint8)
        self.clearance_dt = cv2.distanceTransform(free_space, cv2.DIST_L2, 3)

        # Update planner to use the inflated mask for pathfinding
        self.planner.mask = self.mask_dilated.copy()

def _densify_path(path, step_px=12):
    """Adds intermediate points to a path to make it smoother."""
    if not path or len(path) < 2:
        return path
    
    densified = []
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(1, int(dist / step_px))
        for j in range(steps):
            t = j / steps
            px = p1[0] * (1 - t) + p2[0] * t
            py = p1[1] * (1 - t) + p2[1] * t
            if not densified or (int(px), int(py)) != densified[-1]:
                densified.append((int(px), int(py)))
    densified.append(path[-1])
    return densified

def _is_line_blocked(mask, p1, p2):
    """Checks if a straight line between two points is blocked by an obstacle."""
    x0, y0 = map(int, p1); x1, y1 = map(int, p2)
    dx, sx = abs(x1 - x0), 1 if x0 < x1 else -1
    dy, sy = -abs(y1 - y0), 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if mask[y0, x0] != 0: return True
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 >= dy: err += dy; x0 += sx
        if e2 <= dx: err += dx; y0 += sy
    return False

def get_safe_path(obstacle_manager, start_xy, goal_xy):
    """
    Plans a safe path from start to goal, avoiding obstacles.
    Returns a list of waypoints [(x, y), ...] or None if no path is found.
    """
    om = obstacle_manager
    om.rebuild_if_needed()

    if om.planner is None or om.mask_dilated is None:
        print("[WARN] Planner not initialized.")
        return None

    sx, sy = int(start_xy[0]), int(start_xy[1])
    gx, gy = int(goal_xy[0]), int(goal_xy[1])

    # If start is inside an obstacle, we can't plan
    if om.mask_dilated[sy, sx] != 0:
        print("[WARN] Start point is inside an obstacle's safety zone. Cannot plan.")
        return None # Simplified: In a real scenario, you'd implement an "escape" routine here.

    # 1. Try a direct path first
    if not _is_line_blocked(om.mask_dilated, (sx, sy), (gx, gy)):
        return _densify_path([(sx, sy), (gx, gy)])

    # 2. If direct path is blocked, use A*
    path = om.planner.find_obstacle_aware_path((sx, sy), (gx, gy), simplify_dist=10)
    
    if path:
        return _densify_path(path)
    
    print(f"[WARN] A* failed to find a path from ({sx},{sy}) to ({gx},{gy}).")
    return None