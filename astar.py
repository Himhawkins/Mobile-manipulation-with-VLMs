#!/usr/bin/env python3
"""
Path planning utilities with obstacle-aware line checking and A* pathfinding.
"""
import numpy as np
import cv2
import heapq
import math

class PathPlanner:
    """
    Provides obstacle mask construction, Bresenham line intersection tests,
    and 8-connected A* pathfinding around polygonal obstacles.
    """
    def __init__(self, obstacles, image_shape, arena_corners=None):
        """
        :param obstacles: list of dicts with key "corners": list of 4 (x,y) points
        :param image_shape: (height, width)
        :param arena_corners: optional 4×2 int array of calibrated arena polygon
        """
        h, w = image_shape
        self.mask = np.zeros((h, w), dtype=np.uint8)

        # Fill obstacle polygons into mask
        for obs in obstacles:
            if "corners" not in obs:
                raise ValueError("Each obstacle must contain a 'corners' key")
            corners = np.array(obs["corners"], dtype=np.int32)
            cv2.fillPoly(self.mask, [corners], 255)

        # Optionally mask out everything outside the arena
        if arena_corners is not None:
            arena_mask = np.zeros_like(self.mask)
            poly = np.array(arena_corners, dtype=np.int32)
            cv2.fillPoly(arena_mask, [poly], 255)
            outside = cv2.bitwise_not(arena_mask)
            self.mask = cv2.bitwise_or(self.mask, outside)

    def line_intersects_obstacle(self, p1, p2):
        """
        Check if the straight line segment from p1 to p2 intersects any obstacle.

        :param p1: (x1, y1) start point
        :param p2: (x2, y2) end point
        :returns: True if the line passes through an obstacle pixel
        """
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if self.mask[y, x]:
                    return True
                err -= dy
                if err < 0:
                    y += sy; err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                if self.mask[y, x]:
                    return True
                err -= dx
                if err < 0:
                    x += sx; err += dy
                y += sy
        return bool(self.mask[y2, x2])

    def astar_path(self, start, goal):
        """
        Perform an 8-connected A* search from start to goal on the obstacle mask.

        :param start: (x, y) start point
        :param goal: (x, y) goal point
        :returns: list of waypoints from start to goal, or None if no path
        """
        h, w = self.mask.shape

        def hscore(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        open_set = []
        heapq.heappush(open_set, (hscore(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            f, g, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = came_from.get(current)
                return path[::-1]

            if g > g_score.get(current, float('inf')):
                continue

            for dx, dy in neighbors:
                nx, ny = current[0] + dx, current[1] + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if self.mask[ny, nx]:
                    continue
                tentative_g = g + math.hypot(dx, dy)
                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(
                        open_set,
                        (tentative_g + hscore(neighbor, goal), tentative_g, neighbor)
                    )
        return None

    def simplify_path(self, path, min_dist=10):
        """
        Thin out a dense pixel path by only keeping points at least
        min_dist apart (in Euclidean pixels).
        """
        if not path:
            return path
        simplified = [path[0]]
        last = path[0]
        for pt in path[1:]:
            if math.hypot(pt[0] - last[0], pt[1] - last[1]) >= min_dist:
                simplified.append(pt)
                last = pt
        if simplified[-1] != path[-1]:
            simplified.append(path[-1])
        return simplified

    def find_obstacle_aware_path(self, start, goal, simplify_dist=None):
        """
        If the straight line is clear, returns [start, goal].
        Otherwise does A* and then (optionally) simplifies.
        """
        if not self.line_intersects_obstacle(start, goal):
            path = [start, goal]
        else:
            path = self.astar_path(start, goal)

        if simplify_dist is not None and path:
            path = self.simplify_path(path, simplify_dist)
        return path
