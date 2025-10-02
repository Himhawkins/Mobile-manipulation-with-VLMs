#!/usr/bin/env python3
"""
Path planning utilities with a stable D* Lite implementation for static environments.
"""
import numpy as np
import cv2
import heapq
import math

class PathPlanner:
    """
    Provides obstacle mask construction, Bresenham line intersection tests,
    and 8-connected D* Lite pathfinding around polygonal obstacles.
    """
    def __init__(self, obstacles, image_shape, arena_corners=None):
        h, w = image_shape
        self.h, self.w = h, w
        self.mask = np.zeros((h, w), dtype=np.uint8)
        for obs in obstacles:
            if "corners" not in obs:
                raise ValueError("Each obstacle must contain a 'corners' key")
            corners = np.array(obs["corners"], dtype=np.int32)
            cv2.fillPoly(self.mask, [corners], 255)
        if arena_corners is not None:
            arena_mask = np.zeros_like(self.mask)
            poly = np.array(arena_corners, dtype=np.int32)
            cv2.fillPoly(arena_mask, [poly], 255)
            outside = cv2.bitwise_not(arena_mask)
            self.mask = cv2.bitwise_or(self.mask, outside)
        
        self.g = {}
        self.rhs = {}
        self.U = []

    def line_intersects_obstacle(self, p1, p2):
        x1, y1 = p1; x2, y2 = p2
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1; sy = 1 if y2 > y1 else -1
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if self.mask[y, x]: return True
                err -= dy
                if err < 0: y += sy; err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                if self.mask[y, x]: return True
                err -= dx
                if err < 0: x += sx; err += dy
                y += sy
        return bool(self.mask[y2, x2])

    def _heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _get_cost(self, p1, p2):
        if self.mask[p2[1], p2[0]]:
            return float('inf')
        return self._heuristic(p1, p2)

    def _get_neighbors(self, u):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nx, ny = u[0] + dx, u[1] + dy
                if 0 <= nx < self.w and 0 <= ny < self.h:
                    neighbors.append((nx, ny))
        return neighbors

    def _calculate_key(self, s, start_node):
        h = self._heuristic(s, start_node)
        k1 = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf'))) + h
        k2 = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return (k1, k2)

    def _update_vertex(self, u, start_node, goal_node):
        if u != goal_node:
            min_rhs = float('inf')
            for neighbor in self._get_neighbors(u):
                min_rhs = min(min_rhs, self._get_cost(u, neighbor) + self.g.get(neighbor, float('inf')))
            self.rhs[u] = min_rhs
        
        # This implementation pushes duplicates to the heap, which is fine.
        # Stale entries (with old, worse keys) are ignored when popped.
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
             heapq.heappush(self.U, (self._calculate_key(u, start_node), u))

    def _compute_shortest_path(self, start_node, goal_node):
        """
        Main search loop. This is a simplified version for static environments
        that is guaranteed to terminate.
        """
        while self.U:
            # Termination condition
            if not self.U or (self.U[0][0] >= self._calculate_key(start_node, start_node) and \
               self.rhs.get(start_node, float('inf')) == self.g.get(start_node, float('inf'))):
                break

            k_old, u = heapq.heappop(self.U)
            k_new = self._calculate_key(u, start_node)

            # Ignore stale entries that have a worse key now
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
                continue

            # This is the primary logic: processing "under-consistent" nodes.
            # This is the only case needed for a static pathfind.
            if self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs.get(u, float('inf'))
                for s in self._get_neighbors(u):
                    self._update_vertex(s, start_node, goal_node)
            
            # ** THE FIX: **
            # The 'else' block that handled 'over-consistent' nodes is REMOVED.
            # That logic is for dynamic cost increases and was the source of the infinite loop.
            # By removing it, the algorithm becomes a stable static planner.

    def dstar_lite_path(self, start, goal):
        self.U = []
        self.g = {}
        self.rhs = {}
        self.rhs[goal] = 0
        heapq.heappush(self.U, (self._calculate_key(goal, start), goal))
        
        self._compute_shortest_path(start, goal)

        if self.g.get(start, float('inf')) == float('inf'):
            return None

        # Path Reconstruction
        path = [start]
        current = start
        path_len_limit = self.w * self.h
        for _ in range(path_len_limit):
            if current == goal:
                break
            min_cost = float('inf')
            next_node = None
            for neighbor in self._get_neighbors(current):
                cost = self._get_cost(current, neighbor) + self.g.get(neighbor, float('inf'))
                if cost < min_cost:
                    min_cost = cost
                    next_node = neighbor
            if next_node is None: return None
            path.append(next_node)
            current = next_node
        else:
            print("D* Lite Error: Path reconstruction failed to find goal.")
            return None
        return path

    def simplify_path(self, path, min_dist=10):
        if not path: return path
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
        if self.mask[start[1], start[0]] or self.mask[goal[1], goal[0]]:
            print("Start or goal is inside an obstacle.")
            return None
        if not self.line_intersects_obstacle(start, goal):
            return [start, goal]
        
        path = self.dstar_lite_path(start, goal)
        
        if simplify_dist is not None and path:
            path = self.simplify_path(path, simplify_dist)
        return path