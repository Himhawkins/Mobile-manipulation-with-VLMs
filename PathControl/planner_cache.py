#!/usr/bin/env python3
# modules/planner_cache.py

import os
import cv2
import datetime
import numpy as np
from astar import PathPlanner
from typing import List, Tuple, Dict, Optional

Point = Tuple[int, int]
Polygon = List[Point]

def _dbg(msg: str):
    """A shared debug logging function."""
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[DBG {ts}] {msg}")

def _get_mtime(path: str) -> float:
    """Safely get the last modification time of a file."""
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0.0

def _parse_corners_file(path: str) -> List[Polygon]:
    """Parse a text file containing corner coordinates for polygons."""
    polygons = []
    try:
        with open(path, "r") as f:
            for line in f:
                s = line.strip().replace("(", "").replace(")", "")
                if not s or s.count(',') != 7:
                    continue
                try:
                    nums = list(map(int, s.split(",")))
                    # Group coordinates into (x, y) points
                    polygons.append([(nums[i], nums[i+1]) for i in range(0, 8, 2)])
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        pass # It's okay if the file doesn't exist
    except Exception as e:
        print(f"[WARN] Could not parse corners file '{path}': {e}")
    return polygons

def _read_arena_corners(path: str) -> Optional[List[Point]]:
    """Read arena boundary points from a text file."""
    points = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                points.append(tuple(map(int, line.split(","))))
        return points if points else None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[WARN] Could not read arena corners from '{path}': {e}")
    return None


class PlannerCache:
    """
    Manages obstacle maps, A* planner instances, and associated data.

    This class intelligently caches generated obstacle masks and distance transforms.
    It monitors source files (for obstacles, arena, etc.) and only rebuilds the
    planner and masks when a file has been modified, saving significant computation.
    """

    def __init__(self, data_folder: str, spacing_px: int = 26, plan_spacing_boost_px: int = 8,
                 line_margin_px: int = 1, robot_id: Optional[int] = None, robot_padding: int = 0):
        
        self.data_folder = data_folder
        self.spacing_px = int(spacing_px)
        self.plan_spacing_boost_px = int(plan_spacing_boost_px)
        self.line_margin_px = int(line_margin_px)
        self.robot_id = int(robot_id) if robot_id is not None else None
        self.robot_padding = int(robot_padding)

        # --- File Paths ---
        self.frame_path = os.path.join(data_folder, "frame_img.png")
        self.static_path = os.path.join(data_folder, "obstacles.txt")
        self.real_path = os.path.join(data_folder, "realtime_obstacles.txt")
        self.arena_path = os.path.join(data_folder, "arena_corners.txt")
        self.robot_path = os.path.join(data_folder, "robot_pos.txt")

        # --- Cached Data (initialized to None) ---
        self._mtimes: Dict[str, float] = {"frame": 0, "static": 0, "real": 0, "arena": 0, "robots": 0}
        self._H: Optional[int] = None
        self._W: Optional[int] = None
        self._planner: Optional[PathPlanner] = None
        self._mask_raw: Optional[np.ndarray] = None
        self._mask_dilated: Optional[np.ndarray] = None
        self._mask_plan: Optional[np.ndarray] = None
        self._mask_for_line: Optional[np.ndarray] = None
        self._dt_clearance_raw: Optional[np.ndarray] = None
        self._static_polys: List[Polygon] = []
        self._realtime_polys: List[Polygon] = []
        self._arena: Optional[List[Point]] = None

    def _ensure_size(self):
        """Read the frame image to determine the map dimensions (H, W)."""
        if self._H is not None and self._W is not None:
            return
        frame = cv2.imread(self.frame_path)
        if frame is None:
            raise FileNotFoundError(f"Could not load frame image at '{self.frame_path}'. Cannot determine map size.")
        self._H, self._W = frame.shape[:2]

    def _read_if_changed(self) -> bool:
        """
        Check file modification times and reload data if any have changed.
        Returns True if a change was detected, False otherwise.
        """
        changed = False
        paths_to_check = {
            "frame": self.frame_path, "static": self.static_path, "real": self.real_path,
            "arena": self.arena_path, "robots": self.robot_path
        }
        for key, path in paths_to_check.items():
            mtime = _get_mtime(path)
            if mtime != self._mtimes[key]:
                self._mtimes[key] = mtime
                changed = True
                # Reload data for the specific file that changed
                if key == "static":
                    self._static_polys = _parse_corners_file(path)
                elif key == "real":
                    self._realtime_polys = _parse_corners_file(path)
                elif key == "arena":
                    self._arena = _read_arena_corners(path)
        return changed

    def _read_robot_positions(self) -> List[Tuple[int, float, float, float]]:
        """Read all robot positions from the robot_pos.txt file."""
        robots = []
        try:
            with open(self.robot_path, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"): continue
                    parts = [p for p in line.replace(" ", ",").split(",") if p]
                    try:
                        if len(parts) == 4:
                            rid, x, y, th = map(float, parts)
                            robots.append((int(rid), x, y, th))
                    except (ValueError, IndexError):
                        continue
        except FileNotFoundError:
            pass
        return robots

    def rebuild_if_needed(self):
        """
        The core method of the class. Rebuilds the planner and all derived
        masks if any of the underlying data files have been modified.
        """
        if self._planner is not None and not self._read_if_changed():
            return  # No changes, no need to rebuild

        _dbg("PlannerCache: Rebuilding planner and masks due to file changes.")
        self._ensure_size()
        
        all_polygons = self._static_polys + self._realtime_polys
        obstacles = [{"corners": p} for p in all_polygons]
        self._planner = PathPlanner(obstacles, (self._H, self._W), arena_corners=self._arena)
        
        # 1. Create the raw obstacle mask
        self._mask_raw = self._planner.mask.copy()

        # 2. Add other robots as temporary obstacles
        if self.robot_padding > 0:
            for rid, x, y, _ in self._read_robot_positions():
                if self.robot_id is not None and rid == self.robot_id:
                    continue  # Don't add self as an obstacle
                cv2.circle(self._mask_raw, (int(round(x)), int(round(y))), self.robot_padding, 255, -1)
        
        # 3. Create dilated masks for path planning and line-of-sight checks
        kernel_spacing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.spacing_px + 1, 2 * self.spacing_px + 1))
        self._mask_dilated = cv2.dilate(self._mask_raw, kernel_spacing)

        plan_spacing = self.spacing_px + self.plan_spacing_boost_px
        kernel_plan = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * plan_spacing + 1, 2 * plan_spacing + 1))
        self._mask_plan = cv2.dilate(self._mask_raw, kernel_plan)

        if self.line_margin_px > 0:
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.line_margin_px + 1, 2 * self.line_margin_px + 1))
            self._mask_for_line = cv2.erode(self._mask_dilated, kernel_erode)
        else:
            self._mask_for_line = self._mask_dilated

        # 4. Generate the distance transform for clearance checks
        free_space_mask = np.where(self._mask_raw == 0, 255, 0).astype(np.uint8)
        self._dt_clearance_raw = cv2.distanceTransform(free_space_mask, cv2.DIST_L2, 3)

        # 5. Update the A* planner to use the most inflated mask
        self._planner.mask = self._mask_plan.copy()

    # --- Public Properties for Read-Only Access ---
    @property
    def planner(self) -> Optional[PathPlanner]:
        return self._planner

    @property
    def mask_raw(self) -> Optional[np.ndarray]:
        return self._mask_raw

    @property
    def mask_dilated(self) -> Optional[np.ndarray]:
        return self._mask_dilated

    @property
    def mask_plan(self) -> Optional[np.ndarray]:
        return self._mask_plan

    @property
    def mask_for_line(self) -> Optional[np.ndarray]:
        return self._mask_for_line

    @property
    def clearance_dt(self) -> Optional[np.ndarray]:
        return self._dt_clearance_raw

    @property
    def size(self) -> Tuple[Optional[int], Optional[int]]:
        return (self._H, self._W)