#!/usr/bin/env python3

import math
import os
import time
import json
import datetime
from pathlib import Path
import threading
import cv2
import numpy as np

from astar import PathPlanner

# Global registry of running robot controller threads
_robot_threads = {}  # { robot_id: {
                     #     "ctrl_thread": Thread,
                     #     "stop_event": Event,
                     #     "started_at": float,
                     #     "ctrl_done": bool
                     # } }
_robot_threads_lock = threading.Lock()

# Global lock for writing to the shared command.json file
_COMMAND_JSON_LOCK = threading.Lock()

# ---------- Debug ----------
DEBUG = True
DEBUG_EVERY = 5
def _dbg(msg: str, tick=None):
    if not DEBUG: return
    if tick is not None and tick % DEBUG_EVERY != 0: return
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[DBG {ts}] {msg}")

# ---------- Files / helpers ----------
def _ensure_txt(path, default_lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w') as f:
            for line in default_lines:
                f.write(f"{line}\n")
        _dbg(f"Initialized file: {path}")

# --- Trace file utils ---------------------------------------------------------
def _clear_trace_file(path: str):
    """Reset the trace file so old traces don't persist across runs."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"robots": []}, f)
        _dbg(f"Cleared trace file: {path}")
    except Exception as e:
        print(f"[WARN] Could not clear trace file '{path}': {e}")

def _robot_id_exists(pose_file: str, robot_id: int) -> bool:
    try:
        with open(pose_file, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p for p in line.replace(" ", ",").split(",") if p != ""]
                if len(parts) >= 4:
                    try:
                        rid = int(float(parts[0]))
                        if rid == int(robot_id):
                            return True
                    except ValueError:
                        continue
        return False
    except FileNotFoundError:
        return False

def _load_paths_json(json_path: str) -> dict:
    if not os.path.exists(json_path):
        return {"robots": []}
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "robots" in data and isinstance(data["robots"], list):
                return data
    except Exception:
        pass
    return {"robots": []}

# --- trace JSON helpers -----------------------------------------------
def _load_trace_json(json_path: str) -> dict:
    """
    Structure:
    { "robots": [ { "id": <int>, "points": [[x,y], ...] }, ... ] }
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    if not os.path.exists(json_path):
        return {"robots": []}
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "robots" in data and isinstance(data["robots"], list):
                return data
    except Exception:
        pass
    return {"robots": []}

def _trace_append_point(json_path: str, robot_id: int, x: float, y: float):
    data = _load_trace_json(json_path)
    rid = int(robot_id)
    # find or create entry
    entry = None
    for r in data["robots"]:
        try:
            if int(r.get("id", -1)) == rid:
                entry = r
                break
        except Exception:
            continue
    if entry is None:
        entry = {"id": rid, "points": []}
        data["robots"].append(entry)

    # dedup consecutive duplicates
    if entry["points"]:
        last = entry["points"][-1]
        if int(round(last[0])) == int(round(x)) and int(round(last[1])) == int(round(y)):
            return

    entry["points"].append([int(round(x)), int(round(y))])
    try:
        with open(json_path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[WARN] trace write failed: {e}")

# --- A* segment dump helpers --------------------------------------------------
def _astar_dump_clear(json_path: str, robot_id: int):
    """Remove all segments for this robot from the dump file."""
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        if not os.path.exists(json_path):
            with open(json_path, "w") as f:
                json.dump({"robots": []}, f)
            return

        with open(json_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "robots" not in data:
            data = {"robots": []}

        rid = int(robot_id)
        data["robots"] = [r for r in data["robots"] if int(r.get("id", -1)) != rid]

        with open(json_path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[WARN] astar_dump_clear failed: {e}")


def _astar_dump_append(json_path: str, robot_id: int, goal_xy, path_pts, *, seg_type="normal"):
    """
    Append one planned segment:
      goal_xy: (gx, gy)
      path_pts: [(x,y),...]
      seg_type: "normal" or "escape"
    File layout:
    {
      "robots":[
        {"id": 2, "segments":[
          {"ts": 1712345678.123, "type":"normal", "goal":[gx,gy], "path":[[x,y],...]}
        ]},
        ...
      ]
    }
    """
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        data = {"robots": []}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception:
                pass
        if not isinstance(data, dict):
            data = {"robots": []}
        if "robots" not in data or not isinstance(data["robots"], list):
            data["robots"] = []

        rid = int(robot_id)
        entry = None
        for r in data["robots"]:
            try:
                if int(r.get("id", -1)) == rid:
                    entry = r
                    break
            except Exception:
                continue
        if entry is None:
            entry = {"id": rid, "segments": []}
            data["robots"].append(entry)

        seg = {
            "type": seg_type,
            "goal": [int(goal_xy[0]), int(goal_xy[1])],
            "path": [[int(p[0]), int(p[1])] for p in path_pts],
        }
        entry.setdefault("segments", []).append(seg)

        with open(json_path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[WARN] astar_dump_append failed: {e}")

def _get_robot_path_from_json(json_path: str, robot_id: int):
    """
    Returns list of (x, y, theta_rad, action) for robot_id from json_path,
    where action is int delay_ms OR str 'open'/'close'.
    Returns None if not found or malformed.
    """
    data = _load_paths_json(json_path)
    for entry in data.get("robots", []):
        try:
            if int(entry.get("id", -1)) != int(robot_id):
                continue
            items = entry.get("path", [])
            out = []
            for t in items:
                if not isinstance(t, (list, tuple)) or len(t) < 2:
                    continue
                x = int(t[0]); y = int(t[1])
                theta = 0.0
                action = 0
                if len(t) >= 3:
                    third = t[2]
                    if isinstance(third, str):
                        s = third.strip().lower()
                        if s in ("open", "close"):
                            action = s
                    else:
                        try:
                            theta = float(third)
                        except Exception:
                            theta = 0.0
                if len(t) >= 4:
                    raw = t[3]
                    if isinstance(raw, str):
                        s = raw.strip().lower()
                        action = s if s in ("open", "close") else 0
                    else:
                        try:
                            action = int(raw)
                        except Exception:
                            action = 0
                out.append((x, y, theta, action))
            return out if out else None
        except Exception:
            continue
    return None

def _path_for_robot_exists(json_path: str, robot_id: int) -> bool:
    pts = _get_robot_path_from_json(json_path, robot_id)
    return pts is not None and len(pts) > 0

# ---------- File Interface ----------
class FileInterface:
    def __init__(self, target_file, pose_file, command_file, error_file, robot_id=None):
        """
        target_file can be a JSON path or legacy TXT.
        command_file is now assumed to be 'command.json'.
        """
        self.target_file = target_file
        self.pose_file = pose_file
        self.command_file = command_file  # This will be 'Data/command.json'
        self.error_file = error_file
        self.robot_id = None if robot_id is None else int(robot_id)
        self._initialize_files()

    def _initialize_files(self):
        if not str(self.target_file).lower().endswith(".json"):
            _ensure_txt(self.target_file, ["0,0,0"])
        _ensure_txt(self.pose_file, ["0,0,0.0"])
        # Initialize command.json with an empty structure
        os.makedirs(os.path.dirname(self.command_file), exist_ok=True)
        if not os.path.exists(self.command_file):
            with open(self.command_file, 'w') as f:
                json.dump({"robots": []}, f)
        _ensure_txt(self.error_file, ["0.0,0.0"])

    def _load_lines(self, path):
        try:
            with open(path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"[WARN] Error reading {path}: {e}")
            return []

    def read_targets(self):
        if str(self.target_file).lower().endswith(".json"):
            if self.robot_id is None:
                print("[WARN] target_file is JSON but robot_id not provided; no targets.")
                return []
            quads = _get_robot_path_from_json(self.target_file, self.robot_id)
            return quads if quads is not None else []
        lines = self._load_lines(self.target_file)
        out = []
        for line in lines:
            try:
                parts = [p.strip() for p in line.split(',') if p.strip() != ""]
                if len(parts) < 2: continue
                x = float(parts[0]); y = float(parts[1])
                delay_ms = int(float(parts[2])) if len(parts) >= 3 else 0
                out.append((x, y, 0.0, delay_ms))
            except Exception as e:
                print(f"[WARN] Skipping malformed target '{line}': {e}")
        return out

    def read_pos(self):
        lines = self._load_lines(self.pose_file)
        if not lines:
            return None  # File is empty or could not be read

        # If a robot ID is specified, search for it exclusively.
        if self.robot_id is not None:
            for line in reversed(lines):
                parts = [p for p in line.replace(" ", ",").split(",") if p != ""]
                if len(parts) == 4:
                    try:
                        rid = int(float(parts[0]))
                        if rid == self.robot_id:
                            # Found the robot, return its position
                            return float(parts[1]), float(parts[2]), float(parts[3])
                    except ValueError:
                        continue
            # If the loop finishes without finding the ID, return None.
            return None
        
        # If no robot ID was specified, fall back to legacy single-robot mode.
        else:
            try:
                x_str, y_str, th_str = lines[-1].split(',')
                return float(x_str), float(y_str), float(th_str)
            except Exception as e:
                print(f"[WARN] Malformed legacy pose '{lines[-1]}': {e}")
                return None

    def _update_command_json(self, updates: dict):
        """
        A thread-safe method to read, update, and write command.json.
        'updates' is a dictionary of key-value pairs to update for this robot.
        e.g., {"left": 90, "right": 90} or {"gripper": "open"}
        """
        if self.robot_id is None:
            print("[ERR] Cannot write command without a robot_id.")
            return

        with _COMMAND_JSON_LOCK:
            try:
                data = {"robots": []}
                if os.path.exists(self.command_file):
                    with open(self.command_file, 'r') as f:
                        try:
                            file_data = json.load(f)
                            if isinstance(file_data, dict) and "robots" in file_data:
                                data = file_data
                        except json.JSONDecodeError:
                            pass
                robot_entry = None
                for robot in data.get("robots", []):
                    if robot.get("id") == self.robot_id:
                        robot_entry = robot
                        break
                if robot_entry is None:
                    robot_entry = {"id": self.robot_id}
                    data["robots"].append(robot_entry)
                robot_entry.update(updates)
                with open(self.command_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"[ERR] Saving command to {self.command_file} failed: {e}")

    def write_wheel_command(self, left_speed, right_speed):
        """Writes wheel speeds to the JSON command file."""
        updates = {
            "left": int(round(left_speed)),
            "right": int(round(right_speed)),
        }
        self._update_command_json(updates)

    def write_gripper_command(self, state: str):
        """Writes a gripper state to the JSON command file."""
        if state.lower() in ["open", "close"]:
            self._update_command_json({"gripper": state.lower()})

    def log_error(self, dist_err, angle_err):
        try:
            with open(self.error_file, 'w') as f:
                f.write(f"{dist_err},{angle_err}\n")
        except IOError as e:
            print(f"[ERR] Saving error failed: {e}")

# ---------- Parsing ----------
def _parse_corners_file(path):
    polys = []
    try:
        with open(path, "r") as f:
            for line in f:
                s = line.strip().replace("(", "").replace(")", "")
                if not s: continue
                parts = s.split(",")
                if len(parts) != 8: continue
                try: nums = list(map(int, parts))
                except ValueError: continue
                polys.append([(nums[i], nums[i+1]) for i in range(0, 8, 2)])
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[WARN] Could not read '{path}]: {e}")
    return polys

def _read_arena_corners(path):
    pts = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                x, y = map(int, line.split(","))
                pts.append((x, y))
        return pts if pts else None
    except Exception:
        return None

# ---------- Geometry helpers ----------
def _densify_segment(p0, p1, step_px=12):
    x0, y0 = p0; x1, y1 = p1
    dx, dy = (x1 - x0), (y1 - y0)
    dist = math.hypot(dx, dy)
    if dist <= 1e-6: return [p0]
    n = max(1, int(dist // step_px))
    out = []
    for k in range(n):
        t = k / n
        out.append((int(round(x0 + t * dx)), int(round(y0 + t * dy))))
    out.append((int(round(x1)), int(round(y1))))
    return out

def _densify_polyline(path, step_px=12):
    if not path or len(path) == 1: return list(path)
    out = []
    for i in range(len(path) - 1):
        seg = _densify_segment(path[i], path[i+1], step_px=step_px)
        if i > 0 and seg: seg = seg[1:]
        out.extend(seg)
    return out

def _bresenham_blocked(mask, p0, p1):
    if mask is None: return False
    x0, y0 = map(int, p0); x1, y1 = map(int, p1)
    H, W = mask.shape[:2]
    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < W and 0 <= y0 < H and mask[y0, x0] != 0: return True
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 >= dy: err += dy; x0 += sx
        if e2 <= dx: err += dx; y0 += sy
    return False

def _line_blocked_multi(masks, p0, p1):
    for m in masks:
        if m is not None and _bresenham_blocked(m, p0, p1):
            return True
    return False

def _min_clearance_along_line(dt, p0, p1, stride=3):
    if dt is None: return 0.0
    x0, y0 = map(int, p0); x1, y1 = map(int, p1)
    H, W = dt.shape[:2]
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    xs = np.linspace(x0, x1, n + 1, dtype=int)[::max(1, stride)]
    ys = np.linspace(y0, y1, n + 1, dtype=int)[::max(1, stride)]
    xs = np.clip(xs, 0, W - 1); ys = np.clip(ys, 0, H - 1)
    return float(dt[ys, xs].min())

def _line_ok(masks, dt, p0, p1, clearance_req_px):
    if _line_blocked_multi(masks, p0, p1): return False
    if clearance_req_px <= 0: return True
    return _min_clearance_along_line(dt, p0, p1) >= float(clearance_req_px)

def _polyline_ok(dt, polyline, clearance_req_px, stride=3):
    if clearance_req_px <= 0: return True
    for i in range(len(polyline)-1):
        if _min_clearance_along_line(dt, polyline[i], polyline[i+1], stride=stride) < float(clearance_req_px):
            return False
    return True

# ---------- Cache / masks ----------
def _get_mtime(path):
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0
class PlannerCache:
    def __init__(self, data_folder, spacing_px=26, plan_spacing_boost_px=8, line_margin_px=1,
                 robot_id=None, robot_padding=0):
        self.data_folder = data_folder
        self.spacing_px = int(spacing_px)
        self.plan_spacing_boost_px = int(plan_spacing_boost_px)
        self.line_margin_px = int(line_margin_px)
        self.frame_path = os.path.join(data_folder, "frame_img.png")
        self.static_path = os.path.join(data_folder, "obstacles.txt")
        self.real_path   = os.path.join(data_folder, "realtime_obstacles.txt")
        self.arena_path  = os.path.join(data_folder, "arena_corners.txt")
        self.robot_path  = os.path.join(data_folder, "robot_pos.txt")
        self.robot_id = None if robot_id is None else int(robot_id)
        self.robot_padding = int(robot_padding)
        self._mtimes = {"frame":0, "static":0, "real":0, "arena":0, "robots":0}
        self._H = None; self._W = None
        self._planner = None
        self._mask_raw = None
        self._mask_dilated = None
        self._mask_plan = None
        self._mask_for_line = None
        self._kernel_dilate_spacing = None
        self._kernel_dilate_plan = None
        self._kernel_erode  = None
        self._dt_clearance_raw = None
        self._static_polys = []
        self._realtime_polys = []
        self._arena = None

    def _ensure_size(self):
        if self._H is not None and self._W is not None: return
        frame = cv2.imread(self.frame_path)
        if frame is None:
            raise FileNotFoundError(f"Could not load '{self.frame_path}'. Capture a frame first.")
        self._H, self._W = frame.shape[:2]

    def _read_if_changed(self):
        changed = False
        m = _get_mtime(self.frame_path)
        if m != self._mtimes["frame"]: self._mtimes["frame"] = m; changed = True
        m = _get_mtime(self.static_path)
        if m != self._mtimes["static"]:
            self._static_polys = _parse_corners_file(self.static_path)
            self._mtimes["static"] = m; changed = True
        m = _get_mtime(self.real_path)
        if m != self._mtimes["real"]:
            self._realtime_polys = _parse_corners_file(self.real_path)
            self._mtimes["real"] = m; changed = True
        m = _get_mtime(self.arena_path)
        if m != self._mtimes["arena"]:
            self._arena = _read_arena_corners(self.arena_path)
            self._mtimes["arena"] = m; changed = True
        m = _get_mtime(self.robot_path)
        if m != self._mtimes["robots"]: self._mtimes["robots"] = m; changed = True
        return changed

    def _read_robot_positions(self):
        robots = []
        try:
            with open(self.robot_path, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"): continue
                    parts = [p for p in line.replace(" ", ",").split(",") if p != ""]
                    if len(parts) == 4:
                        try:
                            rid=int(float(parts[0])); x=float(parts[1]); y=float(parts[2]); th=float(parts[3])
                            robots.append((rid, x, y, th))
                        except ValueError: continue
                    elif len(parts) == 3:
                        try:
                            x=float(parts[0]); y=float(parts[1]); th=float(parts[2])
                            robots.append((0, x, y, th))
                        except ValueError: continue
        except FileNotFoundError: pass
        return robots

    def rebuild_if_needed(self):
        if not hasattr(self, "_planner"): self._planner = None
        files_changed = self._read_if_changed()
        if self._planner is not None and not files_changed: return
        self._ensure_size()
        obstacles = [{"corners": p} for p in (self._static_polys + self._realtime_polys)]
        self._planner = PathPlanner(obstacles, (self._H, self._W), arena_corners=self._arena)
        self._mask_raw = self._planner.mask.copy()
        if self.robot_padding > 0:
            H, W = self._mask_raw.shape[:2]
            for rid, x, y, _th in self._read_robot_positions():
                if self.robot_id is not None and int(rid) == self.robot_id: continue
                cx, cy = int(round(x)), int(round(y))
                if 0 <= cx < W and 0 <= cy < H:
                    cv2.circle(self._mask_raw, (cx, cy), int(self.robot_padding), 255, thickness=-1)
        ks = 2 * int(self.spacing_px) + 1
        if self._kernel_dilate_spacing is None or self._kernel_dilate_spacing.shape[0] != ks:
            self._kernel_dilate_spacing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        self._mask_dilated = cv2.dilate(self._mask_raw, self._kernel_dilate_spacing)
        kp = 2 * int(self.spacing_px + self.plan_spacing_boost_px) + 1
        if self._kernel_dilate_plan is None or self._kernel_dilate_plan.shape[0] != kp:
            self._kernel_dilate_plan = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kp, kp))
        self._mask_plan = cv2.dilate(self._mask_raw, self._kernel_dilate_plan)
        if self.line_margin_px > 0:
            km = 2 * int(self.line_margin_px) + 1
            if self._kernel_erode is None or self._kernel_erode.shape[0] != km:
                self._kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (km, km))
            self._mask_for_line = cv2.erode(self._mask_dilated, self._kernel_erode)
        else: self._mask_for_line = self._mask_dilated
        free_u8 = np.where(self._mask_raw == 0, 255, 0).astype(np.uint8)
        self._dt_clearance_raw = cv2.distanceTransform(free_u8, cv2.DIST_L2, 3)
        self._planner.mask = self._mask_plan.copy()
        _dbg("Planner rebuilt (files changed).")

    @property
    def planner(self): return self._planner
    @property
    def mask_raw(self): return self._mask_raw
    @property
    def mask_dilated(self): return self._mask_dilated
    @property
    def mask_plan(self): return self._mask_plan
    @property
    def mask_for_line(self): return self._mask_for_line
    @property
    def kernel_erode(self): return self._kernel_erode
    @property
    def clearance_dt(self): return self._dt_clearance_raw
    @property
    def size(self): return (self._H, self._W)

# ---------- ESCAPE ----------
def _nearest_free(mask, x, y, r_max=50):
    H, W = mask.shape[:2]
    x = int(np.clip(x, 0, W - 1)); y = int(np.clip(y, 0, H - 1))
    if mask[y, x] == 0: return (x, y)
    for r in range(1, int(r_max) + 1):
        x0, x1 = max(0, x - r), min(W - 1, x + r)
        y0, y1 = max(0, y - r), min(H - 1, y + r)
        for xx in range(x0, x1 + 1):
            if mask[y0, xx] == 0: return (xx, y0)
            if mask[y1, xx] == 0: return (xx, y1)
        for yy in range(y0 + 1, y1):
            if mask[yy, x0] == 0: return (x0, yy)
            if mask[yy, x1] == 0: return (x1, yy)
    return None

def _make_escape_masks(raw_mask, dilated_mask, sx, sy, r, kernel_erode=None):
    inflated2 = dilated_mask.copy()
    H, W = inflated2.shape[:2]
    sx = int(np.clip(sx, 0, W-1)); sy = int(np.clip(sy, 0, H-1))
    bubble = np.zeros_like(inflated2, np.uint8)
    cv2.circle(bubble, (sx, sy), int(r), 255, -1)
    only_inflation = cv2.bitwise_and(bubble, cv2.bitwise_not((raw_mask > 0).astype(np.uint8)*255))
    inflated2[only_inflation > 0] = 0
    if kernel_erode is not None:
        line2 = cv2.erode(inflated2, kernel_erode)
    else: line2 = inflated2
    return inflated2, line2

def _find_escape_waypoint(line_mask, inflated_mask, raw_mask, sx, sy, r_samples, dirs=36):
    H, W = inflated_mask.shape[:2]
    sx = int(np.clip(sx, 0, W-1)); sy = int(np.clip(sy, 0, H-1))
    for r in r_samples:
        for th in np.linspace(0, 2*np.pi, dirs, endpoint=False):
            ex = int(round(sx + r * math.cos(th)))
            ey = int(round(sy + r * math.sin(th)))
            if not (0 <= ex < W and 0 <= ey < H): continue
            if inflated_mask[ey, ex] != 0: continue
            if _bresenham_blocked(line_mask, (sx, sy), (ex, ey)): continue
            if _bresenham_blocked(raw_mask,  (sx, sy), (ex, ey)): continue
            return (ex, ey)
    return None

# ---------- Planning (cached masks) ----------
def _plan_segment_cached(cache: PlannerCache, start_xy, goal_xy,
                         simplify_dist=10, step_px=12, offset_px=100,
                         nearest_free_radius=50, clearance_req_px=0):
    planner=cache.planner; mask_raw=cache.mask_raw; mask_dil=cache.mask_dilated
    mask_line=cache.mask_for_line; dt=cache.clearance_dt
    sx, sy = map(int, start_xy); gx, gy = map(int, goal_xy)
    if mask_dil[sy, sx] != 0:
        snap = _nearest_free(mask_dil, sx, sy, r_max=nearest_free_radius)
        if snap is None: _dbg(f"START_ENCLOSED"); return None
        _dbg(f"START_SNAP: {(sx, sy)} -> {snap}"); sx, sy = snap
    if mask_dil[gy, gx] != 0:
        snap = _nearest_free(mask_dil, gx, gy, r_max=nearest_free_radius)
        if snap is None: _dbg(f"GOAL_ENCLOSED"); return None
        _dbg(f"GOAL_SNAP: {(gx, gy)} -> {snap}"); gx, gy = snap
    if _line_ok([mask_raw, mask_line], dt, (sx, sy), (gx, gy), clearance_req_px):
        return _densify_segment((sx, sy), (gx, gy), step_px=step_px)
    path = planner.find_obstacle_aware_path((sx, sy), (gx, gy), simplify_dist)
    if path:
        poly = _densify_polyline(path, step_px=step_px)
        if _polyline_ok(dt, poly, clearance_req_px): return poly
    for dx, dy in [(0, -offset_px), (offset_px, 0), (0, offset_px), (-offset_px, 0)]:
        nx, ny = gx + dx, gy + dy; H, W = mask_dil.shape[:2]
        if not (0 <= nx < W and 0 <= ny < H): continue
        if mask_dil[ny, nx]: continue
        if _line_ok([mask_raw, mask_line], dt, (sx, sy), (nx, ny), clearance_req_px):
            return _densify_segment((sx, sy), (nx, ny), step_px=step_px)
        p = planner.find_obstacle_aware_path((sx, sy), (nx, ny), simplify_dist)
        if p:
            poly = _densify_polyline(p, step_px=step_px)
            if _polyline_ok(dt, poly, clearance_req_px): return poly
    return None

def _goal_for_robot_center(gx: float, gy: float, gtheta: float, offset_px: float):
    try: th = float(gtheta)
    except Exception: th = 0.0
    return (gx - offset_px*math.cos(th), gy - offset_px*math.sin(th))

def _grip_pre_approach(gx: float, gy: float, gtheta: float, offset_px: float, extra_px: float):
    try: th = float(gtheta)
    except Exception: th = 0.0
    cx, cy = _goal_for_robot_center(gx, gy, th, offset_px)
    px = gx - (offset_px + extra_px)*math.cos(th)
    py = gy - (offset_px + extra_px)*math.sin(th)
    return (px, py), (cx, cy)

# ---------- PID Controller ----------
class PIDController:
    def __init__(
        self, iface: FileInterface,
        Kp_dist=0.18, Kp_ang=3.0, Ki_ang=0.02, Kd_ang=0.9,
        dist_tolerance=18.0, ang_tolerance=18.0, final_distance_tol=10.0,
        data_folder="Data", spacing_px=40, plan_spacing_boost_px=8,
        linear_step_px=12, simplify_dist_px=10,
        offset_px=100, replan_check_period_s=0.10, replan_wait_backoff_s=0.20,
        speed_min=70, speed_max=110, speed_neutral=90, max_lin=30,
        nearest_free_radius=50, linecheck_margin_px=1,
        clearance_boost_px=0,
        robot_id=None, robot_padding=0,
        gripper_action_pause_s=0.4,
        trace_json_path: str = str(Path("Data") / "trace_paths.json"),
        trace_stride_px: float = 2.0,
        trace_flush_every: int = 1,
        astar_dump_path: str = str(Path("Data") / "astar_segments.json"),
        clear_astar_dump_on_start: bool = True,
        gripper_offset_px: float = 30.0,
        gripper_pre_approach_extra_px: float = 50.0,
        post_drop_retreat_px: float = 20.0,
    ):
        self.iface = iface
        self.Kp_dist = Kp_dist; self.Kp_ang = Kp_ang; self.Ki_ang = Ki_ang; self.Kd_ang = Kd_ang
        self.dist_tolerance = float(dist_tolerance); self.ang_tolerance_deg = float(ang_tolerance)
        self.final_distance_tol = float(final_distance_tol)
        self.speed_min=int(speed_min); self.speed_max=int(speed_max)
        self.speed_neutral=int(speed_neutral); self.max_lin=float(max_lin)
        self.cache = PlannerCache(data_folder, spacing_px=spacing_px,
            plan_spacing_boost_px=plan_spacing_boost_px, line_margin_px=linecheck_margin_px,
            robot_id=robot_id, robot_padding=robot_padding)
        self.linear_step_px = int(linear_step_px); self.simplify_dist_px = int(simplify_dist_px)
        self.offset_px = int(offset_px); self.nearest_free_radius = int(nearest_free_radius)
        self.replan_check_period_s = float(replan_check_period_s)
        self.replan_wait_backoff_s = float(replan_wait_backoff_s)
        self.clearance_req_px = int(self.cache.spacing_px + int(clearance_boost_px))
        self.integral_ang = 0.0; self.prev_ang_err = 0.0; self.prev_time = time.time()
        self._seg_path = None; self._seg_index = 0; self._seg_goal = None; self._tick = 0
        self._last_pose = None; self._pose_still_cnt = 0
        self._escape_active = False; self._escape_line_mask = None
        self.dir_v_thresh = 1.0; self.dir_ang_thresh = 1.0
        self.gripper_action_pause_s = float(gripper_action_pause_s)
        self.trace_json_path = str(trace_json_path)
        self.trace_stride_px = float(trace_stride_px)
        self.trace_flush_every = int(max(1, trace_flush_every))
        self._trace_last_xy = None; self._trace_since_flush = 0
        self._robot_id_for_trace = int(robot_id) if robot_id is not None else 0
        self.cache.rebuild_if_needed()
        self.astar_dump_path = str(astar_dump_path)
        self._robot_id_for_dump = int(robot_id) if robot_id is not None else 0
        if clear_astar_dump_on_start:
            _astar_dump_clear(self.astar_dump_path, self._robot_id_for_dump)
        self.gripper_offset_px = float(gripper_offset_px)
        self.gripper_pre_approach_extra_px = float(gripper_pre_approach_extra_px)
        self.post_drop_retreat_px = float(post_drop_retreat_px)
        self._grip_state = None

    @staticmethod
    def normalize(angle): return math.atan2(math.sin(angle), math.cos(angle))
    @staticmethod
    def _deg(rad): return math.degrees(rad)

    def _maybe_log_trace(self, x: float, y: float):
        try:
            cur = (float(x), float(y))
            if self._trace_last_xy is None:
                self._trace_last_xy = cur
                _trace_append_point(self.trace_json_path, self._robot_id_for_trace, x, y)
                self._trace_since_flush = 0
                return
            dx = cur[0] - self._trace_last_xy[0]; dy = cur[1] - self._trace_last_xy[1]
            if (dx*dx + dy*dy) >= (self.trace_stride_px * self.trace_stride_px):
                self._trace_last_xy = cur
                _trace_append_point(self.trace_json_path, self._robot_id_for_trace, x, y)
                self._trace_since_flush += 1
        except Exception as e:
            print(f"[WARN] trace log error: {e}")

    def _direction_label(self, v_cmd: float, ang_cmd: float) -> str:
        v_mag = abs(v_cmd); a_mag = abs(ang_cmd)
        if v_mag >= self.dir_v_thresh and a_mag < self.dir_ang_thresh: return "forward" if v_cmd > 0 else "backward"
        if v_mag < self.dir_v_thresh and a_mag >= self.dir_ang_thresh: return "anticlockwise" if ang_cmd > 0 else "clockwise"
        if v_mag < self.dir_v_thresh and a_mag < self.dir_ang_thresh: return "stopped"
        return "forward" if v_cmd > 0 else "backward"

    def _adjust_speed(self, s):
        if 89.75 <= s <= 90.25: return 90.0
        if 90.25 < s <= 95: return s + 1.0
        if 85 <= s < 89.75: return s - 1.0
        return s

    def _pause_at_checkpoint(self, delay_s, stop_event=None):
        if delay_s <= 0: return
        self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)
        _dbg(f"REACHED target; pausing {delay_s:.2f}s")
        t0 = time.time()
        while (time.time() - t0) < delay_s:
            if stop_event and stop_event.is_set(): break
            time.sleep(0.03)

    def _do_gripper_action(self, action: str):
        """Send 'open' or 'close' by writing to the command.json file."""
        action = action.strip().lower()
        if action not in ("open", "close"): return
        self.iface.write_gripper_command(action)

    def _try_escape_plan(self, start_xy):
        if self.cache.planner is None: return None
        sx, sy = map(int, start_xy)
        mask_raw = self.cache.mask_raw; mask_dil = self.cache.mask_dilated
        if mask_raw is None or mask_dil is None: return None
        H, W = mask_dil.shape[:2]
        sx = int(np.clip(sx, 0, W-1)); sy = int(np.clip(sy, 0, H-1))
        if mask_dil[sy, sx] == 0: return None
        km = self.cache.kernel_erode
        bubble_r = max(8, self.cache.spacing_px + 4)
        inflated2, line2 = _make_escape_masks(mask_raw, mask_dil, sx, sy, r=bubble_r, kernel_erode=km)
        esc_wp = _find_escape_waypoint(line2, inflated2, mask_raw, sx, sy,
            r_samples=(self.cache.spacing_px, self.cache.spacing_px+10, self.cache.spacing_px+20), dirs=36)
        if esc_wp is None: _dbg("ESCAPE: no boundary point found."); return None
        dtc = self.cache.clearance_dt
        if _min_clearance_along_line(dtc, (sx, sy), esc_wp) < max(4, self.clearance_req_px * 0.5):
            _dbg("ESCAPE: candidate too close to obstacle."); return None
        _dbg(f"ESCAPE: temporary waypoint {esc_wp} created.")
        self._escape_active = True; self._escape_line_mask = line2
        return _densify_segment((sx, sy), esc_wp, step_px=self.linear_step_px)

    def _ensure_segment_path(self, start_xy, final_goal_xy):
        next_check = getattr(self, "_next_check", 0.0); now = time.time()
        if now >= next_check:
            self.cache.rebuild_if_needed()
            self._next_check = now + self.replan_check_period_s
        if self.cache.planner is None or self.cache.mask_dilated is None: return False
        sx, sy = int(start_xy[0]), int(start_xy[1])
        goal_tuple = (int(final_goal_xy[0]), int(final_goal_xy[1]))
        if self.cache.mask_dilated[sy, sx] != 0:
            if self._escape_active and self._seg_path is not None and self._seg_index < len(self._seg_path):
                cur = (sx, sy); nxt = (int(self._seg_path[self._seg_index][0]), int(self._seg_path[self._seg_index][1]))
                if self._escape_line_mask is None or _bresenham_blocked(self._escape_line_mask, cur, nxt):
                    _dbg(f"ESCAPE_LINE_BLOCKED", self._tick); self._seg_path = None
            if self._seg_path is None:
                path = self._try_escape_plan((sx, sy))
                if not path: _dbg("ESCAPE: could not synthesize path.", self._tick); return False
                self._seg_path = path; self._seg_index = 0; self._seg_goal = goal_tuple
                _dbg(f"NEW_PLAN(ESCAPE): len={len(path)}", self._tick)
                try: _astar_dump_append(self.astar_dump_path, self._robot_id_for_dump,
                                    goal_tuple, self._seg_path, seg_type="escape")
                except Exception as e: print(f"[WARN] astar escape dump failed: {e}")
            return True
        need_new = False
        if (self._seg_path is None or self._seg_goal != goal_tuple or self._escape_active or
            (self._seg_path is not None and self._seg_index >= len(self._seg_path))):
            if self._seg_path is not None and self._seg_index >= len(self._seg_path): _dbg("END_OF_POLYLINE", self._tick)
            if self._escape_active: _dbg("ESCAPE_DONE", self._tick)
            self._escape_active = False; self._escape_line_mask = None; need_new = True
        else:
            if self._seg_index < len(self._seg_path):
                cur = (sx, sy); nxt = (int(self._seg_path[self._seg_index][0]), int(self._seg_path[self._seg_index][1]))
                if self.cache.mask_dilated[nxt[1], nxt[0]] != 0: _dbg(f"WAYPOINT_BLOCKED", self._tick); need_new = True
                elif _line_blocked_multi([self.cache.mask_raw, self.cache.mask_for_line], cur, nxt):
                    _dbg(f"LINE_BLOCKED", self._tick); need_new = True
        if need_new:
            path = _plan_segment_cached(self.cache, (sx, sy), goal_tuple,
                simplify_dist=self.simplify_dist_px, step_px=self.linear_step_px, offset_px=self.offset_px,
                nearest_free_radius=self.nearest_free_radius, clearance_req_px=self.clearance_req_px)
            if not path: _dbg("NO_PLAN", self._tick); return False
            self._seg_path = path; self._seg_index = 0; self._seg_goal = goal_tuple
            _dbg(f"NEW_PLAN: len={len(path)} to goal={goal_tuple}", self._tick)
            try: _astar_dump_append(self.astar_dump_path, self._robot_id_for_dump,
                                goal_tuple, self._seg_path, seg_type="normal")
            except Exception as e: print(f"[WARN] astar dump failed: {e}")
        return True

    def _reverse_retreat(self, retreat_px: float, hold_theta: float, start_xy: tuple[float,float], stop_event=None):
        if retreat_px <= 0: return
        x0, y0 = float(start_xy[0]), float(start_xy[1]); target_theta = float(hold_theta)
        back_lin = -min(self.max_lin * 0.6, 18.0); ang_I = 0.0; prev_err = 0.0
        t_prev = time.time(); TIMEOUT = 3.0
        while True:
            if stop_event and stop_event.is_set(): break
            now = time.time(); dt = max(1e-3, now - t_prev); t_prev = now
            x, y, theta = self.iface.read_pos()
            if math.hypot(x - x0, y - y0) >= float(retreat_px): break
            ang_err = self.normalize(target_theta - theta); ang_I += ang_err * dt
            ang_I = max(-0.4, min(0.4, ang_I)); ang_D = (ang_err - prev_err) / dt; prev_err = ang_err
            ang_ctrl = (self.Kp_ang * ang_err) + (self.Ki_ang * ang_I) + (self.Kd_ang * ang_D)
            left = self.speed_neutral + back_lin - ang_ctrl; right = self.speed_neutral + back_lin + ang_ctrl
            left = max(self.speed_min, min(self.speed_max, self._adjust_speed(left)))
            right = max(self.speed_min, min(self.speed_max, self._adjust_speed(right)))
            self.iface.write_wheel_command(left, right)
            self.iface.log_error(0.0, ang_err)
            if (now - t_prev) > TIMEOUT: break
            time.sleep(0.03)
        self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)

    def run(self, stop_event=None):
        targets = self.iface.read_targets()
        if not targets:
            print(f"No targets found in {self.iface.target_file}")
            return

        idx = 0
        while idx < len(targets) and not (stop_event and stop_event.is_set()):
            self._tick += 1
            now = time.time()
            dt = max(1e-3, now - self.prev_time); self.prev_time = now

            pose = self.iface.read_pos()
            if pose is None:
                _dbg(f"Pose for robot {self.iface.robot_id} not found in '{self.iface.pose_file}'. Sending stop command.", self._tick)
                self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)
                time.sleep(0.5)
                continue

            x, y, theta = pose
            self._maybe_log_trace(x, y)

            cur_pose = (round(x,1), round(y,1))
            if self._last_pose == cur_pose:
                self._pose_still_cnt += 1
                if self._pose_still_cnt % (20*DEBUG_EVERY) == 0:
                    _dbg("POSE_STALE: robot_pos.txt not changing — check upstream pose publisher.")
            else:
                self._pose_still_cnt = 0
            self._last_pose = cur_pose

            goal_x, goal_y, goal_theta, action = targets[idx]
            is_gripper_action = isinstance(action, str) and action in ("open", "close")

            if is_gripper_action and self._grip_state is None:
                theta_dyn = math.atan2(goal_y - y, goal_x - x)
                pre_xy, cen_xy = _grip_pre_approach(
                    goal_x, goal_y, theta_dyn,
                    self.gripper_offset_px,
                    self.gripper_pre_approach_extra_px
                )
                self._grip_state = {
                    "pre": (float(pre_xy[0]), float(pre_xy[1])),
                    "cen": (float(cen_xy[0]), float(cen_xy[1])),
                    "theta": float(theta_dyn),
                    "stage": "to_pre"
                }
                self._seg_path = None
                self._seg_index = 0

            if is_gripper_action and self._grip_state is not None:
                if self._grip_state["stage"] == "to_pre":
                    plan_goal_xy = self._grip_state["pre"]
                elif self._grip_state["stage"] == "to_center":
                    plan_goal_xy = self._grip_state["cen"]
                elif self._grip_state["stage"] == "retreat":
                    plan_goal_xy = self._grip_state["retreat"]
                else:
                    plan_goal_xy = (goal_x, goal_y)
            else:
                plan_goal_xy = (goal_x, goal_y)
            
            if is_gripper_action and self._grip_state is not None and self._grip_state["stage"] == "to_pre":
                theta_live = math.atan2(goal_y - y, goal_x - x)
                if abs(PIDController.normalize(theta_live - self._grip_state["theta"])) > math.radians(8):
                    pre_xy, cen_xy = _grip_pre_approach(
                        goal_x, goal_y, theta_live,
                        self.gripper_offset_px,
                        self.gripper_pre_approach_extra_px
                    )
                    self._grip_state["pre"] = (float(pre_xy[0]), float(pre_xy[1]))
                    self._grip_state["cen"] = (float(cen_xy[0]), float(cen_xy[1]))
                    self._grip_state["theta"] = float(theta_live)
                    self._seg_path = None
                    self._seg_index = 0
                    plan_goal_xy = self._grip_state["pre"]

            if not self._ensure_segment_path((x, y), plan_goal_xy):
                _dbg(f"NO_PLAN: waiting for feasible path to target {idx+1} ({goal_x:.1f},{goal_y:.1f})...", self._tick)
                self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)
                while not (stop_event and stop_event.is_set()):
                    time.sleep(self.replan_wait_backoff_s)
                    pose = self.iface.read_pos() # --- FIX ---: Read to a temp variable
                    if pose is None: # --- FIX ---: Check if pose is None
                        _dbg("Pose lost while waiting for a plan. Continuing to wait.", self._tick)
                        self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)
                        continue # --- FIX ---: Continue waiting
                    x, y, theta = pose # --- FIX ---: Unpack only if valid

                    self._maybe_log_trace(x, y)
                    if self._ensure_segment_path((x, y), plan_goal_xy):
                        _dbg("PLAN_AVAILABLE: resuming motion.", self._tick)
                        break
                if stop_event and stop_event.is_set(): break
                continue

            if not self._seg_path:
                self._seg_path = None
                self._seg_index = 0
                continue

            cur_wp = self._seg_path[min(self._seg_index, len(self._seg_path)-1)]
            tx, ty = float(cur_wp[0]), float(cur_wp[1])

            dist_to_wp = math.hypot(tx - x, ty - y)
            if dist_to_wp < self.dist_tolerance:
                self._seg_index = min(self._seg_index + 1, len(self._seg_path))
                _dbg(f"ADVANCE: waypoint {self._seg_index}/{len(self._seg_path)}", self._tick)

                if self._escape_active and self._seg_index >= len(self._seg_path):
                    self._escape_active = False
                    self._escape_line_mask = None
                    _dbg("ESCAPE_DONE: back to normal planning.", self._tick)

                if self._seg_index >= len(self._seg_path):
                    if math.hypot(plan_goal_xy[0] - x, plan_goal_xy[1] - y) < self.final_distance_tol:
                        self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)
                        if is_gripper_action and self._grip_state is not None:
                            if self._grip_state["stage"] == "to_pre":
                                ang_tol_rad = math.radians(self.ang_tolerance_deg)
                                # --- FIX START ---
                                # We need to check the initial pose read here as well
                                initial_pose = self.iface.read_pos()
                                if initial_pose is None:
                                    _dbg("Cannot get initial pose for pre-align. Skipping alignment.", self._tick)
                                else:
                                    x, y, theta = initial_pose
                                    t_align_start = time.time()
                                    ALIGN_TIMEOUT_S = 3.0

                                    while True:
                                        ang_err_goal = self.normalize(self._grip_state["theta"] - theta)
                                        if abs(ang_err_goal) <= ang_tol_rad:
                                            break
                                        now2 = time.time()
                                        dt2 = max(1e-3, now2 - self.prev_time)
                                        self.prev_time = now2
                                        self.integral_ang += ang_err_goal * dt2
                                        I_LIM = 0.4
                                        self.integral_ang = max(-I_LIM, min(I_LIM, self.integral_ang))
                                        deriv_ang = (ang_err_goal - self.prev_ang_err) / dt2
                                        self.prev_ang_err = ang_err_goal
                                        ang_ctrl = (self.Kp_ang * ang_err_goal) + (self.Ki_ang * self.integral_ang) + (self.Kd_ang * deriv_ang)
                                        left  = self.speed_neutral - ang_ctrl
                                        right = self.speed_neutral + ang_ctrl
                                        left  = max(self.speed_min, min(self.speed_max, self._adjust_speed(left)))
                                        right = max(self.speed_min, min(self.speed_max, self._adjust_speed(right)))
                                        self.iface.write_wheel_command(left, right)
                                        time.sleep(0.03)
                                        
                                        # This is the line from the traceback
                                        pose = self.iface.read_pos()
                                        if pose is None:
                                            _dbg("Lost pose during pre-align. Aborting alignment.", self._tick)
                                            break # Exit the alignment loop
                                        _x, _y, theta = pose
                                        
                                        if (stop_event and stop_event.is_set()) or (time.time() - t_align_start) > ALIGN_TIMEOUT_S:
                                            break
                                # --- FIX END ---
                                self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)
                                pre = self._grip_state["pre"]; cen = self._grip_state["cen"]
                                self._seg_path = _densify_segment(pre, cen, step_px=self.linear_step_px)
                                self._seg_index = 0
                                self._seg_goal = (int(cen[0]), int(cen[1]))
                                self._grip_state["stage"] = "to_center"
                                continue

                            elif self._grip_state["stage"] == "to_center":
                                if isinstance(action, str) and action in ("open", "close"):
                                    _dbg(f"GRIPPER: {action} at (grip {goal_x},{goal_y})", self._tick)
                                    self._do_gripper_action(action)
                                    if self.gripper_action_pause_s > 0:
                                        self._pause_at_checkpoint(self.gripper_action_pause_s, stop_event=stop_event)
                                self._maybe_log_trace(goal_x, goal_y)
                                self._maybe_log_trace(self._grip_state["cen"][0], self._grip_state["cen"][1])
                                if isinstance(action, str) and action == "open" and self.post_drop_retreat_px > 0.0:
                                    th_hold = float(self._grip_state["theta"])
                                    cx, cy = self._grip_state["cen"]
                                    self._reverse_retreat(self.post_drop_retreat_px, th_hold, (cx, cy), stop_event=stop_event)
                                self._grip_state = None
                                idx += 1
                                self._seg_path = None; self._seg_index = 0; self._seg_goal = None
                                self.integral_ang = 0.0; self.prev_ang_err = 0.0
                                continue

                            elif self._grip_state["stage"] == "retreat":
                                self._grip_state = None
                                idx += 1
                                self._seg_path = None; self._seg_index = 0; self._seg_goal = None
                                self.integral_ang = 0.0; self.prev_ang_err = 0.0
                                continue

                        try: ang_tol_rad = math.radians(self.ang_tolerance_deg)
                        except Exception: ang_tol_rad = math.radians(18.0)

                        # --- FIX START ---
                        initial_pose = self.iface.read_pos()
                        if initial_pose is None:
                             _dbg("Cannot get initial pose for final align. Skipping alignment.", self._tick)
                        else:
                            x, y, theta = initial_pose
                            self._maybe_log_trace(x, y)

                            try: gth = float(goal_theta)
                            except Exception: gth = 0.0

                            t_align_start = time.time()
                            ALIGN_TIMEOUT_S = 3.0

                            while True:
                                ang_err_goal = self.normalize(gth - theta)
                                if abs(ang_err_goal) <= ang_tol_rad:
                                    break
                                now2 = time.time()
                                dt2 = max(1e-3, now2 - self.prev_time)
                                self.prev_time = now2
                                self.integral_ang += ang_err_goal * dt2
                                I_LIM = 0.4
                                if self.integral_ang > I_LIM:  self.integral_ang = I_LIM
                                if self.integral_ang < -I_LIM: self.integral_ang = -I_LIM
                                deriv_ang = (ang_err_goal - self.prev_ang_err) / dt2
                                self.prev_ang_err = ang_err_goal
                                ang_ctrl = (self.Kp_ang * ang_err_goal) + (self.Ki_ang * self.integral_ang) + (self.Kd_ang * deriv_ang)
                                left  = self.speed_neutral - ang_ctrl
                                right = self.speed_neutral + ang_ctrl
                                left  = max(self.speed_min, min(self.speed_max, self._adjust_speed(left)))
                                right = max(self.speed_min, min(self.speed_max, self._adjust_speed(right)))
                                self.iface.write_wheel_command(left, right)
                                time.sleep(0.03)

                                pose = self.iface.read_pos()
                                if pose is None:
                                    _dbg("Lost pose during final align. Aborting alignment.", self._tick)
                                    break
                                _x, _y, theta = pose

                                if (stop_event and stop_event.is_set()) or (time.time() - t_align_start) > ALIGN_TIMEOUT_S:
                                    break
                        # --- FIX END ---
                        self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)

                        if isinstance(action, str) and action in ("open", "close"):
                            _dbg(f"GRIPPER: {action} at ({goal_x},{goal_y})", self._tick)
                            self._do_gripper_action(action)
                            if self.gripper_action_pause_s > 0:
                                self._pause_at_checkpoint(self.gripper_action_pause_s, stop_event=stop_event)
                        else:
                            try: delay_s = float(action) / 1000.0
                            except Exception: delay_s = 0.0
                            self._pause_at_checkpoint(delay_s, stop_event=stop_event)

                        self._maybe_log_trace(goal_x, goal_y)
                        idx += 1
                        self._seg_path = None; self._seg_index = 0; self._seg_goal = None
                        self.integral_ang = 0.0; self.prev_ang_err = 0.0
                        continue
                    else:
                        _dbg("LAST_WP_BUT_GOAL_FAR: invalidating path to trigger replan.", self._tick)
                        self._seg_path = None
                        self._seg_index = 0
                        continue
                continue

            heading = math.atan2(ty - y, tx - x)
            ang_err = self.normalize(heading - theta)
            self.integral_ang += ang_err * dt
            I_LIM = 0.4
            if self.integral_ang > I_LIM:  self.integral_ang = I_LIM
            if self.integral_ang < -I_LIM: self.integral_ang = -I_LIM
            deriv_ang = (ang_err - self.prev_ang_err) / dt
            self.prev_ang_err = ang_err
            ang_ctrl = (self.Kp_ang * ang_err) + (self.Ki_ang * self.integral_ang) + (self.Kd_ang * deriv_ang)
            lin_ctrl = max(-self.max_lin, min(self.max_lin, self.Kp_dist * dist_to_wp))
            dtc = self.cache.clearance_dt
            if dtc is not None:
                iy = int(np.clip(round(y), 0, dtc.shape[0]-1))
                ix = int(np.clip(round(x), 0, dtc.shape[1]-1))
                clr_here = float(dtc[iy, ix])
                near_scale = np.clip(clr_here / max(1.0, self.clearance_req_px), 0.4, 1.0)
                lin_ctrl *= near_scale

            forward_gain = max(0.0, math.cos(ang_err)) ** 1.5
            if abs(self._deg(ang_err)) <= self.ang_tolerance_deg:
                forward_gain = 1.0

            left  = self.speed_neutral + (lin_ctrl * forward_gain) - ang_ctrl
            right = self.speed_neutral + (lin_ctrl * forward_gain) + ang_ctrl

            if DEBUG:
                direction = self._direction_label(lin_ctrl * forward_gain, ang_ctrl)
                _dbg(f"dir={direction}", self._tick)
                _dbg(f"idx={idx+1} pos=({x:.1f},{y:.1f}) wp=({tx:.1f},{ty:.1f}) "
                     f"dist={dist_to_wp:.2f}px ang={self._deg(ang_err):.1f}deg "
                     f"fg={forward_gain:.2f} pre=({left:.1f},{right:.1f})", self._tick)

            left  = max(self.speed_min, min(self.speed_max, self._adjust_speed(left)))
            right = max(self.speed_min, min(self.speed_max, self._adjust_speed(right)))

            _dbg(f"cmd=({left:.1f},{right:.1f})", self._tick)
            self.iface.write_wheel_command(left, right)
            self.iface.log_error(dist_to_wp, ang_err)

        self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)
        print("Controller finished.")
        return True

# ---------- Convenience ----------
def run_controller(
    target_file, pose_file, command_file, error_file,
    Kp_dist=0.18, Kp_ang=3.0, Ki_ang=0.02, Kd_ang=0.9,
    dist_tolerance=18.0, ang_tolerance=18.0, final_distance_tol=10.0,
    data_folder="Data", spacing_px=40, plan_spacing_boost_px=8,
    linear_step_px=12, simplify_dist_px=10,
    offset_px=100, replan_check_period_s=0.10, replan_wait_backoff_s=0.20,
    speed_min=70, speed_max=110, speed_neutral=90, max_lin=30,
    nearest_free_radius=50, linecheck_margin_px=1,
    clearance_boost_px=0,
    stop_event=None,
    robot_id=None, robot_padding=0,
    gripper_action_pause_s=0.4,
    trace_json_path: str = str(Path("Data") / "trace_paths.json"),
    trace_stride_px: float = 6.0,
    trace_flush_every: int = 1,
    gripper_offset_px: float = 30.0,
    gripper_pre_approach_extra_px: float = 50.0,
    post_drop_retreat_px: float = 20.0,
):
    iface = FileInterface(target_file, pose_file, command_file, error_file, robot_id=robot_id)
    controller = PIDController(
        iface,
        Kp_dist=Kp_dist, Kp_ang=Kp_ang, Ki_ang=Ki_ang, Kd_ang=Kd_ang,
        dist_tolerance=dist_tolerance, ang_tolerance=ang_tolerance, final_distance_tol=final_distance_tol,
        data_folder=data_folder, spacing_px=spacing_px, plan_spacing_boost_px=plan_spacing_boost_px,
        linear_step_px=linear_step_px, simplify_dist_px=simplify_dist_px, offset_px=offset_px,
        replan_check_period_s=replan_check_period_s, replan_wait_backoff_s=replan_wait_backoff_s,
        speed_min=speed_min, speed_max=speed_max, speed_neutral=speed_neutral, max_lin=max_lin,
        nearest_free_radius=nearest_free_radius, linecheck_margin_px=linecheck_margin_px,
        clearance_boost_px=clearance_boost_px,
        robot_id=robot_id, robot_padding=robot_padding,
        gripper_action_pause_s=gripper_action_pause_s,
        trace_json_path=trace_json_path,
        trace_stride_px=trace_stride_px,
        trace_flush_every=trace_flush_every,
        astar_dump_path=str(Path("Data") / "astar_segments.json"),
        clear_astar_dump_on_start=True,
        gripper_offset_px=gripper_offset_px,
        gripper_pre_approach_extra_px=gripper_pre_approach_extra_px,
        post_drop_retreat_px=post_drop_retreat_px,
    )
    ret = controller.run(stop_event=stop_event)
    return ret

def exec_bot(robot_id=782, robot_padding=30):
    target_file  = str(Path("Targets") / "paths.json")
    pose_file    = str(Path("Data") / "robot_pos.txt")
    command_file = str(Path("Data") / "command.json")
    error_file   = str(Path("Data") / "error.txt")

    if not _robot_id_exists(pose_file, robot_id):
        return "Selected ID doesn't exist"
    if not _path_for_robot_exists(target_file, robot_id):
        return "Path for specific robot doesn't exist. Generate path first"

    run_controller(
        target_file, pose_file, command_file, error_file,
        robot_id=robot_id, robot_padding=robot_padding,
        trace_json_path=str(Path("Data") / "trace_paths.json"),
        trace_stride_px=6.0,
        trace_flush_every=1,
    )
    trace_path = str(Path("Data") / "trace_paths.json")
    _clear_trace_file(trace_path)
    return "Done executing"

def exec_bot_with_thread(stop_event, robot_id=782, robot_padding=30):
    target_file  = str(Path("Targets") / "paths.json")
    pose_file    = str(Path("Data") / "robot_pos.txt")
    command_file = str(Path("Data") / "command.json")
    error_file   = str(Path("Data") / "error.txt")

    if not _robot_id_exists(pose_file, robot_id):
        return "Selected ID doesn't exist"
    if not _path_for_robot_exists(target_file, robot_id):
        return "Path for specific robot doesn't exist. Generate path first"

    run_controller(
        target_file, pose_file, command_file, error_file,
        stop_event=stop_event,
        robot_id=robot_id, robot_padding=robot_padding,
        trace_json_path=str(Path("Data") / "trace_paths.json"),
        trace_stride_px=6.0,
        trace_flush_every=1,
    )
    trace_path = str(Path("Data") / "trace_paths.json")
    _clear_trace_file(trace_path)
    return "Done executing"

def exec_robot_create_thread(robot_id: int, robot_padding: int = 30):
    """
    Spawns a background controller thread for the given robot_id.
    This thread writes to command.json. A separate process (e.g., motion.py)
    is expected to read this file and command the robot hardware.
    Returns a short status string.
    """
    robot_id = int(robot_id)
    robot_padding = int(robot_padding)
    target_file  = str(Path("Targets") / "paths.json")
    pose_file    = str(Path("Data") / "robot_pos.txt")
    command_file = str(Path("Data") / "command.json")
    error_file   = str(Path("Data") / "error.txt")

    if not _robot_id_exists(pose_file, robot_id):
        return "Selected ID doesn't exist"
    if not _path_for_robot_exists(target_file, robot_id):
        return "Path for specific robot doesn't exist. Generate path first"

    with _robot_threads_lock:
        info = _robot_threads.get(robot_id)
        if info and info.get("ctrl_thread") and info["ctrl_thread"].is_alive():
            return f"Thread already running for robot {robot_id}"
        stop_event = threading.Event()
        def _runner():
            try:
                run_controller(
                    target_file, pose_file, command_file, error_file,
                    stop_event=stop_event,
                    robot_id=robot_id, robot_padding=robot_padding,
                    trace_json_path=str(Path("Data") / "trace_paths.json"),
                    trace_stride_px=6.0,
                    trace_flush_every=1,
                )
                trace_path = str(Path("Data") / "trace_paths.json")
                _clear_trace_file(trace_path)
            finally:
                with _robot_threads_lock:
                    info = _robot_threads.get(robot_id)
                    if info:
                        info["ctrl_done"] = True
        ctrl_thread = threading.Thread(target=_runner, name=f"robot-{robot_id}", daemon=True)
        ctrl_thread.start()
        _robot_threads[robot_id] = {
            "ctrl_thread": ctrl_thread,
            "stop_event": stop_event,
            "started_at": time.time(),
            "ctrl_done": False,
        }
    return f"Started controller thread for robot {robot_id}"

def stop_robot_thread(robot_id: int, join_timeout: float = 5.0):
    """
    Signal the running controller thread for robot_id to stop and wait briefly.
    Returns a short status string.
    """
    robot_id = int(robot_id)
    with _robot_threads_lock:
        info = _robot_threads.get(robot_id)
        if not info:
            return f"No running thread for robot {robot_id}"
        ctrl_t = info.get("ctrl_thread")
        ev     = info["stop_event"]
        ev.set()
    deadline = time.time() + max(0.0, float(join_timeout))
    if ctrl_t is not None:
        ctrl_t.join(timeout=max(0.0, deadline - time.time()))
    with _robot_threads_lock:
        info = _robot_threads.get(robot_id)
        still_alive = ctrl_t and ctrl_t.is_alive()
        if not still_alive:
            _robot_threads.pop(robot_id, None)
    if still_alive:
        return f"Stop requested; robot {robot_id} is still stopping"
    trace_path = str(Path("Data") / "trace_paths.json")
    _clear_trace_file(trace_path)
    return f"Stopped thread for robot {robot_id}"

if __name__ == "__main__":
    print(exec_bot(robot_id=2, robot_padding=30))