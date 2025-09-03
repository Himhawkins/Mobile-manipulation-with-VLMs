#!/usr/bin/env python3
"""
Fast Robot Controller with Online Replanning + ESCAPE + Clearance Guard

Now with:
- A* PLANS on an extra-inflated mask (spacing_px + plan_spacing_boost_px)
- Runtime checks still use spacing_px
- Minimum-clearance check applied to BOTH straight shortcuts and A* polylines
- FIX: no runaway waypoint index — replan when polyline is consumed, and
       clamp/invalidate path if last waypoint reached but goal not yet reached.

Tune at bottom:
    spacing_px, plan_spacing_boost_px, clearance_boost_px, Kp/*, max_lin, ...
"""

import math
import os
import time
import datetime
from pathlib import Path

import cv2
import numpy as np

from astar import PathPlanner

# ---------- Debug ----------
DEBUG = True
DEBUG_EVERY = 5
def _dbg(msg: str, tick=None):
    if not DEBUG: return
    if tick is not None and tick % DEBUG_EVERY != 0: return
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[DBG {ts}] {msg}")

# ---------- Files ----------
def _ensure_txt(path, default_lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w') as f:
            for line in default_lines:
                f.write(f"{line}\n")
        _dbg(f"Initialized file: {path}")

class FileInterface:
    def __init__(self, target_file, pose_file, command_file, error_file):
        self.target_file = target_file
        self.pose_file = pose_file
        self.command_file = command_file
        self.error_file = error_file
        self._initialize_files()
    def _initialize_files(self):
        _ensure_txt(self.target_file, ["0,0,0"])
        _ensure_txt(self.pose_file,    ["0,0,0.0"])
        _ensure_txt(self.command_file, ["90,90"])
        _ensure_txt(self.error_file,   ["0.0,0.0"])
    def _load_lines(self, path):
        try:
            with open(path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"[WARN] Error reading {path}: {e}")
            return []
    def read_targets(self):
        lines = self._load_lines(self.target_file)
        out = []
        for line in lines:
            try:
                parts = [p.strip() for p in line.split(',') if p.strip() != ""]
                if len(parts) < 2: continue
                x = float(parts[0]); y = float(parts[1])
                delay_ms = float(parts[2]) if len(parts) >= 3 else 0.0
                out.append((x, y, delay_ms))
            except Exception as e:
                print(f"[WARN] Skipping malformed target '{line}': {e}")
        return out
    def read_pos(self):
        lines = self._load_lines(self.pose_file)
        if not lines: return 0.0, 0.0, 0.0
        try:
            x_str, y_str, th_str = lines[-1].split(',')
            return float(x_str), float(y_str), float(th_str)
        except Exception as e:
            print(f"[WARN] Malformed pose '{lines[-1]}': {e}")
            return 0.0, 0.0, 0.0
    def write_command(self, left_speed, right_speed):
        try:
            with open(self.command_file, 'w') as f:
                f.write(f"{int(round(left_speed))},{int(round(right_speed))}\n")
        except IOError as e:
            print(f"[ERR] Saving command failed: {e}")
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
        if m is not None and _bresenham_blocked(m, p0, p1): return True
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
    try: return os.path.getmtime(path)
    except Exception: return 0.0

class PlannerCache:
    """
    Caches masks and rebuilds only when inputs change.
    - _mask_raw: true obstacles (no inflation)
    - _mask_dilated: inflation with spacing_px (runtime checks)
    - _mask_plan: inflation with spacing_px + plan_spacing_boost_px (A* planning)
    - _mask_for_line: _mask_dilated then lightly eroded for line checks
    - _dt_clearance_raw: distance to true obstacles
    """
    def __init__(self, data_folder, spacing_px=26, plan_spacing_boost_px=8, line_margin_px=1):
        self.data_folder = data_folder
        self.spacing_px = int(spacing_px)
        self.plan_spacing_boost_px = int(plan_spacing_boost_px)
        self.line_margin_px = int(line_margin_px)
        self.frame_path = os.path.join(data_folder, "frame_img.png")
        self.static_path = os.path.join(data_folder, "obstacles.txt")
        self.real_path   = os.path.join(data_folder, "realtime_obstacles.txt")
        self.arena_path  = os.path.join(data_folder, "arena_corners.txt")

        self._mtimes = {"frame":0, "static":0, "real":0, "arena":0}
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
        if m != self._mtimes["frame"]:
            self._mtimes["frame"] = m; changed = True
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
        return changed

    def rebuild_if_needed(self):
        files_changed = self._read_if_changed()
        if self._planner is not None and not files_changed:
            return
        self._ensure_size()

        obstacles = [{"corners": p} for p in (self._static_polys + self._realtime_polys)]
        self._planner = PathPlanner(obstacles, (self._H, self._W), arena_corners=self._arena)

        # RAW mask from planner
        self._mask_raw = self._planner.mask.copy()

        # spacing mask (runtime checks)
        ks = 2 * int(self.spacing_px) + 1
        if self._kernel_dilate_spacing is None or self._kernel_dilate_spacing.shape[0] != ks:
            self._kernel_dilate_spacing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        self._mask_dilated = cv2.dilate(self._mask_raw, self._kernel_dilate_spacing)

        # planning mask (extra inflation for A*)
        kp = 2 * int(self.spacing_px + self.plan_spacing_boost_px) + 1
        if self._kernel_dilate_plan is None or self._kernel_dilate_plan.shape[0] != kp:
            self._kernel_dilate_plan = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kp, kp))
        self._mask_plan = cv2.dilate(self._mask_raw, self._kernel_dilate_plan)

        # line mask = spacing mask then a light erosion
        if self.line_margin_px > 0:
            km = 2 * int(self.line_margin_px) + 1
            if self._kernel_erode is None or self._kernel_erode.shape[0] != km:
                self._kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (km, km))
            self._mask_for_line = cv2.erode(self._mask_dilated, self._kernel_erode)
        else:
            self._mask_for_line = self._mask_dilated

        # Distance to nearest TRUE obstacle (for clearance tests)
        free_u8 = np.where(self._mask_raw == 0, 255, 0).astype(np.uint8)
        self._dt_clearance_raw = cv2.distanceTransform(free_u8, cv2.DIST_L2, 3)

        # IMPORTANT: make A* use the PLANNING mask
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
    else:
        line2 = inflated2
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
    planner   = cache.planner
    mask_raw  = cache.mask_raw
    mask_dil  = cache.mask_dilated
    mask_line = cache.mask_for_line
    dt        = cache.clearance_dt

    sx, sy = map(int, start_xy)
    gx, gy = map(int, goal_xy)

    # snap against spacing mask (not planning mask)
    if mask_dil[sy, sx] != 0:
        snap = _nearest_free(mask_dil, sx, sy, r_max=nearest_free_radius)
        if snap is None:
            _dbg(f"START_ENCLOSED within r={nearest_free_radius}px.")
            return None
        _dbg(f"START_SNAP: {(sx, sy)} -> {snap}")
        sx, sy = snap

    if mask_dil[gy, gx] != 0:
        snap = _nearest_free(mask_dil, gx, gy, r_max=nearest_free_radius)
        if snap is None:
            _dbg(f"GOAL_ENCLOSED within r={nearest_free_radius}px.")
            return None
        _dbg(f"GOAL_SNAP: {(gx, gy)} -> {snap}")
        gx, gy = snap

    # DIRECT shortcut must meet clearance
    if _line_ok([mask_raw, mask_line], dt, (sx, sy), (gx, gy), clearance_req_px):
        return _densify_segment((sx, sy), (gx, gy), step_px=step_px)

    # A* on PLANNING mask (already set in planner)
    path = planner.find_obstacle_aware_path((sx, sy), (gx, gy), simplify_dist)
    if path:
        poly = _densify_polyline(path, step_px=step_px)
        if _polyline_ok(dt, poly, clearance_req_px):
            return poly
        # clearance too small → try offsets around goal

    for dx, dy in [(0, -offset_px), (offset_px, 0), (0, offset_px), (-offset_px, 0)]:
        nx, ny = gx + dx, gy + dy
        H, W = mask_dil.shape[:2]
        if not (0 <= nx < W and 0 <= ny < H): continue
        if mask_dil[ny, nx]: continue
        if _line_ok([mask_raw, mask_line], dt, (sx, sy), (nx, ny), clearance_req_px):
            return _densify_segment((sx, sy), (nx, ny), step_px=step_px)
        p = planner.find_obstacle_aware_path((sx, sy), (nx, ny), simplify_dist)
        if p:
            poly = _densify_polyline(p, step_px=step_px)
            if _polyline_ok(dt, poly, clearance_req_px):
                return poly

    return None

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
        clearance_boost_px=0,   # require >= spacing_px + this along lines/polylines
    ):
        self.iface = iface
        # PID
        self.Kp_dist = Kp_dist; self.Kp_ang = Kp_ang; self.Ki_ang = Ki_ang; self.Kd_ang = Kd_ang
        self.dist_tolerance = float(dist_tolerance)
        self.ang_tolerance_deg = float(ang_tolerance)
        self.final_distance_tol = float(final_distance_tol)
        # Motion limits
        self.speed_min = int(speed_min); self.speed_max = int(speed_max)
        self.speed_neutral = int(speed_neutral); self.max_lin = float(max_lin)
        # Planner cache (planning uses extra inflation)
        self.cache = PlannerCache(
            data_folder, spacing_px=spacing_px,
            plan_spacing_boost_px=plan_spacing_boost_px,
            line_margin_px=linecheck_margin_px
        )
        self.linear_step_px = int(linear_step_px)
        self.simplify_dist_px = int(simplify_dist_px); self.offset_px = int(offset_px)
        self.nearest_free_radius = int(nearest_free_radius)
        # Timing
        self.replan_check_period_s = float(replan_check_period_s)
        self.replan_wait_backoff_s = float(replan_wait_backoff_s)

        # Clearance requirement (relative to true obstacles)
        self.clearance_req_px = int(self.cache.spacing_px + int(clearance_boost_px))

        # State
        self.integral_ang = 0.0; self.prev_ang_err = 0.0
        self.prev_time = time.time()
        self._seg_path = None; self._seg_index = 0; self._seg_goal = None
        self._tick = 0
        self._last_pose = None; self._pose_still_cnt = 0
        # ESCAPE
        self._escape_active = False
        self._escape_line_mask = None

        self.cache.rebuild_if_needed()

    @staticmethod
    def normalize(angle): return math.atan2(math.sin(angle), math.cos(angle))
    @staticmethod
    def _deg(rad): return math.degrees(rad)

    def _adjust_speed(self, s):
        if 89.75 <= s <= 90.25: return 90.0
        if 90.25 < s <= 95: return s + 1.0
        if 85 <= s < 89.75: return s - 1.0
        return s

    def _pause_at_checkpoint(self, delay_s, stop_event=None):
        if delay_s <= 0: return
        self.iface.write_command(self.speed_neutral, self.speed_neutral)
        _dbg(f"REACHED target; pausing {delay_s:.2f}s")
        t0 = time.time()
        while (time.time() - t0) < delay_s:
            if stop_event and stop_event.is_set(): break
            time.sleep(0.03)

    # ---- ESCAPE plan ----
    def _try_escape_plan(self, start_xy):
        if self.cache.planner is None: return None
        sx, sy = map(int, start_xy)
        mask_raw = self.cache.mask_raw
        mask_dil = self.cache.mask_dilated
        if mask_raw is None or mask_dil is None: return None
        H, W = mask_dil.shape[:2]
        sx = int(np.clip(sx, 0, W-1)); sy = int(np.clip(sy, 0, H-1))
        if mask_dil[sy, sx] == 0: return None  # not inside spacing

        km = self.cache.kernel_erode
        bubble_r = max(8, self.cache.spacing_px + 4)
        inflated2, line2 = _make_escape_masks(mask_raw, mask_dil, sx, sy, r=bubble_r, kernel_erode=km)

        esc_wp = _find_escape_waypoint(
            line2, inflated2, mask_raw, sx, sy,
            r_samples=(self.cache.spacing_px, self.cache.spacing_px + 10, self.cache.spacing_px + 20),
            dirs=36
        )
        if esc_wp is None:
            _dbg("ESCAPE: no boundary point found.")
            return None

        dtc = self.cache.clearance_dt
        if _min_clearance_along_line(dtc, (sx, sy), esc_wp) < max(4, self.clearance_req_px * 0.5):
            _dbg("ESCAPE: candidate too close to obstacle — retry.")
            return None

        _dbg(f"ESCAPE: temporary waypoint {esc_wp} created.")
        self._escape_active = True
        self._escape_line_mask = line2
        return _densify_segment((sx, sy), esc_wp, step_px=self.linear_step_px)

    # ---- Segment path ensure ----
    def _ensure_segment_path(self, start_xy, final_goal_xy):
        next_check = getattr(self, "_next_check", 0.0)
        now = time.time()
        if now >= next_check:
            self.cache.rebuild_if_needed()
            self._next_check = now + self.replan_check_period_s

        if self.cache.planner is None or self.cache.mask_dilated is None:
            return False

        sx, sy = int(start_xy[0]), int(start_xy[1])
        goal_tuple = (int(final_goal_xy[0]), int(final_goal_xy[1]))

        # inside spacing → ESCAPE first
        if self.cache.mask_dilated[sy, sx] != 0:
            if self._escape_active and self._seg_path is not None and self._seg_index < len(self._seg_path):
                cur = (sx, sy)
                nxt = (int(self._seg_path[self._seg_index][0]), int(self._seg_path[self._seg_index][1]))
                if self._escape_line_mask is None or _bresenham_blocked(self._escape_line_mask, cur, nxt):
                    _dbg(f"ESCAPE_LINE_BLOCKED: {cur}->{nxt} — recomputing escape.", self._tick)
                    self._seg_path = None
            if self._seg_path is None:
                path = self._try_escape_plan((sx, sy))
                if not path:
                    _dbg("ESCAPE: could not synthesize a safe exit path.", self._tick)
                    return False
                self._seg_path = path
                self._seg_index = 0
                self._seg_goal = goal_tuple
                _dbg(f"NEW_PLAN(ESCAPE): len={len(path)}", self._tick)
            return True

        # --- Outside inflated area: normal logic ---
        need_new = False
        # REPLAN if: no path, goal changed, ESCAPE turning off, OR we've consumed the polyline
        if (self._seg_path is None or
            self._seg_goal != goal_tuple or
            self._escape_active or
            (self._seg_path is not None and self._seg_index >= len(self._seg_path))):
            if self._seg_path is not None and self._seg_index >= len(self._seg_path):
                _dbg("END_OF_POLYLINE: consumed path but goal not reached — replanning.", self._tick)
            if self._escape_active:
                _dbg("ESCAPE_DONE: outside inflated region — switching to normal planning.", self._tick)
            self._escape_active = False
            self._escape_line_mask = None
            need_new = True
        else:
            if self._seg_index < len(self._seg_path):
                cur = (sx, sy)
                nxt = (int(self._seg_path[self._seg_index][0]), int(self._seg_path[self._seg_index][1]))
                if self.cache.mask_dilated[nxt[1], nxt[0]] != 0:
                    _dbg(f"WAYPOINT_BLOCKED: {nxt} inside inflated mask — replanning.", self._tick)
                    need_new = True
                elif _line_blocked_multi([self.cache.mask_raw, self.cache.mask_for_line], cur, nxt):
                    _dbg(f"LINE_BLOCKED: {cur}->{nxt} intersects obstacle — replanning.", self._tick)
                    need_new = True

        if need_new:
            path = _plan_segment_cached(
                self.cache, (sx, sy), goal_tuple,
                simplify_dist=self.simplify_dist_px, step_px=self.linear_step_px,
                offset_px=self.offset_px, nearest_free_radius=self.nearest_free_radius,
                clearance_req_px=self.clearance_req_px
            )
            if not path:
                _dbg("NO_PLAN: planning failed for current start→goal.", self._tick)
                return False
            self._seg_path = path
            self._seg_index = 0
            self._seg_goal = goal_tuple
            _dbg(f"NEW_PLAN: len={len(path)} to goal={goal_tuple}", self._tick)
        return True

    # ---- Main loop ----
    def run(self, stop_event=None):
        targets = self.iface.read_targets()
        if not targets:
            print("No targets found in Targets/path.txt")
            return

        idx = 0
        while idx < len(targets) and not (stop_event and stop_event.is_set()):
            self._tick += 1
            now = time.time()
            dt = max(1e-3, now - self.prev_time); self.prev_time = now

            x, y, theta = self.iface.read_pos()
            cur_pose = (round(x,1), round(y,1))
            if self._last_pose == cur_pose:
                self._pose_still_cnt += 1
                if self._pose_still_cnt % (20*DEBUG_EVERY) == 0:
                    _dbg("POSE_STALE: robot_pos.txt not changing — check upstream pose publisher.")
            else:
                self._pose_still_cnt = 0
            self._last_pose = cur_pose

            goal_x, goal_y, delay_ms = targets[idx]

            if not self._ensure_segment_path((x, y), (goal_x, goal_y)):
                _dbg(f"NO_PLAN: waiting for feasible path to target {idx+1} ({goal_x:.1f},{goal_y:.1f})...", self._tick)
                self.iface.write_command(self.speed_neutral, self.speed_neutral)
                while not (stop_event and stop_event.is_set()):
                    time.sleep(self.replan_wait_backoff_s)
                    x, y, theta = self.iface.read_pos()
                    if self._ensure_segment_path((x, y), (goal_x, goal_y)):
                        _dbg("PLAN_AVAILABLE: resuming motion.", self._tick)
                        break
                if stop_event and stop_event.is_set(): break
                continue

            # Guard: if seg_path somehow empty, force a replan next tick
            if not self._seg_path:
                self._seg_path = None
                self._seg_index = 0
                continue

            cur_wp = self._seg_path[min(self._seg_index, len(self._seg_path)-1)]
            tx, ty = float(cur_wp[0]), float(cur_wp[1])

            # advance
            dist_to_wp = math.hypot(tx - x, ty - y)
            if dist_to_wp < self.dist_tolerance:
                # increment but clamp to avoid runaway index
                self._seg_index = min(self._seg_index + 1, len(self._seg_path))
                _dbg(f"ADVANCE: waypoint {self._seg_index}/{len(self._seg_path)}", self._tick)

                if self._escape_active and self._seg_index >= len(self._seg_path):
                    self._escape_active = False
                    self._escape_line_mask = None
                    _dbg("ESCAPE_DONE: back to normal planning.", self._tick)

                if self._seg_index >= len(self._seg_path):
                    # reached end of polyline: check goal proximity
                    if math.hypot(goal_x - x, goal_y - y) < self.final_distance_tol:
                        self.iface.write_command(self.speed_neutral, self.speed_neutral)
                        self._pause_at_checkpoint(delay_ms / 1000.0, stop_event=stop_event)
                        idx += 1
                        self._seg_path = None; self._seg_index = 0; self._seg_goal = None
                        self.integral_ang = 0.0; self.prev_ang_err = 0.0
                        continue
                    else:
                        # last WP reached but goal still far (likely blocked) → replan
                        _dbg("LAST_WP_BUT_GOAL_FAR: invalidating path to trigger replan.", self._tick)
                        self._seg_path = None
                        self._seg_index = 0
                        continue
                continue

            # PID
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

            # Proximity slowdown
            dtc = self.cache.clearance_dt
            if dtc is not None:
                iy = int(np.clip(round(y), 0, dtc.shape[0]-1))
                ix = int(np.clip(round(x), 0, dtc.shape[1]-1))
                clr_here = float(dtc[iy, ix])
                near_scale = np.clip(clr_here / max(1.0, self.clearance_req_px), 0.4, 1.0)
                lin_ctrl *= near_scale
                _dbg(f"CLR={clr_here:.1f}px scale={near_scale:.2f}", self._tick)

            forward_gain = max(0.0, math.cos(ang_err)) ** 1.5
            if abs(self._deg(ang_err)) <= self.ang_tolerance_deg:
                forward_gain = 1.0

            left  = self.speed_neutral + (lin_ctrl * forward_gain) - ang_ctrl
            right = self.speed_neutral + (lin_ctrl * forward_gain) + ang_ctrl

            if DEBUG:
                _dbg(f"idx={idx+1} pos=({x:.1f},{y:.1f}) wp=({tx:.1f},{ty:.1f}) "
                     f"dist={dist_to_wp:.2f}px ang={self._deg(ang_err):.1f}deg "
                     f"fg={forward_gain:.2f} pre=({left:.1f},{right:.1f})", self._tick)

            left  = max(self.speed_min, min(self.speed_max, self._adjust_speed(left)))
            right = max(self.speed_min, min(self.speed_max, self._adjust_speed(right)))

            _dbg(f"cmd=({left:.1f},{right:.1f})", self._tick)
            self.iface.write_command(left, right)
            self.iface.log_error(dist_to_wp, ang_err)

        self.iface.write_command(self.speed_neutral, self.speed_neutral)
        print("Controller finished.")

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
    stop_event=None
):
    iface = FileInterface(target_file, pose_file, command_file, error_file)
    controller = PIDController(
        iface,
        Kp_dist=Kp_dist, Kp_ang=Kp_ang, Ki_ang=Ki_ang, Kd_ang=Kd_ang,
        dist_tolerance=dist_tolerance, ang_tolerance=ang_tolerance, final_distance_tol=final_distance_tol,
        data_folder=data_folder, spacing_px=spacing_px, plan_spacing_boost_px=plan_spacing_boost_px,
        linear_step_px=linear_step_px, simplify_dist_px=simplify_dist_px, offset_px=offset_px,
        replan_check_period_s=replan_check_period_s, replan_wait_backoff_s=replan_wait_backoff_s,
        speed_min=speed_min, speed_max=speed_max, speed_neutral=speed_neutral, max_lin=max_lin,
        nearest_free_radius=nearest_free_radius, linecheck_margin_px=linecheck_margin_px,
        clearance_boost_px=clearance_boost_px
    )
    controller.run(stop_event=stop_event)

def exec_bot():
    target_file  = str(Path("Targets") / "path.txt")
    pose_file    = str(Path("Data") / "robot_pos.txt")
    command_file = str(Path("Data") / "command.txt")
    error_file   = str(Path("Data") / "error.txt")
    run_controller(target_file, pose_file, command_file, error_file)

def exec_bot_with_thread(stop_event):
    target_file  = str(Path("Targets") / "path.txt")
    pose_file    = str(Path("Data") / "robot_pos.txt")
    command_file = str(Path("Data") / "command.txt")
    error_file   = str(Path("Data") / "error.txt")
    run_controller(target_file, pose_file, command_file, error_file, stop_event=stop_event)

if __name__ == "__main__":
    exec_bot()
