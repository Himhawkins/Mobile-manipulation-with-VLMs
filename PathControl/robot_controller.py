#!/usr/bin/env python3
# modules/robot_controller.py

import math
import time
from pathlib import Path
import cv2
import numpy as np

# Import from our new modules
from .file_utils import FileInterface, _astar_dump_clear, _astar_dump_append, _trace_append_point, _dbg
from .planner_cache import PlannerCache
from .geometry_utils import _line_blocked_multi, _min_clearance_along_line, _bresenham_blocked, _densify_segment, _line_ok, _polyline_ok, _densify_polyline

# ---------- ESCAPE LOGIC ----------
# These helpers are tightly coupled with the controller's state when the robot is stuck.
def _nearest_free(mask, x, y, r_max=50):
    """Find the nearest non-obstacle pixel in a spiral search."""
    H, W = mask.shape[:2]
    x, y = int(np.clip(x, 0, W-1)), int(np.clip(y, 0, H-1))
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
    """Create temporary masks for finding an escape waypoint."""
    inflated2 = dilated_mask.copy()
    H, W = inflated2.shape[:2]
    sx, sy = int(np.clip(sx, 0, W-1)), int(np.clip(sy, 0, H-1))
    bubble = np.zeros_like(inflated2, np.uint8)
    cv2.circle(bubble, (sx, sy), int(r), 255, -1)
    only_inflation = cv2.bitwise_and(bubble, cv2.bitwise_not((raw_mask > 0).astype(np.uint8)*255))
    inflated2[only_inflation > 0] = 0
    line2 = cv2.erode(inflated2, kernel_erode) if kernel_erode is not None else inflated2
    return inflated2, line2

def _find_escape_waypoint(line_mask, inflated_mask, raw_mask, sx, sy, r_samples, dirs=36):
    """Search for a reachable point on the boundary of the robot's inflation bubble."""
    H, W = inflated_mask.shape[:2]
    sx, sy = int(np.clip(sx, 0, W-1)), int(np.clip(sy, 0, H-1))
    for r in r_samples:
        for th in np.linspace(0, 2 * np.pi, dirs, endpoint=False):
            ex = int(round(sx + r * math.cos(th)))
            ey = int(round(sy + r * math.sin(th)))
            if not (0 <= ex < W and 0 <= ey < H): continue
            if inflated_mask[ey, ex] != 0: continue
            if _bresenham_blocked(line_mask, (sx, sy), (ex, ey)): continue
            if _bresenham_blocked(raw_mask, (sx, sy), (ex, ey)): continue
            return (ex, ey)
    return None


# ---------- PLANNING & GRIPPER HELPERS ----------
def _plan_segment_cached(cache: PlannerCache, start_xy, goal_xy, **kwargs):
    """Plans a path using the cached masks, with fallbacks for unreachable goals."""
    simplify_dist = kwargs.get('simplify_dist', 10)
    step_px = kwargs.get('step_px', 12)
    offset_px = kwargs.get('offset_px', 100)
    nearest_free_radius = kwargs.get('nearest_free_radius', 50)
    clearance_req_px = kwargs.get('clearance_req_px', 0)

    planner, mask_raw, mask_dil = cache.planner, cache.mask_raw, cache.mask_dilated
    mask_line, dt = cache.mask_for_line, cache.clearance_dt
    sx, sy = map(int, start_xy)
    gx, gy = map(int, goal_xy)

    # Snap start/goal to nearest free space if they are inside an obstacle
    if mask_dil[sy, sx] != 0:
        snap = _nearest_free(mask_dil, sx, sy, r_max=nearest_free_radius)
        if snap is None: return None
        sx, sy = snap
    if mask_dil[gy, gx] != 0:
        snap = _nearest_free(mask_dil, gx, gy, r_max=nearest_free_radius)
        if snap is None: return None
        gx, gy = snap

    # Try a direct line first
    if _line_ok([mask_raw, mask_line], dt, (sx, sy), (gx, gy), clearance_req_px):
        return _densify_segment((sx, sy), (gx, gy), step_px=step_px)

    # Use A* planner for a complex path
    path = planner.find_obstacle_aware_path((sx, sy), (gx, gy), simplify_dist)
    if path:
        poly = _densify_polyline(path, step_px=step_px)
        if _polyline_ok(dt, poly, clearance_req_px):
            return poly

    # If A* fails, try offset points around the goal
    for dx, dy in [(0, -offset_px), (offset_px, 0), (0, offset_px), (-offset_px, 0)]:
        nx, ny = gx + dx, gy + dy
        if not (0 <= nx < cache.size[1] and 0 <= ny < cache.size[0]) or mask_dil[ny, nx]: continue
        p = planner.find_obstacle_aware_path((sx, sy), (nx, ny), simplify_dist)
        if p:
            poly = _densify_polyline(p, step_px=step_px)
            if _polyline_ok(dt, poly, clearance_req_px):
                return poly
    return None

def _grip_pre_approach(gx: float, gy: float, gtheta: float, offset_px: float, extra_px: float):
    """Calculate pre-approach and final center points for a gripper action."""
    th = float(gtheta)
    # Final robot center for the gripper to be at the target
    center_x = gx - offset_px * math.cos(th)
    center_y = gy - offset_px * math.sin(th)
    # Pre-approach point, backed off from the final center point
    pre_x = gx - (offset_px + extra_px) * math.cos(th)
    pre_y = gy - (offset_px + extra_px) * math.sin(th)
    return (pre_x, pre_y), (center_x, center_y)


# ---------- MAIN CONTROLLER CLASS ----------
class PIDController:
    """The core logic for robot path following and control."""

    def __init__(self, iface: FileInterface, **kwargs):
        self.iface = iface

        # --- PID and Speed Parameters ---
        self.Kp_dist = float(kwargs.get('Kp_dist', 0.18))
        self.Kp_ang = float(kwargs.get('Kp_ang', 3.0))
        self.Ki_ang = float(kwargs.get('Ki_ang', 0.02))
        self.Kd_ang = float(kwargs.get('Kd_ang', 0.9))
        self.speed_min = int(kwargs.get('speed_min', 70))
        self.speed_max = int(kwargs.get('speed_max', 110))
        self.speed_neutral = int(kwargs.get('speed_neutral', 90))
        self.max_lin = float(kwargs.get('max_lin', 30))

        # --- Tolerances ---
        self.dist_tolerance = float(kwargs.get('dist_tolerance', 18.0))
        self.ang_tolerance_deg = float(kwargs.get('ang_tolerance', 18.0))
        self.final_distance_tol = float(kwargs.get('final_distance_tol', 10.0))

        # --- Planning and Caching ---
        robot_id = kwargs.get('robot_id')
        self.cache = PlannerCache(
            kwargs.get('data_folder', "Data"),
            spacing_px=kwargs.get('spacing_px', 40),
            plan_spacing_boost_px=kwargs.get('plan_spacing_boost_px', 8),
            line_margin_px=kwargs.get('linecheck_margin_px', 1),
            robot_id=robot_id,
            robot_padding=kwargs.get('robot_padding', 0)
        )
        self.linear_step_px = int(kwargs.get('linear_step_px', 12))
        self.simplify_dist_px = int(kwargs.get('simplify_dist_px', 10))
        self.offset_px = int(kwargs.get('offset_px', 100))
        self.nearest_free_radius = int(kwargs.get('nearest_free_radius', 50))
        self.replan_check_period_s = float(kwargs.get('replan_check_period_s', 0.10))
        self.replan_wait_backoff_s = float(kwargs.get('replan_wait_backoff_s', 0.05))
        self.clearance_req_px = int(self.cache.spacing_px + int(kwargs.get('clearance_boost_px', 0)))

        # --- Gripper Parameters ---
        self.gripper_action_pause_s = float(kwargs.get('gripper_action_pause_s', 0.4))
        self.gripper_offset_px = float(kwargs.get('gripper_offset_px', 30.0))
        self.gripper_pre_approach_extra_px = float(kwargs.get('gripper_pre_approach_extra_px', 50.0))
        self.post_drop_retreat_px = float(kwargs.get('post_drop_retreat_px', 20.0))

        # --- Tracing and Debugging ---
        self.trace_json_path = str(kwargs.get('trace_json_path', Path("Data") / "trace_paths.json"))
        self.trace_stride_px = float(kwargs.get('trace_stride_px', 2.0))
        self.astar_dump_path = str(kwargs.get('astar_dump_path', Path("Data") / "astar_segments.json"))
        self._robot_id_for_dump = int(robot_id) if robot_id is not None else 0
        self._robot_id_for_trace = self._robot_id_for_dump
        if kwargs.get('clear_astar_dump_on_start', True):
            _astar_dump_clear(self.astar_dump_path, self._robot_id_for_dump)

        # --- Internal State ---
        self.integral_ang = 0.0
        self.prev_ang_err = 0.0
        self.prev_time = time.time()
        self._seg_path = None
        self._seg_index = 0
        self._seg_goal = None
        self._tick = 0
        self._last_pose = None
        self._pose_still_cnt = 0
        self._escape_active = False
        self._escape_line_mask = None
        self._trace_last_xy = None
        self._grip_state = None
        self.cache.rebuild_if_needed()

    @staticmethod
    def normalize(angle):
        """Normalize an angle to the range [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _deg(rad):
        """Convert radians to degrees."""
        return math.degrees(rad)

    def _maybe_log_trace(self, x: float, y: float):
        """Log the robot's position to a trace file if it has moved enough."""
        try:
            cur = (float(x), float(y))
            if self._trace_last_xy is None:
                self._trace_last_xy = cur
                _trace_append_point(self.trace_json_path, self._robot_id_for_trace, x, y)
                return

            dx = cur[0] - self._trace_last_xy[0]
            dy = cur[1] - self._trace_last_xy[1]
            if (dx*dx + dy*dy) >= (self.trace_stride_px**2):
                self._trace_last_xy = cur
                _trace_append_point(self.trace_json_path, self._robot_id_for_trace, x, y)
        except Exception as e:
            print(f"[WARN] trace log error: {e}")

    def _adjust_speed(self, s):
        """Fine-tune motor speeds near the neutral point for better response."""
        if 89.75 <= s <= 90.25: return 90.0
        if 90.25 < s <= 95: return s + 1.0
        if 85 <= s < 89.75: return s - 1.0
        return s

    def _pause_at_checkpoint(self, delay_s, stop_event=None):
        """Stop the robot and wait for a specified duration."""
        if delay_s <= 0: return
        self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)
        _dbg(f"REACHED target; pausing {delay_s:.2f}s")
        t0 = time.time()
        while (time.time() - t0) < delay_s:
            if stop_event and stop_event.is_set(): break
            time.sleep(0.03)

    def _do_gripper_action(self, action: str):
        """Send a gripper command ('open' or 'close')."""
        action = action.strip().lower()
        if action in ("open", "close"):
            self.iface.write_gripper_command(action)

    def _try_escape_plan(self, start_xy):
        """Generate a short-term path to escape from being stuck."""
        if self.cache.planner is None: return None
        sx, sy = map(int, start_xy)
        mask_raw, mask_dil = self.cache.mask_raw, self.cache.mask_dilated
        if mask_raw is None or mask_dil is None: return None

        H, W = mask_dil.shape[:2]
        sx, sy = int(np.clip(sx, 0, W-1)), int(np.clip(sy, 0, H-1))
        if mask_dil[sy, sx] == 0: return None # Not stuck

        km = self.cache.kernel_erode
        bubble_r = max(8, self.cache.spacing_px + 4)
        inflated2, line2 = _make_escape_masks(mask_raw, mask_dil, sx, sy, r=bubble_r, kernel_erode=km)

        esc_wp = _find_escape_waypoint(line2, inflated2, mask_raw, sx, sy,
            r_samples=(self.cache.spacing_px, self.cache.spacing_px+10, self.cache.spacing_px+20))

        if esc_wp is None: return None
        self._escape_active = True
        self._escape_line_mask = line2
        return _densify_segment((sx, sy), esc_wp, step_px=self.linear_step_px)

    def _ensure_segment_path(self, start_xy, final_goal_xy):
        """
        Manages the current path segment. Re-plans if the path is blocked,
        finished, or invalid. Returns True if a valid path exists.
        """
        # Periodically rebuild planner cache if files have changed
        now = time.time()
        if not hasattr(self, "_next_check") or now >= self._next_check:
            self.cache.rebuild_if_needed()
            self._next_check = now + self.replan_check_period_s

        if self.cache.planner is None or self.cache.mask_dilated is None: return False

        sx, sy = int(start_xy[0]), int(start_xy[1])
        goal_tuple = (int(final_goal_xy[0]), int(final_goal_xy[1]))

        # Handle being stuck inside an obstacle
        if self.cache.mask_dilated[sy, sx] != 0:
            if self._seg_path is None:
                path = self._try_escape_plan((sx, sy))
                if not path: return False
                self._seg_path = path
                self._seg_index = 0
                self._seg_goal = goal_tuple
                _astar_dump_append(self.astar_dump_path, self._robot_id_for_dump,
                                   goal_tuple, self._seg_path, seg_type="escape")
            return True

        # Determine if a replan is needed
        need_new_plan = False
        if (self._seg_path is None or self._seg_goal != goal_tuple or self._escape_active or
            self._seg_index >= len(self._seg_path)):
            self._escape_active = False
            self._escape_line_mask = None
            need_new_plan = True
        else:
            # Check if the immediate path ahead is blocked
            cur = (sx, sy)
            nxt = (int(self._seg_path[self._seg_index][0]), int(self._seg_path[self._seg_index][1]))
            if _line_blocked_multi([self.cache.mask_raw, self.cache.mask_for_line], cur, nxt):
                need_new_plan = True

        if need_new_plan:
            path = _plan_segment_cached(self.cache, (sx, sy), goal_tuple,
                simplify_dist=self.simplify_dist_px, step_px=self.linear_step_px,
                offset_px=self.offset_px, nearest_free_radius=self.nearest_free_radius,
                clearance_req_px=self.clearance_req_px)
            if not path: return False
            self._seg_path = path
            self._seg_index = 0
            self._seg_goal = goal_tuple
            _astar_dump_append(self.astar_dump_path, self._robot_id_for_dump,
                               goal_tuple, self._seg_path, seg_type="normal")
        return True

    def _reverse_retreat(self, retreat_px: float, hold_theta: float, start_xy: tuple, stop_event=None):
        """Perform a controlled reverse movement."""
        if retreat_px <= 0: return
        x0, y0 = start_xy
        t_start = time.time()
        while (time.time() - t_start) < 3.0: # 3-second timeout
            if stop_event and stop_event.is_set(): break
            pose = self.iface.read_pos()
            if pose is None:
                time.sleep(0.05)
                continue
            x, y, theta = pose
            if math.hypot(x - x0, y - y0) >= retreat_px: break

            # Simple proportional controller to maintain heading while reversing
            ang_err = self.normalize(hold_theta - theta)
            ang_ctrl = self.Kp_ang * ang_err
            lin_ctrl = -min(self.max_lin * 0.6, 18.0)
            left = self.speed_neutral + lin_ctrl - ang_ctrl
            right = self.speed_neutral + lin_ctrl + ang_ctrl
            self.iface.write_wheel_command(left, right)
            time.sleep(0.03)
        self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)

    def run(self, stop_event=None):
        """The main control loop."""
        targets = self.iface.read_targets()
        if not targets:
            print(f"No targets found in {self.iface.target_file} for robot {self.iface.robot_id}")
            return

        idx = 0
        while idx < len(targets):
            if stop_event and stop_event.is_set(): break
            self._tick += 1
            now = time.time()
            dt = max(1e-3, now - self.prev_time)
            self.prev_time = now

            pose = self.iface.read_pos()
            if pose is None:
                _dbg(f"Pose for robot {self.iface.robot_id} not found. Stopping.", tick=self._tick)
                self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)
                time.sleep(0.5)
                continue
            x, y, theta = pose
            self._maybe_log_trace(x, y)

            # --- Target and State Management ---
            goal_x, goal_y, goal_theta, action = targets[idx]
            is_gripper_action = isinstance(action, str) and action in ("open", "close")

            # Initialize gripper state machine if this is a new gripper action
            if is_gripper_action and self._grip_state is None:
                theta_dyn = math.atan2(goal_y - y, goal_x - x)
                pre_xy, cen_xy = _grip_pre_approach(goal_x, goal_y, theta_dyn, self.gripper_offset_px, self.gripper_pre_approach_extra_px)
                self._grip_state = {"pre": pre_xy, "cen": cen_xy, "theta": theta_dyn, "stage": "to_pre"}
                self._seg_path = None # Force replan to pre-approach point

            # Determine the current planning goal based on gripper state
            if is_gripper_action and self._grip_state:
                plan_goal_xy = self._grip_state[self._grip_state["stage"]]
            else:
                plan_goal_xy = (goal_x, goal_y)

            # --- Path Planning & Safety ---
            if not self._ensure_segment_path((x, y), plan_goal_xy):
                _dbg(f"NO_PLAN: waiting for path to {plan_goal_xy}...", tick=self._tick)
                self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)
                time.sleep(self.replan_wait_backoff_s)
                continue

            # --- Waypoint Advancement ---
            cur_wp = self._seg_path[min(self._seg_index, len(self._seg_path)-1)]
            tx, ty = float(cur_wp[0]), float(cur_wp[1])
            dist_to_wp = math.hypot(tx - x, ty - y)

            if dist_to_wp < self.dist_tolerance:
                self._seg_index += 1

            # --- Target Reached Logic ---
            if self._seg_index >= len(self._seg_path):
                # We are at the end of the planned polyline.
                # Now, check if we are close enough to the *actual* final destination.
                if math.hypot(plan_goal_xy[0] - x, plan_goal_xy[1] - y) < self.final_distance_tol:
                    # Final destination is reached.
                    self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)

                    if is_gripper_action and self._grip_state:
                        # --- Gripper State Machine ---
                        if self._grip_state["stage"] == "to_pre":
                            self._grip_state["stage"] = "cen" # Use "cen" as key
                            self._seg_path = None # Force replan to final center point
                            continue
                        elif self._grip_state["stage"] == "cen":
                            _dbg(f"GRIPPER: {action}", tick=self._tick)
                            self._do_gripper_action(action)
                            self._pause_at_checkpoint(self.gripper_action_pause_s, stop_event)
                            if action == "open" and self.post_drop_retreat_px > 0:
                                self._reverse_retreat(self.post_drop_retreat_px, self._grip_state["theta"], self._grip_state["cen"], stop_event)
                            self._grip_state = None
                            idx += 1
                            continue
                    else:
                        # --- Standard Waypoint Action ---
                        if isinstance(action, (int, float)) and action > 0:
                            self._pause_at_checkpoint(action / 1000.0, stop_event)
                        idx += 1
                        continue
                else:
                    # End of polyline, but not close enough to goal. Force replan.
                    self._seg_path = None
                    continue

            # --- PID Control Calculation ---
            heading_to_wp = math.atan2(ty - y, tx - x)
            ang_err = self.normalize(heading_to_wp - theta)
            self.integral_ang += ang_err * dt
            self.integral_ang = np.clip(self.integral_ang, -0.4, 0.4)
            deriv_ang = (ang_err - self.prev_ang_err) / dt
            self.prev_ang_err = ang_err

            ang_ctrl = (self.Kp_ang * ang_err) + (self.Ki_ang * self.integral_ang) + (self.Kd_ang * deriv_ang)
            lin_ctrl = np.clip(self.Kp_dist * dist_to_wp, -self.max_lin, self.max_lin)

            # Slow down if path ahead is narrow or if turning sharply
            forward_gain = max(0.0, math.cos(ang_err))**1.5
            if abs(self._deg(ang_err)) <= self.ang_tolerance_deg:
                forward_gain = 1.0

            left  = self.speed_neutral + (lin_ctrl * forward_gain) - ang_ctrl
            right = self.speed_neutral + (lin_ctrl * forward_gain) + ang_ctrl

            left  = np.clip(left, self.speed_min, self.speed_max)
            right = np.clip(right, self.speed_min, self.speed_max)

            self.iface.write_wheel_command(left, right)
            self.iface.log_error(dist_to_wp, ang_err)

        # --- Loop End ---
        self.iface.write_wheel_command(self.speed_neutral, self.speed_neutral)
        print(f"Controller for robot {self.iface.robot_id} finished all targets.")