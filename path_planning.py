#!/usr/bin/env python3
"""
Planning helpers + path writers (JSON-only).

JSON format (simple):
{
  "robots": [
    { "id": 782, "path": [[x, y, 0], [x2, y2, 5000], [x3, y3, "open"], ...] }
  ]
}

Functions:
- trace_targets(input_target_list, *, robot_id, data_folder="Data", spacing=50, delay=0,
                offset=100, path_json="Targets/paths.json", verbose=False) -> str
    - Builds a minimal waypoint list for robot_id: [start] + chosen targets
      (each target is either the exact point if directly reachable, or a
       4-direction offset (N,E,S,W) at `offset` px).
    - Saves/updates JSON under the simple shape above (replaces entry if id exists).
    - Returns:
        "Robot not found."               if robot_id is not in Data/robot_pos.txt
        "Path generation successful."    on success

- pick_and_drop(robot_id, pick_coordinates, drop_coordinates, *,
                data_folder="Data", spacing=50, gripper_offset_px=20,
                path_json="Targets/paths.json", verbose=False) -> str
    - Computes approach points a fixed distance (20 px default) from the pick and
      drop coordinates, finds reachable ones, and writes:
        [[sx,sy,0], [pick_approach_x, pick_approach_y, "close"],
         [drop_approach_x, drop_approach_y, "open"]]
    - Replaces/creates the JSON entry for the robot id.

Dependencies:
- astar.PathPlanner
- Functions.Library.Agent.load_data.read_data
- OpenCV, NumPy
"""

import os
import json
import math
from datetime import datetime

import cv2
import numpy as np

from astar import PathPlanner
from Functions.Library.Agent.load_data import read_data


# ---------- JSON I/O (simple {robots:[{id, path}]}) ----------
def _load_paths_json(path: str) -> dict:
    if not os.path.exists(path):
        return {"robots": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "robots" in data and isinstance(data["robots"], list):
                return data
    except Exception:
        pass
    return {"robots": []}


def _save_paths_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _upsert_robot_path(path_json: str, robot_id: int, path_points: list) -> None:
    """
    Upsert robot path into the simple JSON format.
    path_points is a list of [x, y, delay_or_action].
    """
    data = _load_paths_json(path_json)
    robots = data.get("robots", [])

    new_entry = {
        "id": int(robot_id),
        "path": path_points,
    }

    replaced = False
    for i, entry in enumerate(robots):
        if int(entry.get("id", -1)) == int(robot_id):
            robots[i] = new_entry
            replaced = True
            break
    if not replaced:
        robots.append(new_entry)

    data["robots"] = robots
    _save_paths_json(path_json, data)


# ---------- Planning helpers ----------
def _inflate_mask(mask: np.ndarray, spacing: int) -> np.ndarray:
    """Inflate mask by `spacing` pixels (ellipse)."""
    k = 2 * int(spacing) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel)


def _path_exists(planner: PathPlanner, start_xy, goal_xy, simplify_dist: int = 10) -> bool:
    """Return True if a path exists from start_xy to goal_xy."""
    path = planner.find_obstacle_aware_path(start_xy, goal_xy, simplify_dist)
    return bool(path)


def _choose_reachable_point(planner: PathPlanner, current_xy, target_xy, offset_px: int, log) -> tuple | None:
    """
    Choose either:
      - target_xy (if directly reachable), or
      - one of four offset positions at distance=offset_px (N,E,S,W) that is reachable.
    Returns the chosen (x, y) or None if none are reachable.
    """
    if _path_exists(planner, current_xy, target_xy):
        log(f"  ✔ direct ok -> {target_xy}")
        return target_xy

    tx, ty = target_xy
    candidates = [
        (tx, ty - offset_px),  # N
        (tx + offset_px, ty),  # E
        (tx, ty + offset_px),  # S
        (tx - offset_px, ty),  # W
    ]
    for p in candidates:
        if _path_exists(planner, current_xy, p):
            log(f"  ✔ using offset -> {p}")
            return p

    log("  ✖ no reachable offset")
    return None


def _point_in_free(mask: np.ndarray, pt: tuple[int, int]) -> bool:
    """Check if pt lies in free space of the (binary) mask (0 = free)."""
    x, y = int(pt[0]), int(pt[1])
    H, W = mask.shape[:2]
    if not (0 <= x < W and 0 <= y < H):
        return False
    return mask[y, x] == 0


def _angle_spiral(center_angle: float, n: int):
    """
    Yield angles around center_angle in an outward spiral:
    center, +d, -d, +2d, -2d, ...
    with d = 2π/n (for a full circle coverage).
    """
    d = 2.0 * math.pi / max(1, n)
    yield center_angle
    k = 1
    while k <= n // 2:
        yield center_angle + k * d
        yield center_angle - k * d
        k += 1


def _find_reachable_approach(planner: PathPlanner,
                             current_xy: tuple[int, int],
                             target_xy: tuple[int, int],
                             radius_px: int,
                             log,
                             n_angles: int = 32) -> tuple[int, int] | None:
    """
    Find a base position around `target_xy` on the circle of radius `radius_px`
    that is reachable from `current_xy`. Preference is given to the point aligned
    with the vector current->target; otherwise we search angles around it.
    """
    cx, cy = current_xy
    tx, ty = target_xy

    vx, vy = (tx - cx, ty - cy)
    norm = math.hypot(vx, vy)
    if norm < 1e-6:
        # If we're already at the target, pick an arbitrary direction
        base_angle = 0.0
    else:
        base_angle = math.atan2(vy, vx)

    H, W = planner.mask.shape[:2]

    for ang in _angle_spiral(base_angle, n_angles):
        # Place base point so that gripper (ahead by radius) would sit at target
        ax = int(round(tx - radius_px * math.cos(ang)))
        ay = int(round(ty - radius_px * math.sin(ang)))

        if not (0 <= ax < W and 0 <= ay < H):
            continue
        if not _point_in_free(planner.mask, (ax, ay)):
            continue
        if _path_exists(planner, current_xy, (ax, ay)):
            log(f"  ✔ approach {radius_px}px at angle={math.degrees(ang):.1f} -> ({ax},{ay})")
            return (ax, ay)

    log("  ✖ no reachable approach on circle")
    return None

def _heading_rad(p_from: tuple[int, int], p_to: tuple[int, int]) -> float:
    """Return heading (radians) from p_from -> p_to, in (-π, π]."""
    dx = float(p_to[0] - p_from[0])
    dy = float(p_to[1] - p_from[1])
    return math.atan2(dy, dx)


# ---------- Public APIs ----------
def trace_targets(
    input_target_list,
    *,
    robot_id: int,
    data_folder: str = "Data",
    spacing: int = 50,
    delay: int | list[int] = 0,
    offset: int = 100,
    path_json: str = "Targets/paths.json",
    verbose: bool = False,
) -> str:
    """
    Build a minimal list of points for `robot_id`: [start] + [chosen targets],
    where each target is either the original (if directly reachable) or one of
    four offset points (N,E,S,W) at `offset` px.

    Saves JSON as:
      { "robots": [ { "id": <robot_id>, "path": [[x,y,theta_rad,delay_or_0], ...] } ] }

    Returns:
      - "Robot not found."
      - "Path generation successful."
    """
    log = print if verbose else (lambda *a, **k: None)

    # --- Load environment & robot pose ---
    data = read_data(data_folder)
    if data is None:
        raise RuntimeError(f"Could not read data from '{data_folder}'")

    robots = data.get("robot_pos", None)
    if robots is None or len(robots) == 0:
        return "Robot not found."

    matches = robots[np.where(robots[:, 0].astype(int) == int(robot_id))] if isinstance(robots, np.ndarray) else []
    if matches is None or len(matches) == 0:
        return "Robot not found."

    sx, sy = int(matches[0][1]), int(matches[0][2])
    # try to read initial heading (deg) if present (4th column)
    s_theta_deg = None
    try:
        s_theta_deg = float(matches[0][3])
    except Exception:
        s_theta_deg = None

    start_xy = (sx, sy)

    arena = [tuple(map(int, row)) for row in data.get("arena_corners", [])]
    polys = [[tuple(map(int, p)) for p in poly] for poly in data.get("obstacles", [])]

    # --- Planner with inflated mask ---
    frame_path = os.path.join(data_folder, "frame_img.png")
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image at '{frame_path}'")
    H, W = frame.shape[:2]

    obstacles = [{"corners": poly} for poly in polys]
    planner = PathPlanner(obstacles, (H, W), arena_corners=arena)
    planner.mask = _inflate_mask(planner.mask, int(spacing))

    # --- Normalize inputs ---
    targets = [(int(x), int(y)) for x, y in input_target_list]
    if isinstance(delay, (int, float)):
        delays = [int(delay)] * len(targets)
    else:
        delays = [int(d) for d in delay]
        if len(delays) != len(targets):
            raise ValueError("Length of 'delay' must match number of targets")

    # --- Choose points (either target or offset) ---
    chosen_points: list[tuple[int, int]] = []
    current = start_xy

    log(f"[trace] id={robot_id} start={current} spacing={spacing}px offset={offset}px")
    for i, tgt in enumerate(targets):
        log(f"[target {i+1}] desired={tgt}")
        choice = _choose_reachable_point(planner, current, tgt, int(offset), log)
        if choice is None:
            log(f"  -> skipping unreachable {tgt}")
            continue
        choice = (int(choice[0]), int(choice[1]))
        chosen_points.append(choice)
        current = choice
        log(f"  -> chosen {choice} (delay={delays[i]})")

    # --- Build JSON path with theta in radians ---
    if s_theta_deg is None:
        if len(chosen_points) > 0:
            s_theta = _heading_rad(start_xy, chosen_points[0])
        else:
            s_theta = 0.0
    else:
        s_theta = math.radians(s_theta_deg)

    out_path: list[list[int | float | str]] = [[start_xy[0], start_xy[1], float(s_theta), 0]]

    prev = start_xy
    for i, pt in enumerate(chosen_points):
        th = _heading_rad(prev, pt)
        out_path.append([pt[0], pt[1], float(th), int(delays[i])])
        prev = pt

    _upsert_robot_path(path_json, int(robot_id), out_path)
    log(f"[save] {path_json} updated for id={robot_id}")

    return "Path generation successful."


def pick_and_drop(
    robot_id: int,
    pick_coordinates: tuple[int, int],
    drop_coordinates: tuple[int, int],
    *,
    data_folder: str = "Data",
    spacing: int = 50,
    gripper_offset_px: int = 10,
    path_json: str = "Targets/paths.json",
    verbose: bool = False,
) -> str:
    """
    Generate a path for picking and dropping an object with a forward gripper.

    Output JSON path for robot_id (theta in radians):
      [[sx,sy,theta_start,0],
       [pick_approach_x, pick_approach_y, theta_face_pick, "close"],
       [drop_approach_x, drop_approach_y, theta_face_drop, "open"]]
    """
    log = print if verbose else (lambda *a, **k: None)

    # --- Load environment & pose ---
    data = read_data(data_folder)
    if data is None:
        raise RuntimeError(f"Could not read data from '{data_folder}'")

    robots = data.get("robot_pos", None)
    if robots is None or len(robots) == 0:
        return "Robot not found."

    matches = robots[np.where(robots[:, 0].astype(int) == int(robot_id))] if isinstance(robots, np.ndarray) else []
    if matches is None or len(matches) == 0:
        return "Robot not found."

    sx, sy = int(matches[0][1]), int(matches[0][2])
    start_xy = (sx, sy)

    # read initial theta if present (deg → rad)
    s_theta_deg = None
    try:
        s_theta_deg = float(matches[0][3])
    except Exception:
        s_theta_deg = None

    arena = [tuple(map(int, row)) for row in data.get("arena_corners", [])]
    polys = [[tuple(map(int, p)) for p in poly] for poly in data.get("obstacles", [])]

    frame_path = os.path.join(data_folder, "frame_img.png")
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image at '{frame_path}'")
    H, W = frame.shape[:2]

    obstacles = [{"corners": poly} for poly in polys]
    planner = PathPlanner(obstacles, (H, W), arena_corners=arena)
    planner.mask = _inflate_mask(planner.mask, int(spacing))

    # --- Compute approaches ---
    pick_xy = (int(pick_coordinates[0]), int(pick_coordinates[1]))
    drop_xy = (int(drop_coordinates[0]), int(drop_coordinates[1]))

    log(f"[pick&drop] id={robot_id} start={start_xy} spacing={spacing}px approach_r={gripper_offset_px}px")
    log(f"  pick={pick_xy}  drop={drop_xy}")

    pick_app = _find_reachable_approach(planner, start_xy, pick_xy, gripper_offset_px, log)
    if pick_app is None:
        return "Unable to find reachable pick approach."

    # Bias drop search by pick->drop direction (we pass pick_xy as the "current" base)
    drop_app = _find_reachable_approach(planner, pick_xy, drop_xy, gripper_offset_px, log)
    if drop_app is None:
        return "Unable to find reachable drop approach."

    # Start theta: prefer provided; else face toward pick_app; else 0
    if s_theta_deg is None:
        s_theta = _heading_rad(start_xy, pick_app) if pick_app is not None else 0.0
    else:
        s_theta = math.radians(s_theta_deg)

    theta_pick = _heading_rad(pick_app, pick_xy)
    theta_drop = _heading_rad(drop_app, drop_xy)

    path_points = [
        [start_xy[0], start_xy[1], float(s_theta), 0],
        [pick_app[0], pick_app[1], float(theta_pick), "close"],
        [drop_app[0], drop_app[1], float(theta_drop), "open"],
    ]

    _upsert_robot_path(path_json, int(robot_id), path_points)
    log(f"[save] {path_json} updated for id={robot_id}")

    return "Pick & drop path generation successful."




# ---------- Quick tests ----------
if __name__ == "__main__":
    # Adjust these to your local map/poses to test quickly.
    # Assumes:
    # - Data/frame_img.png exists
    # - Data/robot_pos.txt has at least one line: id,x,y,theta   (e.g., "782,810,278,0.0")
    # - Functions.Library.Agent.load_data.read_data understands your Data/* files

    TEST_ROBOT_ID = 2
    PATH_JSON = "Targets/paths.json"

    # 1) trace_targets test
    try:
        print("=== trace_targets test ===")
        msg = trace_targets(
            input_target_list=[(1331, 379), (148, 265)],
            robot_id=1,
            data_folder="Data",
            spacing=40,
            delay=5000,
            offset=100,
            path_json=PATH_JSON,
            verbose=True,
        )
        print("trace_targets:", msg)
    except Exception as e:
        print("[test trace_targets] error:", e)

    # 2) pick_and_drop test
    try:
        print("\n=== pick_and_drop test ===")
        msg2 = pick_and_drop(
            robot_id=TEST_ROBOT_ID,
            pick_coordinates=(800, 320),
            drop_coordinates=(820, 360),
            data_folder="Data",
            spacing=40,
            gripper_offset_px=10,
            path_json=PATH_JSON,
            verbose=True,
        )
        print("pick_and_drop:", msg2)
    except Exception as e:
        print("[test pick_and_drop] error:", e)
