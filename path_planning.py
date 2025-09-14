#!/usr/bin/env python3

import os
import json

import cv2
import numpy as np

from astar import PathPlanner
from Functions.Library.Agent.load_data import read_data


def _inflate_mask(mask: np.ndarray, spacing: int) -> np.ndarray:
    """Inflate mask by `spacing` pixels (ellipse)."""
    k = 2 * int(spacing) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel)


def _try_plan(planner: PathPlanner, start_xy, goal_xy, simplify_dist=10) -> bool:
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
    log(f"  Trying direct plan: {current_xy} -> {target_xy}")
    if _try_plan(planner, current_xy, target_xy):
        log("    ✔ Direct path OK")
        return target_xy

    tx, ty = target_xy
    candidates = [
        (tx, ty - offset_px),  # North
        (tx + offset_px, ty),  # East
        (tx, ty + offset_px),  # South
        (tx - offset_px, ty),  # West
    ]

    for p in candidates:
        log(f"    Trying offset {p}")
        if _try_plan(planner, current_xy, p):
            log(f"    ✔ Using offset {p}")
            return p

    log("    ✖ No reachable offset")
    return None


def _load_paths_json(path: str) -> dict:
    """Load JSON structure { 'robots': [ {id, path}, ... ] } or return empty."""
    if not os.path.exists(path):
        return {"robots": []}
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "robots" in data and isinstance(data["robots"], list):
                return data
    except Exception:
        pass
    return {"robots": []}


def _save_paths_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True
    )
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _upsert_robot_path(json_path: str, robot_id: int, path_points: list[list[int]]):
    """
    Upsert robot path into JSON (SIMPLE form):
    {
      "id": <int>,
      "path": [[x, y, delay], ...]  # includes start as the first triple [sx, sy, 0]
    }
    - If the robot id exists, replace its entry.
    - Else, append new entry.
    """
    data = _load_paths_json(json_path)
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
    _save_paths_json(json_path, data)


def trace_targets(
    input_target_list,
    output_json_path="Data/paths.json",
    start=None,
    data_folder="Data",
    spacing=50,
    delay=0,
    offset=100,
    robot_id=None,
    verbose=False,
):
    """
    Build a minimal list of points:
      path = [[sx, sy, 0], [x1, y1, d1], [x2, y2, d2], ...]
    where each subsequent point is either the target itself (if reachable) or a
    reachable offset (N/E/S/W by `offset` px). Persist to SIMPLE JSON.

    Returns:
    - "Robot not found."
    - "Path generation successful."
    """
    log = print if verbose else (lambda *_: None)

    # --- Load data ---
    data = read_data(data_folder)
    if data is None:
        raise RuntimeError(f"Could not read data from '{data_folder}'")

    arena = [tuple(map(int, row)) for row in data.get("arena_corners", [])]
    polygon_obs = [[tuple(map(int, pt)) for pt in poly] for poly in data.get("obstacles", [])]
    robots_arr = data.get("robot_pos", None)

    # --- Resolve robot start pose ---
    if robots_arr is not None and len(robots_arr) > 0:
        if robot_id is None:
            rid, rx, ry, _ = robots_arr[0]
            robot_id = int(rid)
            log(f"[info] robot_id not provided; defaulting to first robot id={robot_id}")
        else:
            matches = robots_arr[np.where(robots_arr[:, 0].astype(int) == int(robot_id))]
            if matches.shape[0] == 0:
                return "Robot not found."
            rx, ry = matches[0][1], matches[0][2]

        if start is None:
            start = (int(rx), int(ry))
    else:
        # No robots in data at all
        if robot_id is not None:
            return "Robot not found."
        if start is None:
            raise RuntimeError("No robot poses available and no explicit start provided.")

    # --- Prepare planner ---
    frame_path = os.path.join(data_folder, "frame_img.png")
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image at '{frame_path}'")
    H, W = frame.shape[:2]

    obstacles = [{"corners": poly} for poly in polygon_obs]
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

    # --- Build minimal path list: START + chosen (with offsets) ---
    current = (int(start[0]), int(start[1]))
    json_path_points: list[list[int]] = [[current[0], current[1], 0]]  # include start first
    log(f"[trace] start={current} spacing={spacing}px offset={offset}px")

    for i, (tgt, d) in enumerate(zip(targets, delays)):
        log(f"[target {i+1}] desired={tgt}")
        chosen = _choose_reachable_point(planner, current, tgt, int(offset), log)
        if chosen is not None:
            cx, cy = int(chosen[0]), int(chosen[1])
            json_path_points.append([cx, cy, int(d)])
            current = (cx, cy)
            log(f"  -> appended: {[cx, cy, int(d)]}")
        else:
            log(f"  -> skipping unreachable target {tgt}")

    # --- Save/replace in SIMPLE JSON by id ---
    robot_id_to_save = 0 if robot_id is None else int(robot_id)
    _upsert_robot_path(output_json_path, robot_id_to_save, json_path_points)
    log(f"[write] paths.json (id={robot_id_to_save}) -> {output_json_path}")

    return "Path generation successful."


# ---- Quick local test ----
if __name__ == "__main__":
    msg = trace_targets(
        input_target_list=[(1331, 479), (48, 265)],
        output_json_path="Targets/paths.json",
        data_folder="Data",
        spacing=20,
        delay=5000,
        offset=100,
        robot_id=782,   # or set a valid id
        verbose=False
    )
    print(msg)
