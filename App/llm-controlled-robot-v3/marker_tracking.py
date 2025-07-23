"""
marker_tracking.py

Provides a helper to generate an obstacle-aware marker-tracking path
and write it out as x,y,theta steps.
"""
import os
import math
from typing import List, Tuple, Dict, Any
from tkinter import messagebox
from astar import PathPlanner


def generate_marker_track(
    sequence: List[str],
    home_pos: Any,
    alphabet_info: Any,
    obstacles: List[Dict[str,Any]],
    arena_corners: Any,
    camera_size: Tuple[int,int],
    last_robot_angle: float,
    out_dir: str="targets/marker_track"
) -> None:
    """
    Given a list like ["Home","A","B","Home"], build a routing
    through obstacles and save x,y,theta steps to targets.txt.

    - sequence: ordered labels to visit
    - home_pos: (x,y) for "Home"
    - alphabet_info: list of (letter, (x,y))
    - obstacles: list of dicts with bbox/centroid/area
    - arena_corners: for internal PathPlanner if needed
    - camera_size: (width, height) of full image
    - last_robot_angle: initial heading for first point
    - out_dir: directory to write targets.txt
    """
    # --- 1) normalize home_pos ---
    if isinstance(home_pos, list):
        # e.g. [('Home',(x,y))]
        _, coord, corners = home_pos[0]
        hx, hy = coord
    else:
        hx, hy = home_pos

    lookup: Dict[str,Tuple[int,int]] = {
        "HOME": (int(hx), int(hy))
    }

    # --- 2) normalize alphabet_info ---
    items = alphabet_info.items() if isinstance(alphabet_info, dict) else alphabet_info
    for let, coord in items:
        cx, cy = coord
        lookup[let.upper()] = (int(cx), int(cy))

    # resolve sequence to points
    try:
        pts = [ lookup[label.upper()] for label in sequence ]
    except KeyError as e:
        messagebox.showerror("Error", f"Unknown marker '{e.args[0]}' in sequence.")
        return

    planner = PathPlanner(
        obstacles,
        camera_size,
        arena_corners=arena_corners
    )

    full_path: List[Tuple[int, int]] = []
    for start, goal in zip(pts, pts[1:]):
        route = planner.find_obstacle_aware_path(start, goal, simplify_dist=10)
        if route is None:
            messagebox.showerror(
                "Routing Failed",
                f"No path from {start} to {goal} (blocked)."
            )
            return
        if not full_path:
            full_path.extend(route)
        else:
            full_path.extend(route[1:])

    # compute headings
    routed_with_theta: List[Tuple[int, int, float]] = []
    for i, (x, y) in enumerate(full_path):
        if i == 0:
            θ = last_robot_angle
        else:
            x0, y0 = full_path[i-1]
            θ = math.atan2(y - y0, x - x0)
        routed_with_theta.append((x, y, θ))

    # write out
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "targets.txt")
    with open(out_path, "w") as f:
        total = len(routed_with_theta)
        f.write(f"({total}/{total})\n")
        for x, y, θ in routed_with_theta:
            f.write(f"{x},{y},{θ:.3f}\n")
    return routed_with_theta

def pathPlanning(list_of_points, arena_corners, obstacles, camera_size, out_dir="targets/output"):
      planner = PathPlanner(
        obstacles,
        camera_size,
        arena_corners=arena_corners
    )
    full_path: List[Tuple[int, int]] = []
    for start, goal in zip(list_of_points, list_of_points[1:]):
        route = planner.find_obstacle_aware_path(start, goal, simplify_dist=10)
        if route is None:
            messagebox.showerror(
                "Routing Failed",
                f"No path from {start} to {goal} (blocked)."
            )
            return
        if not full_path:
            full_path.extend(route)
        else:
            full_path.extend(route[1:])
        
    # compute headings
    routed_with_theta: List[Tuple[int, int, float]] = []
    for i, (x, y) in enumerate(full_path):
        if i == 0:
            θ = 0
        else:
            x0, y0 = full_path[i-1]
            θ = math.atan2(y - y0, x - x0)
        routed_with_theta.append((x, y, θ))
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "targets.txt")
    with open(out_path, "w") as f:
        total = len(routed_with_theta)
        f.write(f"({total}/{total})\n")
        for x, y, θ in routed_with_theta:
            f.write(f"{x},{y},{θ:.3f}\n")

    return routed_with_theta