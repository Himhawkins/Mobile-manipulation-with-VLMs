#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from astar import PathPlanner
from Functions.Library.Agent.load_data import read_data

def find_path_with_offset(planner, current_pos, target_pos, offset):
    """
    Checks for a valid path to points offset from the target in four cardinal directions.
    """
    tx, ty = target_pos
    
    # Define the four points to check: North, East, South, West
    offset_points = [
        (tx, ty - offset),  # Top
        (tx + offset, ty),  # Right
        (tx, ty + offset),  # Bottom
        (tx - offset, ty)   # Left
    ]

    # Try to find a path to any of the offset points
    for point in offset_points:
        path = planner.find_obstacle_aware_path(current_pos, point, 10)
        if path:
            # If a path is found, return it and the successful point
            return path, point
    
    # If no path was found after checking all four points
    return None, None

def trace_targets(
    input_target_list,
    output_target_path="Targets/path.txt",
    start=None,
    data_folder="Data",
    spacing=50,
    delay=0,
    out_path="Data/trace_overlay.png",
    offset=100  # <-- New changeable offset argument
):
    # Sections 1, 2, 3, 4 (Loading, Setup, etc.) remain the same
    # ... (paste the setup code from your previous version here) ...
    data = read_data(data_folder)
    if data is None:
        raise RuntimeError(f"Could not read data from '{data_folder}'")
    arena = [tuple(map(int, row)) for row in data['arena_corners']]
    polygon_obs = [ [tuple(map(int, pt)) for pt in poly] for poly in data['obstacles']]
    sx, sy, _ = data['robot_pos']
    if start is None:
        start = (int(sx), int(sy))
    frame_path = os.path.join(data_folder, "frame_img.png")
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image at '{frame_path}'")
    h, w = frame.shape[:2]
    obs = [{"corners": poly} for poly in polygon_obs]
    planner = PathPlanner(obs, (h, w), arena)
    k = 2 * int(spacing) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    planner.mask = cv2.dilate(planner.mask, kernel)
    targets = [(int(x), int(y)) for x,y in input_target_list]
    if isinstance(delay, (int, float)):
        delays = [int(delay)] * len(targets)
    else:
        delays = [int(d) for d in delay]
        if len(delays) != len(targets):
            raise ValueError("Length of 'delay' must match number of targets")

    # --- Pathfinding Logic with Your Offset Method ---
    paths = []
    successful_delays = []
    current = start
    
    for i, tgt in enumerate(targets):
        path = planner.find_obstacle_aware_path(current, tgt, 10)
        final_tgt = tgt
        
        if not path:
            print(f"Segment {i+1}: {current} → {tgt} is UNREACHABLE.")
            print(f" -> Target is inside an obstacle. Trying {offset}px offset points...")
            path, final_tgt = find_path_with_offset(planner, current, tgt, offset)

        # If a path was found (either original or adjusted), add it
        if path:
            print(f"Segment {i+1}: {current} → {final_tgt} [{len(path)} steps]")
            paths.append(path)
            successful_delays.append(delays[i])
            current = final_tgt
        else:
            print(f" -> SKIPPING target {tgt} as no path could be generated.")
            
    # Sections 5 and 6 (Plotting and Saving) remain the same
    # ... (paste the plotting and saving code from your previous version here) ...
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.axis("off")
    ax.add_patch(Polygon(arena, closed=True, fill=False, edgecolor='yellow', linewidth=2))
    for poly in polygon_obs:
        ax.add_patch(Polygon(poly, closed=True, facecolor='red', alpha=0.3, edgecolor='white'))
    ax.plot(start[0], start[1], 'o', color='cyan', markersize=10, label='robot')
    if targets:
        txs, tys = zip(*targets)
        ax.scatter(txs, tys, s=80, facecolors='none', edgecolors='white', label='targets')
    for path in paths:
        xs, ys = zip(*path)
        ax.plot(xs, ys, '-', linewidth=2, color='lime')
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    with open(output_target_path, "w") as f:
        for i, path in enumerate(paths):
            for j, (x, y) in enumerate(path):
                if j == len(path) - 1:
                    f.write(f"{x},{y},{successful_delays[i]}\n")
                else:
                    f.write(f"{x},{y},{0}\n")
    return f"Path Planned! and saved to {output_target_path}"


if __name__ == "__main__":
    DATA_FOLDER  = "Data"
    SPACING      = 20
    input_list = [[1464.0, 220.0], [34.0, 88.0]]
    # input_list = [[1394.0, 220.0], [104.0, 88.0]]

    trace_targets(
        input_target_list=input_list,
        output_target_path="Targets/path.txt",
        data_folder=DATA_FOLDER,
        spacing=SPACING,
        delay=5000,
        out_path="Data/trace_overlay.png"
    )
