#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from astar import PathPlanner
from Functions.Library.Agent.load_data import read_data

def trace_targets(
    input_target_list,          # now a list of [x, y] or (x, y)
    output_target_path,
    start=None,
    data_folder="Data",
    spacing=50,
    delay=0,
    out_path="Data/trace_overlay.png"
):
    # 1) load world
    data = read_data(data_folder)
    if data is None:
        raise RuntimeError(f"Could not read data from '{data_folder}'")

    arena = [tuple(map(int, row)) for row in data['arena_corners']]
    polygon_obs = [ [tuple(map(int, pt)) for pt in poly] for poly in data['obstacles']]
    obs = [{"corners": [tuple(pt) for pt in row]} for row in data['obstacles']]
    sx, sy, _ = data['robot_pos']
    if start is None:
        start = (int(sx), int(sy))

    # 2) load the image
    frame_path = os.path.join(data_folder, "frame_img.png")
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image at '{frame_path}'")
    h, w = frame.shape[:2]

    # 3) convert polygon obstacles into bounding boxes for planner
    obs = [{"corners": poly} for poly in polygon_obs]

    planner = PathPlanner(obs, (h, w), arena)
    k = 2 * int(spacing) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    planner.mask = cv2.dilate(planner.mask, kernel)

    # 4) use the provided list directly
    targets = [(int(x), int(y)) for x,y in input_target_list]

    # --- delays per checkpoint ---
    if isinstance(delay, (int, float)):
        delays = [int(delay)] * len(targets)
    else:
        # assume iterable of delays, one per target
        delays = [int(d) for d in delay]
        if len(delays) != len(targets):
            raise ValueError("Length of 'delay' must match number of targets")

    paths = []
    current = start
    for idx, tgt in enumerate(targets, start=1):
        path = planner.find_obstacle_aware_path(current, tgt, 10)
        if not path:
            print(f"Segment {idx}: {current} → {tgt} is UNREACHABLE")
        else:
            print(f"Segment {idx}: {current} → {tgt} [{len(path)} steps]")
            paths.append(path)
            current = tgt

    # 5) plot & save overlay
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.axis("off")

    # arena boundary
    ax.add_patch(Polygon(arena, closed=True, fill=False,
                         edgecolor='yellow', linewidth=2))

    # draw polygon obstacles
    for poly in polygon_obs:
        ax.add_patch(Polygon(poly, closed=True, facecolor='red', alpha=0.3, edgecolor='white'))

    # robot start
    ax.plot(start[0], start[1], 'o', color='cyan', markersize=10, label='robot')

    # raw targets
    if targets:
        txs, tys = zip(*targets)
        ax.scatter(txs, tys, s=80, facecolors='none',
                   edgecolors='white', label='targets')

    # draw paths
    for path in paths:
        xs, ys = zip(*path)
        ax.plot(xs, ys, '-', linewidth=2, color='lime')

    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    # 6) save: write delay only at checkpoint (last point of each path)
    with open(output_target_path, "w") as f:
        for i, path in enumerate(paths):
            for j, (x, y) in enumerate(path):
                if j == len(path) - 1:  # checkpoint reached
                    f.write(f"{x},{y},{delays[i]}\n")
                else:
                    f.write(f"{x},{y},{0}\n")


    return f"Path Planned! and saved to {output_target_path}"


if __name__ == "__main__":
    DATA_FOLDER  = "Data"
    SPACING      = 20
    input_list = [[250.0, 294.0], [966.0, 333.0]]

    trace_targets(
        input_target_list=input_list,
        output_target_path="Targets/path.txt",
        data_folder=DATA_FOLDER,
        spacing=SPACING,
        delay=5000,
        out_path="Data/trace_overlay.png"
    )
