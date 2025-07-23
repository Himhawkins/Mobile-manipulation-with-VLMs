#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

from astar import PathPlanner
from Functions.Library.Agent.load_data import read_data

def trace_targets(input_target_list, output_target_path, start=None, data_folder="Data", spacing=20):
    data = read_data(data_folder)
    if data is None:
        raise RuntimeError(f"Could not read data from '{data_folder}'")

    arena = [tuple(map(int,row)) for row in data['arena_corners']]
    obs   = [{"bbox":tuple(map(int,row))} for row in data['obstacles']]
    sx, sy, _ = data['robot_pos']
    if start is None: 
        start = (int(sx), int(sy))


    frame_path = os.path.join(data_folder, "frame_img.png")
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image at '{frame_path}'")
    h, w = frame.shape[:2]

    planner = PathPlanner(obs, (h, w), arena)
    # buffer mask
    k = 2*spacing + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    planner.mask = cv2.dilate(planner.mask, kernel)

    targets = input_target_list
    paths = []
    current = start

    for idx, tgt in enumerate(targets, start=1):
        path = planner.astar_path(current, tgt)
        if not path:
            print(f"Segment {idx}: {current} → {tgt} is UNREACHABLE")
        else:
            print(f"Segment {idx}: {current} → {tgt} [{len(path)} steps]")
            paths.append(path)
            current = tgt
    
    improved_points = []
    for path in paths:
        improved_points.extend(path)
    
    # Save them
    with open(output_target_path, "w") as f:
        for x, y in improved_points:
            f.write(f"{x},{y}\n")

    return frame, arena, [r["bbox"] for r in obs], start, targets, paths

def plot_trace(frame, arena, obstacles, start, targets, paths):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(img)
    ax.set_axis_off()

    # arena boundary
    poly = Polygon(arena, closed=True, fill=False,
                   edgecolor='yellow', linewidth=2)
    ax.add_patch(poly)

    # obstacles
    for x,y,w,h in obstacles:
        rect = Rectangle((x,y), w, h, facecolor='red', alpha=0.3)
        ax.add_patch(rect)

    # robot start
    ax.plot(start[0], start[1], 'o', color='cyan', markersize=10, label='robot')

    # raw targets
    txs, tys = zip(*targets)
    ax.scatter(txs, tys, s=80, facecolors='none', edgecolors='white', label='targets')

    # draw each A* segment
    for path in paths:
        xs, ys = zip(*path)
        ax.plot(xs, ys, '-', linewidth=2, color='lime')

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_FOLDER  = "Data"
    TARGETS_FILE = "Targets/targets.txt"
    SPACING      = 5

    # Trace and plot
    frame, arena, obstacles, start, targets, paths = trace_targets(
        data_folder=DATA_FOLDER,
        input_target_path=TARGETS_FILE,
        output_target_path="Targets/improved_targets.txt",
        spacing=SPACING
    )