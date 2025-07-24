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
    spacing=20
):
    """
    Traces A* paths between each consecutive pair in input_target_list,
    avoiding obstacles read from data_folder, and writes the
    concatenated reachable points to output_target_path.

    :param input_target_list: List[Tuple[int,int]] — your sequence of targets
    :param output_target_path: str — CSV path to write improved targets
    :param start: Tuple[int,int] or None — optional override of robot start pos
    :param data_folder: str — where to read frame_img.png, arena_corners, obstacles
    :param spacing: int — dilation padding for obstacle mask
    :returns: (frame, arena, obstacles, start, targets, paths)
    """
    # 1) load world
    data = read_data(data_folder)
    if data is None:
        raise RuntimeError(f"Could not read data from '{data_folder}'")

    arena = [tuple(map(int, row)) for row in data['arena_corners']]
    obs   = [{"bbox": tuple(map(int, row))} for row in data['obstacles']]
    sx, sy, _ = data['robot_pos']
    if start is None:
        start = (int(sx), int(sy))

    # 2) load the image
    frame_path = os.path.join(data_folder, "frame_img.png")
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image at '{frame_path}'")
    h, w = frame.shape[:2]

    # 3) build planner and dilate obstacle mask
    planner = PathPlanner(obs, (h, w), arena)
    k = 2 * spacing + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    planner.mask = cv2.dilate(planner.mask, kernel)

    # 4) use the provided list directly
    targets = [(int(x), int(y)) for x, y in input_target_list]

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

    # 5) flatten and save
    improved_points = [pt for path in paths for pt in path]
    with open(output_target_path, "w") as f:
        for x, y in improved_points:
            f.write(f"{x},{y}\n")

    return frame, arena, [r["bbox"] for r in obs], start, targets, paths


def plot_trace(frame, arena, obstacles, start, targets, paths):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.imshow(img); ax.set_axis_off()

    # arena
    poly = Polygon(arena, closed=True, fill=False, edgecolor='yellow', linewidth=2)
    ax.add_patch(poly)
    # obstacles
    for x,y,w_,h_ in obstacles:
        ax.add_patch(Rectangle((x,y), w_, h_, facecolor='red', alpha=0.3))
    # start
    ax.plot(start[0], start[1], 'o', color='cyan', markersize=10, label='robot')
    # raw targets
    txs, tys = zip(*targets)
    ax.scatter(txs, tys, s=80, facecolors='none', edgecolors='white', label='targets')
    # paths
    for path in paths:
        xs, ys = zip(*path)
        ax.plot(xs, ys, '-', linewidth=2, color='lime')

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    DATA_FOLDER  = "Data"
    SPACING      = 20
    # example list of targets instead of a file
    input_list = [[250,50], [250,80], [350,300]]

    frame, arena, obstacles, start, targets, paths = trace_targets(
        input_target_list=input_list,
        output_target_path="Targets/improved_targets.txt",
        data_folder=DATA_FOLDER,
        spacing=SPACING
    )
    plot_trace(frame, arena, obstacles, start, targets, paths)
