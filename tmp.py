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
    spacing=20,
    out_path="Data/trace_overlay.png"
):
    # 1) load world
    data = read_data(data_folder)
    if data is None:
        raise RuntimeError(f"Could not read data from '{data_folder}'")

    arena = [tuple(map(int, row)) for row in data['arena_corners']]
    obs   = [{"bbox": tuple(map(int, row))} for row in data['obstacles']]

    robots = data['robot_pos']   # shape (N,4): [id,x,y,theta]
    if robots is not None and robots.shape[0] > 0:
        # if no robot_id provided, default to first
        if robot_id is None:
            rid, rx, ry, _ = robots[0]
            robot_id = int(rid)
        else:
            # get selected robot pos
            match = robots[np.where(robots[:,0].astype(int) == int(robot_id))]
            if match.shape[0] == 0:
                raise RuntimeError(f"Robot id {robot_id} not found in robot_pos.txt")
            rx, ry = match[0][1], match[0][2]

        # set start if not manually provided
        if start is None:
            start = (int(rx), int(ry))

    # 2) load the image
    frame_path = os.path.join(data_folder, "frame_img.png")
    frame = cv2.imread(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Could not load image at '{frame_path}'")
    h, w = frame.shape[:2]

    # 3) build planner and dilate obstacle mask
    planner = PathPlanner(obs, (h, w), arena)
    k = 2 * int(spacing) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    planner.mask = cv2.dilate(planner.mask, kernel)

    # 4) use the provided list directly
    # for a in input_target_list:
    #     print(a)
    targets = [(int(x), int(y)) for x,y in input_target_list]

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

    # --- plot & save ---
    # img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax.imshow(img_rgb)
    # ax.axis("off")

    # # arena boundary
    # ax.add_patch(Polygon(arena, closed=True, fill=False,
    #                      edgecolor='yellow', linewidth=2))

    # # obstacles
    # for x, y, w_, h_ in [r["bbox"] for r in obs]:
    #     ax.add_patch(Rectangle((x, y), w_, h_, facecolor='red', alpha=0.3))

    # # robot start
    # ax.plot(start[0], start[1], 'o', color='cyan', markersize=10, label='robot')

    # # raw targets
    # if targets:
    #     txs, tys = zip(*targets)
    #     ax.scatter(txs, tys, s=80, facecolors='none',
    #                edgecolors='white', label='targets')

    # # draw paths
    # for path in paths:
    #     xs, ys = zip(*path)
    #     ax.plot(int(xs), int(ys), '-', linewidth=2, color='lime')

    # ax.legend(loc='upper right')
    # plt.tight_layout()

    # fig.savefig(out_path, bbox_inches='tight')
    # plt.close(fig)

    # 5) flatten and save
    improved_points = [pt for path in paths for pt in path]
    with open(output_target_path, "w") as f:
        for x, y in improved_points:
            f.write(f"{x},{y}\n")

    return "Path Planned! and saved to {output_target_path}"

if __name__ == "__main__":
    DATA_FOLDER  = "Data"
    SPACING      = 20
    # example list of targets instead of a file
    input_list = [[458.0, 325.0], [449.0, 204.0]]#[[150,50], [200,80], [350,300]]

    # trace_targets(
    #     input_target_list=input_list,
    #     output_target_path="Targets/improved_targets.txt",
    #     data_folder=DATA_FOLDER,
    #     spacing=SPACING,
    #     out_path="Data/trace_overlay.png"
    # )

    gg={'input_target_list': [[458.0, 325.0], [449.0, 204.0]],
     'output_target_path': 'Targets/path.txt', 
     'out_path': 'Data/trace_overlay.png', 
     'data_folder': 'Data', 'spacing': 5.0}
    trace_targets(**gg)
    