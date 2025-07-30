#!/usr/bin/env python3
import os
import cv2
import numpy as np
from astar import PathPlanner
from Functions.Library.Agent.load_data import read_data

def point_selection(window_name='Point Selection',
                    image_name='live.jpg',
                    image_folder='.',
                    data_folder='Data',
                    button_height=50,
                    spacing=10):
    # 1) Load image
    img_path = os.path.join(image_folder, image_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not read '{image_name}' from '{image_folder}'.")
        return

    h, w = image.shape[:2]

    # 2) Load world data
    data = read_data(data_folder)
    if data is None:
        print(f"Error: Could not read data from '{data_folder}'.")
        return

    arena = [tuple(map(int, row)) for row in data['arena_corners']]
    obs   = [{"bbox": tuple(map(int, row))} for row in data['obstacles']]
    sx, sy, _ = data['robot_pos']
    current = (int(sx), int(sy))

    # 3) Build planner and dilate mask
    planner = PathPlanner(obs, (h, w), arena)
    k = 2 * spacing + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    planner.mask = cv2.dilate(planner.mask, kernel)

    # 4) Prepare canvas + buttons
    canvas = np.zeros((h + button_height, w, 3), dtype=np.uint8)
    canvas[:h] = image.copy()
    btn_w = w // 4
    save_btn  = ((w//8,         h + 5), (w//8 + btn_w,         h + button_height - 5))
    reset_btn = ((w - w//8 - btn_w, h + 5), (w - w//8, h + button_height - 5))

    points = []     # raw click targets
    paths  = []     # each A* segment

    def draw():
        # base image
        canvas[:h] = image.copy()
        # draw arena boundary
        cv2.polylines(canvas, [np.array(arena, np.int32)], True, (0,255,255), 2)
        # draw obstacles
        for x,y,ww,hh in [r['bbox'] for r in obs]:
            cv2.rectangle(canvas, (x,y), (x+ww, y+hh), (0,0,255), -1)
        # draw all paths
        for path in paths:
            for (x1,y1),(x2,y2) in zip(path, path[1:]):
                cv2.line(canvas, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        # draw current & clicked points
        cv2.circle(canvas, current, 6, (255,255,0), -1)
        for x,y in points:
            cv2.circle(canvas, (x,y), 5, (255,255,255), -1)
        # buttons
        cv2.rectangle(canvas, save_btn[0], save_btn[1],  (50,50,50), -1)
        cv2.putText(canvas, "Save",  (save_btn[0][0]+20, save_btn[1][1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.rectangle(canvas, reset_btn[0], reset_btn[1], (50,50,50), -1)
        cv2.putText(canvas, "Reset", (reset_btn[0][0]+15, reset_btn[1][1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow(window_name, canvas)

    def click_event(event, x, y, flags, param):
        nonlocal points, paths, current
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Save button
        if save_btn[0][0] <= x <= save_btn[1][0] and save_btn[0][1] <= y <= save_btn[1][1]:
            out_pts = os.path.join(data_folder, "improved_targets.txt")
            with open(out_pts, 'w') as f:
                for path in paths:
                    for px, py in path:
                        f.write(f"{int(px)},{int(py)}\n")
            print(f"Saved full path to {out_pts}")
            return

        # Reset button
        if reset_btn[0][0] <= x <= reset_btn[1][0] and reset_btn[0][1] <= y <= reset_btn[1][1]:
            points = []
            paths  = []
            current = (int(sx), int(sy))
            print("Reset all selections.")
            draw()
            return

        # Otherwise: click in image → plan segment
        if y < h:
            target = (x, y)
            pts_path = planner.find_obstacle_aware_path(current, target, 10)
            if not pts_path:
                print(f"⚠️ Unable to reach {target} from {current}")
            else:
                print(f"Path segment: {current} → {target} ({len(pts_path)} steps)")
                points.append(target)
                paths.append(pts_path)
                current = target
            draw()

    # window setup
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    draw()

    print("Click in the image to add targets.  Save or Reset via buttons below.  ESC to exit.")
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    print("Final target points:", points)
    return paths


if __name__ == "__main__":
    point_selection(window_name="Point Selection",
                    image_name="frame_img.png",
                    image_folder="Data",
                    data_folder="Data")
