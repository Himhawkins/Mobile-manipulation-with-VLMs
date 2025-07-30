#!/usr/bin/env python3
import os
import cv2
import numpy as np
from astar import PathPlanner
from Functions.Library.Agent.load_data import read_data

def point_selection(data_folder='Data',
                    output_target_path='Targets/path.txt',
                    spacing=30):
    window_name = 'Point Selection'
    # 1) load image
    img_path = os.path.join(data_folder, "frame_img.png")
    image   = cv2.imread(img_path)
    if image is None:
        print(f"Error: Could not read '{img_path}'.")
        return
    h, w = image.shape[:2]

    # 2) load world data
    data = read_data(data_folder)
    if data is None:
        print(f"Error: Could not read data from '{data_folder}'.")
        return
    arena = [tuple(map(int, row)) for row in data['arena_corners']]
    obs   = [{"bbox": tuple(map(int, row))} for row in data['obstacles']]
    sx, sy, _ = data['robot_pos']
    current   = (int(sx), int(sy))

    # 3) build planner + dilate mask for path planning
    planner = PathPlanner(obs, (h, w), arena)
    k       = 2 * spacing + 1
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    planner.mask = cv2.dilate(planner.mask, kernel)

    # 4) compute inner (eroded) arena boundary
    arena_mask   = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(arena_mask, [np.array(arena, np.int32)], 255)
    eroded_arena = cv2.erode(arena_mask, kernel)
    ctrs, _      = cv2.findContours(eroded_arena,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not ctrs:
        inner_boundary = arena
    else:
        large  = max(ctrs, key=cv2.contourArea)
        approx = cv2.approxPolyDP(large, spacing, True)
        inner_boundary = [tuple(pt[0]) for pt in approx]

    # 5) prepare canvas & buttons
    button_height = 50
    canvas        = np.zeros((h + button_height, w, 3), dtype=np.uint8)
    canvas[:h]    = image.copy()
    btn_w         = w // 4
    save_btn      = ((w//8,           h + 5),
                     (w//8 + btn_w,  h + button_height - 5))
    reset_btn     = ((w - w//8 - btn_w, h + 5),
                     (w - w//8,          h + button_height - 5))

    points = []
    paths  = []
    done   = False
    result_message = None

    # 6) helper to draw dashed lines
    def draw_dashed_line(img, pt1, pt2, color,
                         thickness=1, dash_len=5, gap_len=5):
        x1, y1 = pt1; x2, y2 = pt2
        dist   = int(np.hypot(x2-x1, y2-y1))
        if dist == 0:
            return
        dx, dy = (x2-x1)/dist, (y2-y1)/dist
        drawn  = 0
        while drawn < dist:
            start = int(drawn)
            end   = int(min(dist, drawn + dash_len))
            xa = int(x1 + dx*start); ya = int(y1 + dy*start)
            xb = int(x1 + dx*end);   yb = int(y1 + dy*end)
            cv2.line(img, (xa, ya), (xb, yb), color, thickness)
            drawn += dash_len + gap_len

    # 7) main draw routine
    def draw():
        canvas[:h] = image.copy()

        # a) solid arena boundary
        cv2.polylines(canvas, [np.array(arena, np.int32)],
                      True, (0,255,255), 2)

        # b) inner dotted arena (eroded inward)
        for i in range(len(inner_boundary)):
            p1 = inner_boundary[i]
            p2 = inner_boundary[(i+1) % len(inner_boundary)]
            draw_dashed_line(canvas, p1, p2,
                             color=(255,255,0),
                             thickness=1, dash_len=8, gap_len=8)

        # c) obstacles + their outer spacing
        for x, y, ww, hh in [r['bbox'] for r in obs]:
            cv2.rectangle(canvas, (x, y), (x+ww, y+hh),
                          (0,0,255), -1)
            tl = (x - spacing,     y - spacing)
            br = (x + ww + spacing, y + hh + spacing)
            draw_dashed_line(canvas, tl, (br[0], tl[1]),
                             (0,255,255), 1, 6, 6)
            draw_dashed_line(canvas, (br[0], tl[1]), br,
                             (0,255,255), 1, 6, 6)
            draw_dashed_line(canvas, br, (tl[0], br[1]),
                             (0,255,255), 1, 6, 6)
            draw_dashed_line(canvas, (tl[0], br[1]), tl,
                             (0,255,255), 1, 6, 6)

        # d) draw all planned paths
        for path in paths:
            for (x1,y1), (x2,y2) in zip(path, path[1:]):
                cv2.line(canvas, (int(x1),int(y1)),
                         (int(x2),int(y2)), (0,255,0), 2)

        # e) robot start & clicked points
        cv2.circle(canvas, current, 6, (255,255,0), -1)
        for px, py in points:
            cv2.circle(canvas, (px,py), 5, (255,255,255), -1)

        # f) Save / Reset buttons
        cv2.rectangle(canvas, save_btn[0], save_btn[1],
                      (50,50,50), -1)
        cv2.putText(canvas, "Save",
                    (save_btn[0][0]+20, save_btn[1][1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 2)
        cv2.rectangle(canvas, reset_btn[0], reset_btn[1],
                      (50,50,50), -1)
        cv2.putText(canvas, "Reset",
                    (reset_btn[0][0]+15, reset_btn[1][1]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 2)

        cv2.imshow(window_name, canvas)

    # 8) click logic
    def click_event(event, x, y, flags, param):
        nonlocal points, paths, current, done, result_message
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Save
        if (save_btn[0][0] <= x <= save_btn[1][0]
            and save_btn[0][1] <= y <= save_btn[1][1]):
            with open(output_target_path, 'w') as f:
                for path in paths:
                    for ux, uy in path:
                        f.write(f"{int(ux)},{int(uy)}\n")
            result_message = (
                f"Path Planned! and saved to {output_target_path}"
            )
            done = True
            return

        # Reset
        if (reset_btn[0][0] <= x <= reset_btn[1][0]
            and reset_btn[0][1] <= y <= reset_btn[1][1]):
            points, paths = [], []
            current = (int(sx), int(sy))
            print("Reset all selections.")
            draw()
            return

        # Plan segment
        if y < h:
            target = (x, y)
            segment = planner.find_obstacle_aware_path(current, target, 10)
            if not segment:
                print(f"Unable to reach {target} from {current}")
            else:
                print(f"Path segment: {current} → {target}"
                      f" ({len(segment)} steps)")
                points.append(target)
                paths.append(segment)
                current = target
            draw()

    # 9) start window
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    draw()

    print("Click to add targets; Save or Reset with buttons; ESC to exit.")
    while True:
        # 1) if they clicked “Save”
        if done:
            break

        # 2) if they clicked the window “X” → getWindowProperty might throw
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            # window no longer exists
            break

        # 3) if they pressed ESC
        if cv2.waitKey(50) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    # 10) return appropriate message
    if result_message:
        return result_message
    else:
        return "User didn't select any points"


if __name__ == "__main__":
    msg = point_selection()
    print(msg)
