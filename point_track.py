#!/usr/bin/env python3
import os
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from astar import PathPlanner
from Functions.Library.Agent.load_data import read_data
import argparse

def point_track(data_folder='Data',
                output_target_path='Targets/path.txt',
                spacing=50,
                delay=0):  # <- NEW
    """
    Launch a CustomTkinter GUI for point selection and path planning.
    Returns a status message upon Save or "User didn't select any points" on close.
    """
    # --- load data ---
    img_path = os.path.join(data_folder, "frame_img.png")
    frame = cv2.imread(img_path)
    if frame is None:
        return f"Error: Could not read '{img_path}'"
    h, w = frame.shape[:2]

    data = read_data(data_folder)
    if data is None:
        return f"Error: Could not read data from '{data_folder}'"
    arena = [tuple(map(int, row)) for row in data['arena_corners']]

    # Use 4-corner obstacle polygons directly
    obs = [{"corners": [tuple(pt) for pt in row]} for row in data['obstacles']]

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

        # set current if not manually provided
        if current is None:
            current = (int(rx), int(ry))

    # sx, sy, _ = data['robot_pos']
    # current = (int(sx), int(sy))

    spacing = int(spacing)

    # build planner
    planner = PathPlanner(obs, (h, w), arena)
    k = 2 * spacing + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    planner.mask = cv2.dilate(planner.mask, kernel)

    # compute inner boundary
    arena_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(arena_mask, [np.array(arena, np.int32)], 255)
    eroded = cv2.erode(arena_mask, kernel)
    ctrs, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ctrs:
        inner_boundary = arena
    else:
        large = max(ctrs, key=cv2.contourArea)
        approx = cv2.approxPolyDP(large, spacing, True)
        inner_boundary = [tuple(pt[0]) for pt in approx]

    # convert frame for display
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # state
    points = []
    paths = []
    result_message = None

    class App(ctk.CTk):
        def __init__(self):
            super().__init__()
            self.title("Point Selection")
            # canvas
            self.canvas = tk.Canvas(self, width=w, height=h)
            self.canvas.pack()
            self.tk_img = ImageTk.PhotoImage(pil)
            self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw", tags="bg")
            # buttons
            btn_frame = ctk.CTkFrame(self)
            btn_frame.pack(fill="x", pady=5)
            self.save_btn = ctk.CTkButton(btn_frame, text="Save", command=self.save)
            self.save_btn.pack(side="left", padx=20)
            self.reset_btn = ctk.CTkButton(btn_frame, text="Reset", command=self.reset)
            self.reset_btn.pack(side="right", padx=20)
            # mouse click
            self.canvas.bind("<Button-1>", self.on_click)
            self.draw_overlay()

        def draw_overlay(self):
            self.canvas.delete("overlay")
            img = frame.copy()

            # draw arena
            cv2.polylines(img, [np.array(arena, np.int32)], True, (0, 255, 255), 2)

            # draw inner boundary
            for i in range(len(inner_boundary)):
                cv2.line(img,
                        inner_boundary[i],
                        inner_boundary[(i + 1) % len(inner_boundary)],
                        (255, 255, 0), 1, cv2.LINE_AA)

            # draw obstacles + offset polygons
            for ob in obs:
                corners = np.array(ob["corners"], dtype=np.int32)

                # Draw original obstacle in red
                cv2.fillPoly(img, [corners], (0, 0, 255))

                # Create mask of the obstacle
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [corners], 255)

                # Dilate the mask to get padding area
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * spacing + 1, 2 * spacing + 1))
                dilated = cv2.dilate(mask, kernel)

                # Find contours of the dilated mask
                contours, _ = cv2.findContours(
                    dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    offset_poly = cv2.approxPolyDP(contours[0], epsilon=2.0, closed=True)
                    cv2.polylines(img, [offset_poly], isClosed=True, color=(0, 255, 255), thickness=2)

            # draw paths
            for path in paths:
                for (x1, y1), (x2, y2) in zip(path, path[1:]):
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # draw robot & points
            cv2.circle(img, current, 6, (255, 255, 0), -1)
            for px, py in points:
                cv2.circle(img, (px, py), 5, (255, 255, 255), -1)

            self.overlay = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.overlay, anchor="nw", tags="overlay")


        def on_click(self, event):
            nonlocal points, paths, current
            x, y = event.x, event.y
            segment = planner.find_obstacle_aware_path(current, (x, y), 10)
            if segment:
                points.append((x, y))
                paths.append(segment)
                current = (x, y)
            else:
                print(f"Unable to reach {(x,y)} from {current}")
            self.draw_overlay()

        def save(self):
            nonlocal result_message
            with open(output_target_path, 'w') as f:
                for p in paths:
                    for i, (ux, uy) in enumerate(p):
                        is_main_checkpoint = (i == len(p) - 1)
                        this_delay = delay if is_main_checkpoint else 0
                        f.write(f"{int(ux)},{int(uy)},{int(this_delay)}\n")  # now writing x,y,delay
            result_message = f"Path Planned! and saved to {output_target_path}"
            self.destroy()


        def reset(self):
            nonlocal points, paths, current
            points.clear()
            paths.clear()
            current = (int(rx), int(ry))
            self.draw_overlay()

    app = App()
    app.mainloop()
    return result_message or "User didn't select any points"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Point selection GUI")
    parser.add_argument('--data_folder', default='Data')
    parser.add_argument('--output_target_path', default='Targets/path.txt')
    parser.add_argument('--spacing', type=float, default=25)
    parser.add_argument('--delay', type=float, default=0, help="Delay (ms) at main checkpoints")  # <- NEW
    args = parser.parse_args()

    msg = point_track(
        data_folder=args.data_folder,
        output_target_path=args.output_target_path,
        spacing=args.spacing,
        delay=args.delay  # <- NEW
    )
    print(msg)
