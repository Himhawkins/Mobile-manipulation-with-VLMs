#!/usr/bin/env python3
import os
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from astar import PathPlanner
from Functions.Library.Agent.load_data import read_data

def point_track(data_folder='Data',
                output_target_path='Targets/path.txt',
                spacing=25):
    """
    Launch a CustomTkinter GUI for point selection and path planning.
    Returns a status message upon Save or "User didn't select any points" on close.
    """
    img_path = os.path.join(data_folder, "frame_img.png")
    frame = cv2.imread(img_path)
    if frame is None:
        return f"Error: Could not read '{img_path}'"
    h, w = frame.shape[:2]

    data = read_data(data_folder)
    if data is None:
        return f"Error: Could not read data from '{data_folder}'"
    
    arena = [tuple(map(int, row)) for row in data['arena_corners']]
    obs = [{"corners": [tuple(map(int, pt)) for pt in row]} for row in data['obstacles']]
    sx, sy, _ = data['robot_pos']
    current = (int(sx), int(sy))

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

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    points = []
    paths = []
    result_message = None

    class App(ctk.CTk):
        def __init__(self):
            super().__init__()
            self.title("Point Selection")
            self.canvas = tk.Canvas(self, width=w, height=h)
            self.canvas.pack()
            self.tk_img = ImageTk.PhotoImage(pil)
            self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw", tags="bg")

            btn_frame = ctk.CTkFrame(self)
            btn_frame.pack(fill="x", pady=5)
            self.save_btn = ctk.CTkButton(btn_frame, text="Save", command=self.save)
            self.save_btn.pack(side="left", padx=20)
            self.reset_btn = ctk.CTkButton(btn_frame, text="Reset", command=self.reset)
            self.reset_btn.pack(side="right", padx=20)

            self.canvas.bind("<Button-1>", self.on_click)
            self.draw_overlay()

        def draw_overlay(self):
            self.canvas.delete("overlay")
            img = frame.copy()
            # draw arena
            cv2.polylines(img, [np.array(arena, np.int32)], True, (0,255,255), 2)
            # draw inner boundary
            for i in range(len(inner_boundary)):
                cv2.line(img, inner_boundary[i], inner_boundary[(i+1)%len(inner_boundary)],
                         (255,255,0), 1, cv2.LINE_AA)

            # draw polygonal obstacles + spacing visualization
            for obs_poly in obs:
                corners = np.array(obs_poly['corners'], dtype=np.int32)
                cv2.fillPoly(img, [corners], (0,0,255))

                # draw dotted spacing box around polygon's bbox for visual feedback
                x, y, w_box, h_box = cv2.boundingRect(corners)
                tl = (x - spacing, y - spacing)
                br = (x + w_box + spacing, y + h_box + spacing)
                for dx in range(tl[0], br[0], 10):
                    cv2.line(img, (dx, tl[1]), (dx+5, tl[1]), (0,255,255), 1)
                    cv2.line(img, (dx, br[1]), (dx+5, br[1]), (0,255,255), 1)
                for dy in range(tl[1], br[1], 10):
                    cv2.line(img, (tl[0], dy), (tl[0], dy+5), (0,255,255), 1)
                    cv2.line(img, (br[0], dy), (br[0], dy+5), (0,255,255), 1)

            # draw path lines
            for path in paths:
                for (x1, y1), (x2, y2) in zip(path, path[1:]):
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

            # draw current position & points
            cv2.circle(img, current, 6, (255,255,0), -1)
            for px, py in points:
                cv2.circle(img, (px, py), 5, (255,255,255), -1)

            self.overlay = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
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
                print(f"⚠️ Unable to reach {(x, y)} from {current}")
            self.draw_overlay()

        def save(self):
            nonlocal result_message
            with open(output_target_path, 'w') as f:
                for path in paths:
                    for px, py in path:
                        f.write(f"{int(px)},{int(py)}\n")
            result_message = f"Path Planned! and saved to {output_target_path}"
            self.destroy()

        def reset(self):
            nonlocal points, paths, current
            points.clear()
            paths.clear()
            current = (int(sx), int(sy))
            self.draw_overlay()

    app = App()
    app.mainloop()
    return result_message or "User didn't select any points"


if __name__ == "__main__":
    msg = point_track()
    print(msg)
