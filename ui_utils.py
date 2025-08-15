import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import json
import os
import cv2
from PIL import Image, ImageTk, ImageOps
import numpy as np
from astar import PathPlanner
from Functions.Library.Agent.load_data import read_data

class CTkMessageBox(ctk.CTkToplevel):
    def __init__(self, parent, title, message, text_color):
        super().__init__(parent)
        self.title(title)
        self.geometry("300x120")
        self.resizable(False, False)
        self.transient(parent)
        self.update_idletasks()
        try:
            self.grab_set()
        except Exception:
            pass
        self.focus_force()
        print(f"[]")
        ctk.CTkLabel(self, text=message, text_color=text_color).pack(pady=(20, 20), padx=20)
        ctk.CTkButton(self, text="OK", command=self.destroy).pack(pady=(0, 5))
        self.wait_window()

class CheckGroup(ctk.CTkFrame):
    def __init__(self, master, label, suboptions, selected=None, **kwargs):
        super().__init__(master, **kwargs)
        self.checked = ctk.BooleanVar()
        self.sub_vars = []
        self.label = label
        self.main_checkbox = ctk.CTkCheckBox(self, text=label, variable=self.checked, command=self.on_main_toggle)
        self.main_checkbox.pack(anchor="w", pady=(0, 2))
        self.sub_frame = ctk.CTkFrame(self)
        self.sub_frame.pack(anchor="w", padx=20)
        for sublabel in suboptions:
            var = ctk.BooleanVar(value=(selected is not None and sublabel in selected))
            chk = ctk.CTkCheckBox(self.sub_frame, text=sublabel, variable=var, command=self.on_sub_toggle)
            chk.pack(anchor="w")
            self.sub_vars.append((sublabel, var))
        self.checked.set(all(var.get() for _, var in self.sub_vars))

    def on_main_toggle(self):
        new_state = self.checked.get()
        for _, var in self.sub_vars:
            var.set(new_state)

    def on_sub_toggle(self):
        all_checked = all(var.get() for _, var in self.sub_vars)
        self.checked.set(all_checked)

    def get_selected(self):
        return {
            "group": self.checked.get(),
            "options": {label: var.get() for label, var in self.sub_vars}
        }

def get_app_settings():
    settings_path = "Settings/settings.json"
    default_settings = {
        "arena_width": "800",
        "arena_height": "800",
        "corner_prompt": "detect four circular markers",
        "obstacle_prompt": "detect black rectangular obstacles"
    }
    try:
        with open(settings_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_settings

def open_settings_popup(app):
    settings_path = "Settings/settings.json"
    os.makedirs(os.path.dirname(settings_path), exist_ok=True)
    default_settings = {
        "aruco_id": "782",
        "arena_width": "800",
        "arena_height": "800",
        "corner_prompt": "detect four circular markers",
        "obstacle_prompt": "detect black rectangular obstacles"
    }
    try:
        with open(settings_path, "r") as f:
            settings = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        settings = default_settings

    popup = ctk.CTkToplevel(app)
    popup.title("Settings")
    popup.geometry("400x450")
    popup.resizable(False, False)
    app.settings_popup = popup

    def close_popup():
        popup.destroy()
        app.settings_popup = None
    popup.protocol("WM_DELETE_WINDOW", close_popup)

    vars = {
        "aruco_id": ctk.StringVar(value=settings.get("aruco_id", default_settings["aruco_id"])),
        "arena_width": ctk.StringVar(value=settings.get("arena_width", default_settings["arena_width"])),
        "arena_height": ctk.StringVar(value=settings.get("arena_height", default_settings["arena_height"])),
        "corner_prompt": ctk.StringVar(value=settings.get("corner_prompt", default_settings["corner_prompt"])),
        "obstacle_prompt": ctk.StringVar(value=settings.get("obstacle_prompt", default_settings["obstacle_prompt"])),
    }
    ctk.CTkLabel(popup, text="Robot ArUco ID:").pack(anchor="w", padx=20, pady=(10, 0))
    ctk.CTkEntry(popup, textvariable=vars["aruco_id"]).pack(fill="x", padx=20, pady=5)
    ctk.CTkLabel(popup, text="Arena Width:").pack(anchor="w", padx=20, pady=(10, 0))
    ctk.CTkEntry(popup, textvariable=vars["arena_width"]).pack(fill="x", padx=20, pady=5)
    ctk.CTkLabel(popup, text="Arena Height:").pack(anchor="w", padx=20, pady=(10, 0))
    ctk.CTkEntry(popup, textvariable=vars["arena_height"]).pack(fill="x", padx=20, pady=5)
    ctk.CTkLabel(popup, text="Arena Corner Prompt:").pack(anchor="w", padx=20, pady=(10, 0))
    ctk.CTkEntry(popup, textvariable=vars["corner_prompt"]).pack(fill="x", padx=20, pady=5)
    ctk.CTkLabel(popup, text="Obstacles Prompt:").pack(anchor="w", padx=20, pady=(10, 0))
    ctk.CTkEntry(popup, textvariable=vars["obstacle_prompt"]).pack(fill="x", padx=20, pady=5)

    btn_frame = ctk.CTkFrame(popup)
    btn_frame.pack(pady=20)
    def save_settings():
        new_settings = {k: v.get().strip() for k, v in vars.items()}
        try:
            with open(settings_path, "w") as f:
                json.dump(new_settings, f, indent=4)
            close_popup()
        except Exception as e:
            CTkMessageBox(popup, "Save Error", str(e), "red")

    ctk.CTkButton(btn_frame, text="Save", command=save_settings).pack(side="left", padx=10)
    ctk.CTkButton(btn_frame, text="Cancel", command=close_popup).pack(side="right", padx=10)

def overlay_obstacles(frame, obstacles_path="Data/obstacles.txt"):
    overlay = frame.copy()

    try:
        with open(obstacles_path, "r") as f:
            for line in f:
                line = line.strip().replace("(", "").replace(")", "")
                parts = list(map(int, line.split(",")))
                if len(parts) == 8:
                    corners = [(parts[i], parts[i+1]) for i in range(0, 8, 2)]
                    cv2.fillPoly(overlay, [np.array(corners, dtype=np.int32)], (0, 0, 255))
    except Exception as e:
        print(f"[ERROR] Could not read obstacles from '{obstacles_path}': {e}")
    return overlay

def show_frame_with_overlay(parent, frame, arena_path="Data/arena_corners.txt", obstacles_path="Data/obstacles.txt"):
    overlay = frame.copy()

    # 1. Draw arena (green polygon)
    try:
        with open(arena_path, "r") as f:
            arena_pts = [list(map(int, line.strip().split(","))) for line in f if line.strip()]
        if len(arena_pts) == 4:
            pts = np.array(arena_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        else:
            print("[WARN] Arena corners file does not contain exactly 4 points.")
    except Exception as e:
        print(f"[ERROR] Reading arena corners failed: {e}")

    # 2. Draw obstacles (red polygons from corner format)
    try:
        with open(obstacles_path, "r") as f:
            for line in f:
                line = line.strip().replace("(", "").replace(")", "")
                parts = list(map(int, line.split(",")))
                if len(parts) != 8:
                    continue
                corners = [(parts[i], parts[i+1]) for i in range(0, 8, 2)]
                cv2.fillPoly(overlay, [np.array(corners, dtype=np.int32)], (0, 0, 255))
    except Exception as e:
        print(f"[ERROR] Reading obstacles failed: {e}")

    # 3. Convert to ImageTk-compatible format
    rgb_img = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    ctk_img = ctk.CTkImage(light_image=pil_img, size=pil_img.size)

    # 4. Create popup
    popup = ctk.CTkToplevel(parent)
    popup.title("Arena Preview")
    popup.geometry(f"{pil_img.width}x{pil_img.height}")
    label = ctk.CTkLabel(popup, text="", image=ctk_img)
    label.pack(expand=True, fill="both")
    label.image = ctk_img  # prevent garbage collection

    popup.lift()
    popup.focus()
    popup.grab_set()


def get_overlay_frame(
    warp_size=(640, 640),
    img_path="Data/frame_img.png",
    arena_path="Data/arena_corners.txt",
    obstacles_path="Data/obstacles.txt",
    path_file="Targets/path.txt"
):
    overlay = cv2.imread(img_path)
    if overlay is None:
        print(f"[get_overlay_frame] Warning: Could not read image from '{img_path}'")
        # Create a blank fallback image
        w, h = warp_size
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

    def sort_corners(pts):
        """Sorts 4 corner points: [TL, TR, BR, BL]"""
        pts = np.array(pts, dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],      # Top-left
            pts[np.argmin(d)],      # Top-right
            pts[np.argmax(s)],      # Bottom-right
            pts[np.argmax(d)]       # Bottom-left
        ], dtype=np.float32)

    # Read and draw arena
    try:
        with open(arena_path, "r") as f:
            arena_pts = [list(map(int, line.strip().split(","))) for line in f if line.strip()]
        if len(arena_pts) == 4:
            cv2.polylines(overlay, [np.array(arena_pts, dtype=np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
            src = sort_corners(arena_pts)
        else:
            print("[get_overlay_frame] Arena corners not 4 points.")
            return overlay
    except Exception as e:
        print(f"[get_overlay_frame] Arena read error: {e}")
        return overlay

    # Draw obstacles (filled red polygons from corner format)
    try:
        with open(obstacles_path, "r") as f:
            for line in f:
                line = line.strip().replace("(", "").replace(")", "")
                parts = list(map(int, line.split(",")))
                if len(parts) == 8:
                    corners = [(parts[i], parts[i+1]) for i in range(0, 8, 2)]
                    cv2.fillPoly(overlay, [np.array(corners, dtype=np.int32)], (0, 0, 255))
    except Exception as e:
        print(f"[get_overlay_frame] Obstacle read error: {e}")

    # Draw path (green lines)
    try:
        if os.path.exists(path_file):
            with open(path_file, "r") as f:
                pts = [tuple(map(int, line.strip().split(","))) for line in f if "," in line]
            for i in range(1, len(pts)):
                cv2.line(overlay, pts[i - 1], pts[i], (0, 255, 0), 2)
    except Exception as e:
        print(f"[get_overlay_frame] Path read error: {e}")

    # Warp to top-down view
    try:
        w, h = warp_size
        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(overlay, M, (w, h))
        return warped
    except Exception as e:
        print(f"[get_overlay_frame] Warp error: {e}")
        return overlay

# def draw_path_on_frame(frame, path_file="path.txt", color=(0, 255, 0), thickness=2):
#     if not os.path.exists(path_file):
#         print(f"[draw_path_on_frame] Path file not found: {path_file}")
#         return frame

#     try:
#         with open(path_file, "r") as f:
#             pts = [tuple(map(int, line.strip().split(","))) for line in f if "," in line]

#         if len(pts) < 3:
#             return frame  # nothing to draw

#         for i in range(1, len(pts)):
#             cv2.line(frame, pts[i - 1], pts[i], color, thickness)

#     except Exception as e:
#         print(f"[draw_path_on_frame] Error: {e}")

#     return frame

import os
import cv2

def draw_path_on_frame(
    frame,
    path_file="path.txt",
    color=(0, 255, 0),
    thickness=2,
    show_checkpoints=True,             # set True to mark delay>0 points
    checkpoint_color=(0, 255, 255),
    checkpoint_radius=4,
    annotate_delay=False,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.4,
    text_thickness=1
):
    """
    Draws a polyline from a path file where each line is:
        x,y,delay
    - Only (x,y) are used to draw the path.
    - If show_checkpoints is True, points with delay>0 are highlighted/annotated.
    """
    if not os.path.exists(path_file):
        print(f"[draw_path_on_frame] Path file not found: {path_file}")
        return frame

    pts = []       # list[(x, y)]
    delays = []    # list[float]

    try:
        with open(path_file, "r", encoding="utf-8") as f:
            for ln, raw in enumerate(f, 1):
                line = raw.strip()
                if not line:
                    continue

                # Accept comma- or space-separated
                parts = [p for p in line.replace(",", " ").split() if p]
                if len(parts) < 2:
                    # malformed line; skip
                    continue

                try:
                    x = int(float(parts[0]))
                    y = int(float(parts[1]))
                    d = float(parts[2]) if len(parts) >= 3 else 0.0
                except ValueError:
                    print(f"[draw_path_on_frame] Skipping bad line {ln}: '{raw.rstrip()}'")
                    continue

                pts.append((x, y))
                delays.append(d)

        if len(pts) < 2:
            return frame  # nothing to draw

        # Draw the polyline
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, thickness)

        # Optionally mark checkpoints (delay > 0)
        if show_checkpoints:
            for (x, y), d in zip(pts, delays):
                if d > 0:
                    cv2.circle(frame, (x, y), checkpoint_radius, checkpoint_color, -1)
                    if annotate_delay:
                        cv2.putText(
                            frame, f"{d:g}", (x + 5, y - 5),
                            font, font_scale, checkpoint_color, text_thickness, cv2.LINE_AA
                        )

    except Exception as e:
        print(f"[draw_path_on_frame] Error: {e}")

    return frame


def get_arena_dimensions(settings_path="Settings/settings.json"):
    DEFAULT_WIDTH = 1200
    DEFAULT_HEIGHT = 900
    try:
        with open(settings_path, "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("Warning: settings.json not found. Using default arena size.")
        settings = {}
    except json.JSONDecodeError as e:
        print(f"Warning: could not parse settings.json ({e}). Using default arena size.")
        settings = {}
    arena_width = int(settings.get("arena_width", str(DEFAULT_WIDTH)))
    arena_height = int(settings.get("arena_height", str(DEFAULT_HEIGHT)))

    return (arena_width, arena_height)

def point_selection(data_folder='Data',
                    output_target_path='Targets/path.txt',
                    spacing=25):
    """
    Launch a CustomTkinter GUI for point selection and path planning.
    Supports 4-corner polygon obstacles.
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
    polygon_obs = [ [tuple(map(int, pt)) for pt in poly] for poly in data['obstacles'] ]
    sx, sy, _ = data['robot_pos']
    current = (int(sx), int(sy))

    # convert to bounding boxes for PathPlanner
    obs = []
    for poly in polygon_obs:
        x, y, w_, h_ = cv2.boundingRect(np.array(poly))
        obs.append({"bbox": (x, y, w_, h_)})

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

            # arena
            cv2.polylines(img, [np.array(arena, np.int32)], True, (0,255,255), 2)

            # inner boundary
            for i in range(len(inner_boundary)):
                cv2.line(img,
                         inner_boundary[i],
                         inner_boundary[(i+1)%len(inner_boundary)],
                         (255,255,0), 1, cv2.LINE_AA)

            # draw polygon obstacles
            for poly in polygon_obs:
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(img, [pts], (0,0,255))
                cv2.polylines(img, [pts], isClosed=True, color=(255,255,255), thickness=1)

            # draw paths
            for path in paths:
                for (x1,y1),(x2,y2) in zip(path, path[1:]):
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

            # draw robot and points
            cv2.circle(img, current, 6, (255,255,0), -1)
            for px,py in points:
                cv2.circle(img, (px,py), 5, (255,255,255), -1)

            self.overlay = ImageTk.PhotoImage(
                Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.overlay, anchor="nw", tags="overlay")

        def on_click(self, event):
            nonlocal points, paths, current
            x, y = event.x, event.y
            segment = planner.find_obstacle_aware_path(current, (x,y), 10)
            if segment:
                points.append((x,y))
                paths.append(segment)
                current = (x,y)
            else:
                print(f"Unable to reach {(x,y)} from {current}")
            self.draw_overlay()

        def save(self):
            nonlocal result_message
            with open(output_target_path, 'w') as f:
                for p in paths:
                    for ux,uy in p:
                        f.write(f"{int(ux)},{int(uy)}\n")
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