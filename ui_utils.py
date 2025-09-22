import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import json
import math
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

def overlay_obstacles(
    frame,
    obstacles_path="Data/obstacles.txt",
    realtime_obstacles_path="Data/realtime_obstacles.txt",
    thickness=2
):
    """
    Draws thin rectangles for:
      - Static obstacles from `obstacles_path` in RED
      - Realtime obstacles from `realtime_obstacles_path` in BLUE

    Each line in the files should be:
      (x1,y1),(x2,y2),(x3,y3),(x4,y4)
    """
    overlay = frame.copy()

    def _draw_from_file(path, color):
        try:
            with open(path, "r") as f:
                for line in f:
                    s = line.strip().replace("(", "").replace(")", "")
                    if not s:
                        continue
                    parts = s.split(",")
                    if len(parts) != 8:
                        continue  # skip malformed lines
                    try:
                        nums = list(map(int, parts))
                    except ValueError:
                        continue
                    corners = [(nums[i], nums[i+1]) for i in range(0, 8, 2)]
                    cv2.polylines(
                        overlay,
                        [np.array(corners, dtype=np.int32)],
                        isClosed=True,
                        color=color,
                        thickness=thickness
                    )
        except FileNotFoundError:
            # Silent if file not present (common during startup); comment in if you want logs.
            # print(f"[WARN] Obstacles file not found: {path}")
            pass
        except Exception as e:
            print(f"[ERROR] Could not read obstacles from '{path}': {e}")

    # Static (red) and realtime (blue)
    _draw_from_file(obstacles_path, (0, 0, 255))       # red (BGR)
    _draw_from_file(realtime_obstacles_path, (255, 0, 0))  # blue (BGR)

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


def draw_path_on_frame(
    frame,
    json_path="Targets/paths.json",
    # Colors to cycle through for different robots (BGR, OpenCV order)
    colors=None,
    # Dotted line controls
    dot_gap_px=12,           # center-to-center spacing of dots
    dot_radius=2,            # radius of each dot
    # Markers
    show_checkpoints=True,   # mark delay>0 and action points ("open"/"close")
    checkpoint_color=(0, 255, 255),
    checkpoint_radius=4,
    action_open_color=(0, 255, 0),   # green
    action_close_color=(0, 0, 255),  # red
    action_radius=5,
    annotate_delay=False,    # if True, also label numbers/"open"/"close"
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.4,
    text_thickness=1,
):
    """
    Draw dotted polylines for ALL robots listed in paths.json:

    {
      "robots": [
        { "id": <int>, "path": [[x, y, tag], ...] },
        ...
      ]
    }

    The third element 'tag' can be:
      - a number (delay in ms), e.g. 5000
      - a string action "open" or "close"

    - Each robot's path is drawn as a dotted line with evenly spaced filled circles.
    - Dots are placed using 'dot_gap_px' across all segments.
    - If show_checkpoints is True:
        * numeric tag > 0 -> yellow checkpoint marker (+ optional numeric label)
        * "open" -> green action marker (+ optional text)
        * "close" -> red action marker (+ optional text)
    """
    if not os.path.exists(json_path):
        print(f"[draw_path_on_frame] JSON not found: {json_path}")
        return frame

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[draw_path_on_frame] Failed to read JSON: {e}")
        return frame

    robots = data.get("robots", [])
    if not isinstance(robots, list) or len(robots) == 0:
        return frame  # nothing to draw

    # Default color palette (BGR)
    if colors is None:
        colors = [
            (0, 255, 0),     # green
            (0, 0, 255),     # red
            (255, 0, 0),     # blue
            (255, 255, 0),   # cyan
            (255, 0, 255),   # magenta
            (0, 255, 255),   # yellow
            (0, 165, 255),   # orange
            (128, 0, 128),   # purple-ish
        ]

    gap = max(2, int(dot_gap_px))
    r = max(1, int(dot_radius))
    act_r = max(2, int(action_radius))

    def _draw_dotted_polyline(pts, color):
        """Draw dots across all segments with even spacing."""
        carry = 0.0
        for i in range(1, len(pts)):
            (x0, y0) = pts[i - 1]
            (x1, y1) = pts[i]
            dx, dy = (x1 - x0), (y1 - y0)
            seg_len = math.hypot(dx, dy)
            if seg_len <= 1e-6:
                continue

            dist_along = carry
            while dist_along <= seg_len:
                t = dist_along / seg_len
                xi = int(round(x0 + t * dx))
                yi = int(round(y0 + t * dy))
                cv2.circle(frame, (xi, yi), r, color, -1)
                dist_along += gap

            carry = dist_along - seg_len if dist_along > seg_len else 0.0

    def _parse_tag(raw):
        """
        Return ('delay', value) for numeric delays,
               ('action', 'open'|'close') for string actions,
               or (None, None) if not present/invalid.
        """
        if raw is None:
            return (None, None)
        # Try numeric first
        try:
            val = float(raw)
            return ("delay", val)
        except Exception:
            pass
        # Then action string
        if isinstance(raw, str):
            tag = raw.strip().lower()
            if tag in ("open", "close"):
                return ("action", tag)
        return (None, None)

    # Process each robot path
    for idx, entry in enumerate(robots):
        raw_path = entry.get("path", [])
        if not isinstance(raw_path, list) or len(raw_path) < 2:
            continue  # need at least two points

        # Parse points for this robot
        pts, tags = [], []  # tags: list of ('delay', value) or ('action', 'open'|'close') or (None, None)
        for item in raw_path:
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                continue
            try:
                x = int(item[0]); y = int(item[1])
            except Exception:
                continue
            kind, val = _parse_tag(item[2] if len(item) >= 3 else None)
            pts.append((x, y))
            tags.append((kind, val))

        if len(pts) < 2:
            continue

        color = colors[idx % len(colors)]
        _draw_dotted_polyline(pts, color)

        if not show_checkpoints:
            continue

        # Draw markers for this robot: delay>0 or actions
        for (x, y), (kind, val) in zip(pts, tags):
            if kind == "delay":
                try:
                    if float(val) > 0:
                        cv2.circle(frame, (x, y), checkpoint_radius, checkpoint_color, -1)
                        if annotate_delay:
                            cv2.putText(
                                frame, f"{float(val):g}", (x + 5, y - 5),
                                font, font_scale, checkpoint_color, text_thickness, cv2.LINE_AA
                            )
                except Exception:
                    pass
            elif kind == "action":
                if val == "open":
                    cv2.circle(frame, (x, y), act_r, action_open_color, 2)  # hollow circle
                    if annotate_delay:
                        cv2.putText(
                            frame, "open", (x + 5, y - 5),
                            font, font_scale, action_open_color, text_thickness, cv2.LINE_AA
                        )
                elif val == "close":
                    cv2.circle(frame, (x, y), act_r, action_close_color, -1)  # filled
                    if annotate_delay:
                        cv2.putText(
                            frame, "close", (x + 5, y - 5),
                            font, font_scale, action_close_color, text_thickness, cv2.LINE_AA
                        )

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
            current = (int(rx), int(ry))
            self.draw_overlay()

    app = App()
    app.mainloop()

    return result_message or "User didn't select any points"