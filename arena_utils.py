import os
import json
import customtkinter as ctk
from customtkinter import CTkImage
import cv2
import time
import numpy as np
from PIL import Image, ImageTk

SETTINGS_PATH = "Settings/settings.json"
ARENA_SETTINGS_PATH = "Settings/arena_settings.json"

def refresh_cameras(max_index=10, current_index=None):
    """
    Returns a list of camera labels like ['Camera 0', 'Camera 2'],
    including the currently active camera even if it appears busy.
    """
    working = []

    for i in range(max_index):
        path = f"/dev/video{i}"
        if not os.path.exists(path):
            continue
        if i == current_index:
            # Current camera is already open and working
            working.append(f"{i}")
            continue
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        ret, _ = cap.read()
        cap.release()
        if ret:
            working.append(f"{i}")

    return working

def load_arena_settings():
    default_settings = {
        "rows": 1,
        "cols": 1,
        "cells": {}
    }

    if not os.path.exists(ARENA_SETTINGS_PATH):
        os.makedirs(os.path.dirname(ARENA_SETTINGS_PATH), exist_ok=True)
        with open(ARENA_SETTINGS_PATH, "w") as f:
            json.dump(default_settings, f, indent=2)
        return default_settings

    try:
        with open(ARENA_SETTINGS_PATH, "r") as f:
            data = json.load(f)
            # Validate and merge with defaults
            settings = default_settings.copy()
            settings.update({k: data.get(k, v) for k, v in default_settings.items()})
            return settings
    except json.JSONDecodeError:
        # fallback if corrupted
        return default_settings

def open_all_cameras(
    settings,
    test_read=True,
    warmup_frames=3,
    width=1280,
    height=960,
    fps=15,
    retries=3,
    reopen_once=True
):
    """
    Opens only the camera IDs referenced by the active grid in `settings`.
    Uses MJPEG to reduce bandwidth, enforces width/height/fps, and performs
    warm-up + retries so multi-cam (3+) setups are more reliable.

    Returns:
        caps: dict[int, cv2.VideoCapture]
    """
    rows = int(settings.get("rows", 0))
    cols = int(settings.get("cols", 0))
    cells = settings.get("cells", {})

    def _open_cam(cam_id):
        cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)

        # Prefer MJPEG (compressed) to reduce USB bandwidth pressure.
        # Not all cameras honor this, but setting it won't hurt.
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

        # Target resolution & FPS (many cams clamp to nearest supported values)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)

        # Keep buffer tiny to avoid lag (may be ignored by some drivers)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        return cap

    def _try_read(cap, attempts=3, sleep_s=0.02):
        """Try a few reads to confirm frames are flowing."""
        ok = False
        for _ in range(attempts):
            ret, _ = cap.read()
            if ret:
                ok = True
                break
            time.sleep(sleep_s)
        return ok

    # Collect only the cameras referenced by the active grid
    needed_camera_ids = set()
    for r in range(rows):
        for c in range(cols):
            key = f"{r},{c}"
            cell = cells.get(key)
            if not cell or "camera" not in cell:
                continue
            try:
                cam_id = int(cell["camera"])
            except (TypeError, ValueError):
                continue
            needed_camera_ids.add(cam_id)

    caps = {}
    for cam_id in sorted(needed_camera_ids):
        cap = None
        opened = False

        for attempt in range(retries):
            # First open
            cap = _open_cam(cam_id)
            if not cap.isOpened():
                if cap is not None:
                    cap.release()
                time.sleep(0.05)
                continue

            # Optional test read
            ok = True
            if test_read:
                ok = _try_read(cap, attempts=3)
                # If initial reads failed, try one explicit reopen (often fixes V4L2 timeouts)
                if not ok and reopen_once:
                    cap.release()
                    time.sleep(0.05)
                    cap = _open_cam(cam_id)
                    if cap.isOpened():
                        ok = _try_read(cap, attempts=3)

            if ok:
                # Warm-up frames (flush the pipeline so downstream .read() is instant)
                for _ in range(max(0, warmup_frames)):
                    cap.read()
                caps[cam_id] = cap
                opened = True
                break
            else:
                cap.release()
                time.sleep(0.05)

        if not opened:
            print(f"[WARN] Camera {cam_id}: unable to deliver frames after {retries} attempt(s). Skipping.")

    # Optionally: note any cells outside active grid (ignored)
    # (kept from your original for config sanity)
    for key in cells.keys():
        try:
            r, c = map(int, key.split(","))
            if r >= rows or c >= cols:
                # print(f"[INFO] Ignoring extra cell {key} outside active {rows}x{cols} grid.")
                pass
        except Exception:
            # print(f"[WARN] Bad cell key format: {key}")
            pass

    return caps

def save_arena_settings(arena_settings):
    # 1. Save the arena_settings.json
    os.makedirs(os.path.dirname(ARENA_SETTINGS_PATH), exist_ok=True)
    with open(ARENA_SETTINGS_PATH, "w") as f:
        json.dump(arena_settings, f, indent=2)

    # 2. Calculate total width and height
    rows = arena_settings.get("rows", 0)
    cols = arena_settings.get("cols", 0)
    cells = arena_settings.get("cells", {})

    # Width = sum of widths in a row minus overlaps between adjacent cells
    total_width = 0
    for c in range(cols):
        key = f"0,{c}"
        cell = cells.get(key, {})
        total_width += cell.get("width", 0)

    # Height = sum of heights in a column (no vertical overlap in your data)
    total_height = 0
    for r in range(rows):
        key = f"{r},0"
        cell = cells.get(key, {})
        total_height += cell.get("height", 0)

    # 3. Update settings.json
    settings_data = {}
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, "r") as f:
            try:
                settings_data = json.load(f)
            except json.JSONDecodeError:
                pass

    settings_data["arena_width"] = str(total_width)
    settings_data["arena_height"] = str(total_height)

    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings_data, f, indent=2)

def safe_grab_set(window):
    def try_grab():
        try:
            window.grab_set()
        except Exception as e:
            print(f"[Grab Failed] {e}")
    window.after(200, try_grab)

def warp_arena_frame_extended(frame, cell_key="0,0"):
    """
    Warps a frame to show a "zoomed-out" view with a 20px expanded border
    using information from the original image.
    """
    arena_settings = load_arena_settings()
    cell = arena_settings.get("cells", {}).get(cell_key, None)
    if cell is None:
        raise ValueError(f"No cell configuration found for {cell_key}")

    # Define the pixel amount for the expanded view on each side
    # output_content_overlap = 30

    REF_WIDTH = 800
    REF_HEIGHT = 600
    rotation = cell.get("rotation", 0)

    # These are the dimensions for the primary *content* area in the final output
    output_content_width = int(cell.get("width", 800))
    output_content_height = int(cell.get("height", 600))
    output_content_overlap = int(cell.get("overlap", 0))

    # Calculate the total dimensions of the final expanded frame
    final_width = output_content_width + 2 * output_content_overlap
    final_height = output_content_height + 2 * output_content_overlap

    # Source points from the camera feed (relative to the REF_WIDTH/HEIGHT)
    src_pts = np.array([
        cell.get("topLeft", [0, 0]),
        cell.get("topRight", [0, 0]),
        cell.get("bottomRight", [0, 0]),
        cell.get("bottomLeft", [0, 0])
    ], dtype=np.float32)

    # --- Pre-processing Step (same as before) ---
    # Resize input to a consistent reference size and apply rotation
    processed_frame = cv2.resize(frame, (REF_WIDTH, REF_HEIGHT))
    if rotation == 90:
        processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_180)
    elif rotation == 270:
        processed_frame = cv2.rotate(processed_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # --- New Warping Logic ---
    # The destination points are now the corners of the *content area* within
    # the final, larger output frame. We are mapping the source corners
    # to a rectangle inset by output_content_overlap.
    dst_pts = np.array([
        [output_content_overlap, output_content_overlap],
        [final_width - 1 - output_content_overlap, output_content_overlap],
        [final_width - 1 - output_content_overlap, final_height - 1 - output_content_overlap],
        [output_content_overlap, final_height - 1 - output_content_overlap]
    ], dtype=np.float32)

    # Get the matrix that maps from the source corners to the inset destination
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Warp the frame directly to the final, larger dimensions.
    # OpenCV fills the expanded border by continuing the perspective transform.
    expanded_frame = cv2.warpPerspective(
        processed_frame,
        matrix,
        (final_width, final_height)
    )

    return expanded_frame

def warp_arena_frame(frame, cell_key="0,0"):
    arena_settings = load_arena_settings()
    cell = arena_settings.get("cells", {}).get(cell_key, None)
    if cell is None:
        raise ValueError(f"No cell configuration found for {cell_key}")

    REF_WIDTH = 800
    REF_HEIGHT = 600
    rotation = cell.get("rotation", 0)
    output_width = int(cell.get("width", 800))
    output_height = int(cell.get("height", 600))

    src_pts = np.array([
        cell.get("topLeft", [0, 0]),
        cell.get("topRight", [0, 0]),
        cell.get("bottomRight", [0, 0]),
        cell.get("bottomLeft", [0, 0])
    ], dtype=np.float32)

    dst_pts = np.array([
        [0, 0],
        [REF_WIDTH - 1, 0],
        [REF_WIDTH - 1, REF_HEIGHT - 1],
        [0, REF_HEIGHT - 1]
    ], dtype=np.float32)

    # Perspective warp
    frame = cv2.resize(frame, (REF_WIDTH, REF_HEIGHT))
    # Apply rotation
    if rotation == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, matrix, (REF_WIDTH, REF_HEIGHT))


    # Final resize to output dimensions
    warped = cv2.resize(warped, (output_width, output_height))
    return warped

def launch_grid_popup(parent_app, camera_options):
    popup = ctk.CTkToplevel(parent_app)
    popup.title("Grid Configurator")
    popup.geometry("500x600")
    safe_grab_set(popup)

    def validate_input(new_value):
        if new_value == "":
            return True
        if new_value.isdigit():
            val = int(new_value)
            return 1 <= val <= 5
        return False

    def on_grid_cell_click(r, c):
        open_cell_config_popup(popup, r, c, camera_options)

    # --- Frame 2: Preview Grid (must define before update_preview) ---
    preview_frame = ctk.CTkFrame(popup)
    preview_frame.pack(pady=10, padx=10, fill="both", expand=True)

    def update_preview():
        for widget in preview_frame.winfo_children():
            widget.destroy()

        try:
            rows = int(row_entry.get())
            cols = int(col_entry.get())

            for i in range(50):
                preview_frame.grid_rowconfigure(i, weight=0)
                preview_frame.grid_columnconfigure(i, weight=0)

            for r in range(rows):
                preview_frame.grid_rowconfigure(r, weight=1)
                for c in range(cols):
                    preview_frame.grid_columnconfigure(c, weight=1)
                    btn = ctk.CTkButton(
                        preview_frame,
                        text="+",
                        command=lambda r=r, c=c: on_grid_cell_click(r, c)
                    )
                    btn.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")

        except ValueError:
            pass

    vcmd = parent_app.register(validate_input)
    settings = load_arena_settings()
    saved_rows = settings.get("rows", 1)
    saved_cols = settings.get("cols", 1)

    # --- Frame 1: Inputs ---
    input_frame = ctk.CTkFrame(popup)
    input_frame.pack(pady=10, padx=10, fill="x")

    row_label = ctk.CTkLabel(input_frame, text="No of Rows:")
    row_label.grid(row=0, column=0, padx=5, pady=5)
    row_entry = ctk.CTkEntry(input_frame, validate="key", validatecommand=(vcmd, "%P"))
    row_entry.grid(row=0, column=1, padx=5, pady=5)

    col_label = ctk.CTkLabel(input_frame, text="No of Columns:")
    col_label.grid(row=1, column=0, padx=5, pady=5)
    col_entry = ctk.CTkEntry(input_frame, validate="key", validatecommand=(vcmd, "%P"))
    col_entry.grid(row=1, column=1, padx=5, pady=5)

    row_entry.insert(0, str(saved_rows))
    col_entry.insert(0, str(saved_cols))
    update_preview()

    row_entry.bind("<KeyRelease>", lambda e: update_preview())
    col_entry.bind("<KeyRelease>", lambda e: update_preview())

    # --- Frame 3: Save/Cancel ---
    button_frame = ctk.CTkFrame(popup)
    button_frame.pack(pady=10, padx=10, fill="x")

    def on_save():
        try:
            rows = int(row_entry.get())
            cols = int(col_entry.get())

            # Load full settings and update rows/cols
            settings = load_arena_settings()
            settings["rows"] = rows
            settings["cols"] = cols

            save_arena_settings(settings)
            popup.destroy()
        except ValueError:
            print("Invalid row/column input")


    def on_cancel():
        popup.destroy()

    ctk.CTkButton(button_frame, text="Save", command=on_save).pack(side="left", expand=True, padx=20)
    ctk.CTkButton(button_frame, text="Cancel", command=on_cancel).pack(side="right", expand=True, padx=20)

    return popup

def open_cell_config_popup(parent, row, col, camera_options):
    should_stream = [True]
    is_dragging = [False]
    corner_coords = {
        "TL": (0, 0),
        "TR": (0, 0),
        "BL": (0, 0),
        "BR": (0, 0)
    }
    
    cam_popup = ctk.CTkToplevel(parent)
    cam_popup.title(f"Configure Cell {row},{col}")
    cam_popup.geometry("800x900")
    safe_grab_set(cam_popup)

    # ---- 1. TOP ROW: Camera selector + width/height (evenly spaced) ---
    top_row = ctk.CTkFrame(cam_popup)
    top_row.pack(pady=10, padx=10, fill="x")

    # Select Camera label
    ctk.CTkLabel(top_row, text="Camera:").grid(row=0, column=0, padx=5, pady=5, sticky="ew")

    # Option menu
    selected_cam = ctk.StringVar(value=camera_options[0] if camera_options else "0")
    cam_menu = ctk.CTkOptionMenu(top_row, values=camera_options, variable=selected_cam, width=50)
    cam_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    # Width label and entry
    ctk.CTkLabel(top_row, text="Width:").grid(row=0, column=2, padx=5, pady=5, sticky="ew")
    width_entry = ctk.CTkEntry(top_row)
    width_entry.insert(0, "640")
    width_entry.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

    # Height label and entry
    ctk.CTkLabel(top_row, text="Height:").grid(row=0, column=4, padx=5, pady=5, sticky="ew")
    height_entry = ctk.CTkEntry(top_row)
    height_entry.insert(0, "480")
    height_entry.grid(row=0, column=5, padx=5, pady=5, sticky="ew")

    # Overlap label and entry
    ctk.CTkLabel(top_row, text="Overlap:").grid(row=0, column=6, padx=5, pady=5, sticky="ew")
    overlap_entry = ctk.CTkEntry(top_row)
    overlap_entry.insert(0, "0")
    overlap_entry.grid(row=0, column=7, padx=5, pady=5, sticky="ew")
    
    # Rotate button
    rotation_angle = [0]  # mutable container to persist across nested functions

    def rotate_feed():
        rotation_angle[0] = (rotation_angle[0] + 90) % 360

    rotate_btn = ctk.CTkButton(top_row, text="Rotate", command=rotate_feed)
    rotate_btn.grid(row=0, column=8, padx=5, pady=5, sticky="ew")

    # Grid layout
    top_row.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6, 7, 8), weight=1, uniform="x")

    corner_coords = {
        "TL": (0, 0),
        "TR": (0, 0),
        "BL": (0, 0),
        "BR": (0, 0)
    }

    cell_key = f"{row},{col}"
    settings = load_arena_settings()
    cell_settings = settings.get("cells", {}).get(cell_key, {})

    # Pre-fill entries if available
    selected_cam.set(str(cell_settings.get("camera", camera_options[0] if camera_options else "0")))
    width_entry.delete(0, "end")
    width_entry.insert(0, str(cell_settings.get("width", 640)))
    height_entry.delete(0, "end")
    height_entry.insert(0, str(cell_settings.get("height", 480)))
    overlap_entry.delete(0, "end")
    overlap_entry.insert(0, str(cell_settings.get("overlap", 0)))
    rotation_angle[0] = cell_settings.get("rotation", 0)
    corner_coords.update({
        "TL": tuple(cell_settings.get("topLeft", (0, 0))),
        "TR": tuple(cell_settings.get("topRight", (0, 0))),
        "BL": tuple(cell_settings.get("bottomLeft", (0, 0))),
        "BR": tuple(cell_settings.get("bottomRight", (0, 0)))
    })

    # 2. Buttons: TL, TR, BL, BR
    corner_frame = ctk.CTkFrame(cam_popup)
    corner_frame.pack(pady=10, padx=10, fill="x")

    corner_frame.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="x")

    corner_buttons = {}

    selected_corner = [None]  # Track which one is selected

    def on_corner_click(name):
        if selected_corner[0] == name:
            # Deselect: re-enable all and reset colors
            for n, b in corner_buttons.items():
                b.configure(state="normal", fg_color="transparent", text_color="white")
            selected_corner[0] = None
        else:
            # Select new: disable others and highlight this one
            for n, b in corner_buttons.items():
                if n == name:
                    b.configure(state="normal", fg_color="red", text_color="white")
                else:
                    b.configure(state="disabled", fg_color="transparent", text_color="white")
            selected_corner[0] = name

    for i, name in enumerate(["TL", "TR", "BL", "BR"]):
        btn = ctk.CTkButton(
            corner_frame,
            text=name,
            command=lambda n=name: on_corner_click(n)
        )
        btn.grid(row=0, column=i, padx=10, pady=5, sticky="ew")
        btn.configure(fg_color="transparent", text_color="white")
        corner_buttons[name] = btn

    # 3. Live feed
    img_frame = ctk.CTkFrame(cam_popup, width=800, height=600)
    img_frame.pack(padx=10, pady=10)
    img_frame.pack_propagate(False)  # prevents resizing by contents
    cam_popup.resizable(False, False)

    img_label = ctk.CTkLabel(img_frame, text="", width=800, height=600)
    img_label.pack()

    def on_mouse_press(event):
        if selected_corner[0] is None:
            return
        is_dragging[0] = True  # Start drag

    def on_mouse_release(event):
        if not is_dragging[0] or selected_corner[0] is None:
            return
        is_dragging[0] = False  # End drag

        img_x = int(event.x)
        img_y = int(event.y)

        # Save to corner
        corner_coords[selected_corner[0]] = (img_x, img_y)

        # Reset button states
        for n, b in corner_buttons.items():
            b.configure(state="normal", fg_color="transparent", text_color="white")
        selected_corner[0] = None
    
    def on_mouse_motion(event):
        if not is_dragging[0] or selected_corner[0] is None:
            return

        img_x = int(event.x)
        img_y = int(event.y)

        # Temporarily update that corner position for visual feedback
        corner_coords[selected_corner[0]] = (img_x, img_y)

    img_label.bind("<ButtonPress-1>", on_mouse_press)
    img_label.bind("<ButtonRelease-1>", on_mouse_release)
    img_label.bind("<B1-Motion>", on_mouse_motion)

    frame_size = [800, 600]
    original_aspect = 4 / 3

    cap = [None]

    def show_frame():
        if not should_stream[0]:
            return  # Exit loop if we're closing

        if cap[0]:
            ret, frame = cap[0].read()
            if ret:
                # --- Rotation ---
                angle = rotation_angle[0]
                if angle == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif angle == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # --- Resize and display ---
                target_w, target_h = frame_size
                aspect = 1 / original_aspect if angle in [90, 270] else original_aspect

                if target_w / target_h > aspect:
                    new_h = target_h
                    new_w = int(aspect * new_h)
                else:
                    new_w = target_w
                    new_h = int(new_w / aspect)

                # Resize frame first
                frame = cv2.resize(frame, (new_w, new_h))

                # Draw corner rectangle if all corners are defined
                tl = corner_coords["TL"]
                tr = corner_coords["TR"]
                br = corner_coords["BR"]
                bl = corner_coords["BL"]

                # Only draw if at least one corner is not (0,0)
                if any(pt != (0, 0) for pt in [tl, tr, br, bl]):
                    points = [tl, tr, br, bl]
                    scaled_pts = [(int(x), int(y)) for (x, y) in points]

                    # Draw polygon
                    cv2.polylines(
                        frame,
                        [np.array(scaled_pts, dtype=np.int32)],
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2
                    )
                    # Draw corner dots
                    for pt in scaled_pts:
                        cv2.circle(
                            frame,
                            pt,
                            radius=5,
                            color=(0, 0, 255),  # Red dot
                            thickness=-1        # Filled circle
                        )

                # Now convert to RGB and display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                img = CTkImage(light_image=Image.fromarray(frame), size=(new_w, new_h))
                img_label.configure(image=img)
                img_label.image = img

        img_label.after(30, show_frame)


    def switch_camera(*_):
        try:
            if cap[0]:
                cap[0].release()
            # Extract number from 'Camera 0', 'Camera 2', etc.
            cam_index = int(selected_cam.get().split()[-1])
            cap[0] = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        except Exception as e:
            print(f"Camera error: {e}")

    selected_cam.trace_add("write", switch_camera)
    switch_camera()
    show_frame()

    # 4. Save/Cancel
    def save_settings_and_close():
        try:

            # Read UI values
            camera_idx = int(selected_cam.get().split()[-1])
            width = int(width_entry.get())
            height = int(height_entry.get())
            overlap = int(overlap_entry.get())  # new line
            rotation = rotation_angle[0]  # assuming you're using a list for mutability

            # Load existing settings
            settings = load_arena_settings()
            cell_key = f"{row},{col}"

            # Ensure 'cells' exists
            if "cells" not in settings:
                settings["cells"] = {}

            # Store settings for this cell
            settings["cells"][cell_key] = {
                "camera": camera_idx,
                "width": width,
                "height": height,
                "rotation": rotation,
                "overlap": overlap,  # ← ADD THIS
                "topLeft": list(corner_coords.get("TL", (0, 0))),
                "topRight": list(corner_coords.get("TR", (0, 0))),
                "bottomLeft": list(corner_coords.get("BL", (0, 0))),
                "bottomRight": list(corner_coords.get("BR", (0, 0)))
            }
            # Save back to JSON
            save_arena_settings(settings)

        except Exception as e:
            print(f"[Save Error] {e}")
        
        finally:
            should_stream[0] = False
            if cap[0]:
                cap[0].release()
            cam_popup.destroy()


    def cancel_and_close():
        should_stream[0] = False
        if cap[0]:
            cap[0].release()
        cam_popup.destroy()

    def on_window_close():
        should_stream[0] = False
        if cap[0]:
            cap[0].release()
        cam_popup.destroy()

    cam_popup.protocol("WM_DELETE_WINDOW", on_window_close)

    button_row = ctk.CTkFrame(cam_popup)
    button_row.pack(pady=15)

    ctk.CTkButton(button_row, text="Save", command=save_settings_and_close).pack(side="left", padx=15)
    ctk.CTkButton(button_row, text="Cancel", command=cancel_and_close).pack(side="right", padx=15)

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.geometry("300x150")

    btn = ctk.CTkButton(
        app, 
        text="Open Grid Popup", 
        command=lambda: launch_grid_popup(app, refresh_cameras()))
    btn.pack(pady=40)

    app.mainloop()
