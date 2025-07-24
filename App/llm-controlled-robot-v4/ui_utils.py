import customtkinter as ctk
import json
import os
import cv2
from PIL import Image
import numpy as np

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

def overlay_arena_and_obstacles(frame, arena_path="Data/arena_corners.txt", obstacles_path="Data/obstacles.txt"):
    overlay = frame.copy()
    # 1. Draw arena (yellow polygon)
    try:
        with open(arena_path, "r") as f:
            arena_pts = [list(map(int, line.strip().split(",")))
                         for line in f if line.strip()]
        if len(arena_pts) == 4:
            pts = np.array(arena_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=True,
                          color=(0, 255, 255), thickness=2)
        else:
            print(f"[WARN] '{arena_path}' does not contain exactly 4 points.")
    except Exception as e:
        print(f"[ERROR] Could not read arena corners from '{arena_path}': {e}")
    # 2. Draw obstacles (red rectangles)
    try:
        with open(obstacles_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 4:
                    continue
                x, y, w, h = map(int, parts)
                cv2.rectangle(overlay,
                              (x, y),
                              (x + w, y + h),
                              color=(0, 0, 255),
                              thickness=-1)  # filled
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

    # 2. Draw obstacles (red rectangles)
    try:
        with open(obstacles_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 4:
                    continue
                x, y, w, h = map(int, parts)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
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
