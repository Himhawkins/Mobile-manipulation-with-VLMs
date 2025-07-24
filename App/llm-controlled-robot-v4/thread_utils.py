import threading
import customtkinter as ctk

from ui_utils import CTkMessageBox, get_app_settings
from detection import detect_arena, detect_and_get_bbox, save_img_to_path

def run_in_thread(callback, on_start=None, on_complete=None):
    """
    Runs a callback function in a separate thread, with optional UI-safe
    on_start and on_complete functions to run before and after.

    Useful for keeping the UI responsive during heavy computation.
    """
    def wrapper():
        if on_start:
            on_start()
        try:
            callback()
        finally:
            if on_complete:
                on_complete()
    thread = threading.Thread(target=wrapper)
    thread.start()

def disable_button(app, btn_name):
    for child in app.children.values():
        if isinstance(child, ctk.CTkFrame):
            for btn in child.winfo_children():
                if isinstance(btn, ctk.CTkButton) and btn.cget("text") == btn_name:
                    btn.configure(state="disabled")

def enable_button(app, btn_name):
    for child in app.children.values():
        if isinstance(child, ctk.CTkFrame):
            for btn in child.winfo_children():
                if isinstance(btn, ctk.CTkButton) and btn.cget("text") == btn_name:
                    btn.configure(state="normal")

def run_task(app):
    app.settings = get_app_settings()
    corner_prompt = app.settings.get("corner_prompt")
    obstacle_prompt = app.settings.get("obstacle_prompt")
    save_img_to_path(app.current_frame, save_path="Data/frame_img.png")
    try:
        detect_arena(img_path="Data/frame_img.png", prompt=corner_prompt, save_path="Data/arena_corners.txt")
    except ValueError as e:
        app.after(0, lambda e=e: CTkMessageBox(app, "Detection Error", str(e), "yellow"))
    except Exception as e:
        app.after(0, lambda e=e: CTkMessageBox(app, "Error", str(e), "red"))

    try:
        detect_and_get_bbox(img_path="Data/frame_img.png", prompt=obstacle_prompt, save_path="Data/obstacles.txt")
    except Exception as e:
        app.after(0, lambda e=e: CTkMessageBox(app, "Error", str(e), "red"))

