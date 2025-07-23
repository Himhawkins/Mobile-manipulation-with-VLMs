import threading
import customtkinter as ctk

from ui_utils import CTkMessageBox, get_app_settings
from detection import detect_arena, detect_and_get_bbox

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
    try:
        detect_arena(app.current_frame, corner_prompt, save_path="Data/arena_corners.txt")
    except ValueError as e:
        app.after(0, lambda e=e: CTkMessageBox(app, "Detection Error", str(e), "yellow"))
    except Exception as e:
        app.after(0, lambda e=e: CTkMessageBox(app, "Error", str(e), "red"))

    try:
        detect_and_get_bbox(app.current_frame, obstacle_prompt, save_path="Data/obstacles.txt")
    except Exception as e:
        app.after(0, lambda e=e: CTkMessageBox(app, "Error", str(e), "red"))

