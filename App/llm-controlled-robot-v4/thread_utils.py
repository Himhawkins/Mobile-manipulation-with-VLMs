import threading
import customtkinter as ctk

from ui_utils import CTkMessageBox, get_app_settings
from detection import detect_arena, detect_and_get_bbox, save_img_to_path
from Functions.Library.Agent.gemini import call_gemini_agent

def set_preview_text(text_box, text: str):
    text_box.configure(state="normal")
    text_box.delete("1.0", ctk.END)      # ← use 1.0
    text_box.insert("1.0", text)         # ← use 1.0
    text_box.configure(state="disabled")

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

def disable_button(btn_name):
    btn_name.configure(state="disabled")

def enable_button(btn_name):
    btn_name.configure(state="normal")
    
def callibrate_task(app):
    app.settings = get_app_settings()
    corner_prompt = app.settings.get("corner_prompt")
    obstacle_prompt = app.settings.get("obstacle_prompt")
    save_img_to_path(app.current_frame, save_path="Data/frame_img.png")
    try:
        detect_arena(img_path="Data/frame_img.png", prompt=corner_prompt, save_path="Data/arena_corners.txt")
        app.after(0, lambda: CTkMessageBox(app, "Status", "Calibrated successfully", "white"))
    except ValueError as e:
        app.after(0, lambda e=e: CTkMessageBox(app, "Detection Error", str(e), "yellow"))
        return False
    except Exception as e:
        app.after(0, lambda e=e: CTkMessageBox(app, "Error", str(e), "red"))
        return False

    try:
        detect_and_get_bbox(img_path="Data/frame_img.png", prompt=obstacle_prompt, save_path="Data/obstacles.txt")
    except Exception as e:
        app.after(0, lambda e=e: CTkMessageBox(app, "Error", str(e), "red"))
        return False
    
    return True

def run_task(app, text_box, user_prompt, agent_name):
    # if (callibrate_task(app) == True):
    #     text_out = call_gemini_agent(user_prompt, agent_name)
    #     text_box.after(0, lambda: set_preview_text(text_box=text_box, text=text_out))
    # else:
    #     return
    text_out = call_gemini_agent(user_prompt, agent_name)
    text_box.after(0, lambda: set_preview_text(text_box=text_box, text=text_out))