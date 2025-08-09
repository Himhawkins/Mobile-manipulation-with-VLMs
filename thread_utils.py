import threading
import customtkinter as ctk

from ui_utils import CTkMessageBox, get_app_settings
from detection import detect_arena, detect_and_get_bbox, detect_obstacles, save_img_to_path
from Functions.Library.Agent.gemini import call_gemini_agent
from controller import exec_bot_with_thread
from motion import move_robot_with_thread

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
    obstacle_prompt = app.settings.get("obstacle_prompt")
    save_img_to_path(app.current_frame, save_path="Data/frame_img.png")

    try:
        # detect_and_get_bbox(img_path="Data/frame_img.png", prompt=obstacle_prompt, save_path="Data/obstacles.txt")
        detect_arena(img_path="Data/frame_img.png", save_path="Data/arena_corners.txt")
        detect_obstacles(img_path="Data/frame_img.png", prompt=obstacle_prompt, save_path="Data/obstacles.txt")
        app.after(0, lambda: CTkMessageBox(app, "Status", "Calibrated successfully", "white"))
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

def _reset_execute_ui(app, execute_btn):
    """
    Reset the execute button back to its default (Execute/blue) state.
    """
    app.is_executing = False
    execute_btn.configure(text="Execute", fg_color="#7228E9")


def toggle_execute(app, serial_var, execute_btn,
                   baud_rate: int = 115200,
                   command_file: str = "Data/command.txt",
                   send_interval_s: float = 0.1):
    # initialize flags/event if needed
    if not hasattr(app, 'is_executing'):
        app.is_executing = False
    if not hasattr(app, 'stop_event'):
        app.stop_event = threading.Event()

    if not app.is_executing:
        # --- START EXECUTION ---
        port = serial_var.get().strip()
        if not port:
            CTkMessageBox(app, "Error", "Please select a serial port before executing.", "red")
            return

        app.is_executing = True
        app.stop_event.clear()
        execute_btn.configure(text="Stop", fg_color="#FF3B30")

        # 1) Exec Bot thread
        app.exec_thread = threading.Thread(
            target=exec_bot_with_thread,
            kwargs={'stop_event': app.stop_event},
            daemon=True
        )
        app.exec_thread.start()

        # 2) Move robot thread
        app.move_thread = threading.Thread(
            target=move_robot_with_thread,
            kwargs={
                'serial_port': port,
                'baud_rate': baud_rate,
                'command_file': command_file,
                'stop_event': app.stop_event,
                'send_interval_s': send_interval_s
            },
            daemon=True
        )
        app.move_thread.start()

        # 3) Watch for exec_thread completion
        def _on_exec_done():
            # block until exec_bot_with_thread returns
            app.exec_thread.join()
            # signal the motion thread to stop
            app.stop_event.set()
            # UI reset on main thread
            app.after(0, lambda: _reset_execute_ui(app, execute_btn))

        watcher = threading.Thread(target=_on_exec_done, daemon=True)
        watcher.start()

    else:
        # --- USER PRESSED STOP EARLY ---
        app.stop_event.set()
        _reset_execute_ui(app, execute_btn)