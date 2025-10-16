# vision_dashboard/core/tasks.py

import threading
import cv2


from vision.detectors import VLDetector, BGSubtractor
# from services.gemini import call_gemini_agent
from ui_utils import CTkMessageBox, set_preview_text

# Initialize detectors once to be reused.
# In a real app, these might be part of a larger service.
VLM_DETECTOR = VLDetector()
BG_DETECTOR = BGSubtractor()


def calibrate_task(client, obstacle_prompt: str):
    """
    Tells the server to perform the calibration sequence.
    It fetches the latest frame from the Data service, runs detection,
    and updates the Data service with the results.
    """
    print("TASK: Sending calibration request to server...")
    try:
        # This is now a single, clean RPC call. The server does the heavy lifting.
        client.Tasks.calibrate(obstacle_prompt)
        print("TASK: Server acknowledged calibration.")
    except Exception as e:
        print(f"TASK: Calibration failed. Server error: {e}")
        # The UI update must be scheduled on the app's main thread.
        # We can't do that from here, so the main_window's on_complete will handle popups.


def run_gemini_task(app_instance, text_box, user_prompt, agent_name):
    """Calls the Gemini agent and displays the output in a text box."""
    print(f"TASK: Running Gemini agent '{agent_name}' with prompt: {user_prompt[:50]}...")
    try:
        #response = call_gemini_agent(user_prompt, agent_name)
        print("GEMINI NOT LOADED YET")
    except Exception as e:
        response = f"Error calling Gemini:\n{e}"
    
    # Schedule the UI update on the main thread via the app instance
    app_instance.after(0, lambda: set_preview_text(text_box, response))
    print("TASK: Gemini task complete.")


def toggle_execute(app_instance, execute_btn):
    """
    Manages the start and stop of robot execution by sending commands to the Robot service.
    """
    client = app_instance.client
    if not hasattr(app_instance, 'is_executing'):
        app_instance.is_executing = False

    if not app_instance.is_executing:
        # --- START EXECUTION ---
        try:
            print("TASK: Sending START command to Robot service...")
            client.Robot.start_execution()
            app_instance.is_executing = True
            execute_btn.configure(text="Stop", fg_color="#FF3B30")
        except Exception as e:
            CTkMessageBox(app_instance, "Error", f"Could not start robot:\n{e}", "red")
    else:
        # --- STOP EXECUTION ---
        try:
            print("TASK: Sending STOP command to Robot service...")
            client.Robot.stop_execution()
            app_instance.is_executing = False
            execute_btn.configure(text="Execute", fg_color="#7228E9")
        except Exception as e:
            CTkMessageBox(app_instance, "Error", f"Could not stop robot:\n{e}", "red")