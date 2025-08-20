# /dashboard_webapp/app.py

import cv2
import json
import threading
from flask import Flask, render_template, Response, request, jsonify

# Assume your original helper modules are available and work without customtkinter
# You might need to adjust them to return data instead of updating a GUI
from camera_utils import start_camera, read_frame, draw_robot_pose
from detection import detect_robot_pose
from port_utils import refresh_cameras, refresh_serial_ports
from agent_utils import get_agent_folders, get_agent_functions, save_agent_to_disk, get_all_functions
from thread_utils import callibrate_task, run_task, toggle_execute_web # toggle_execute needs modification for web
from ui_utils import get_app_settings, overlay_arena_and_obstacles, draw_path_on_frame, get_overlay_frame

app = Flask(__name__)

# --- State Management ---
# In a real-world app, you might use a more robust state management solution
class AppState:
    def __init__(self):
        self.cap = None
        self.camera_index = 0
        self.is_executing = False
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def set_camera(self, index):
        with self.lock:
            self.camera_index = index
            if self.cap:
                self.cap.release()
            self.cap = start_camera(self.camera_index)

    def get_frame(self):
        with self.lock:
            if self.cap and self.cap.isOpened():
                return read_frame(self.cap)
        return None

app_state = AppState()

# --- Video Streaming ---
def generate_frames():
    """Generator function for video streaming."""
    while True:
        frame = app_state.get_frame()
        if frame is None:
            # Could stream a "No Signal" image here
            continue

        # --- Apply all the same processing as your original app ---
        settings = get_app_settings()
        aruco_id = int(settings.get("aruco_id", "782"))
        pose = detect_robot_pose(frame=frame, aruco_id=aruco_id)

        if pose:
            cx, cy, theta, pts = pose
            processed_frame = draw_robot_pose(frame, cx, cy, theta, pts)
        else:
            processed_frame = frame

        overlay = overlay_arena_and_obstacles(
            frame=processed_frame,
            arena_path="Data/arena_corners.txt",
            obstacles_path="Data/obstacles.txt"
        )
        final_frame = draw_path_on_frame(overlay, path_file="Targets/path.txt")
        # --- End of processing ---

        ret, buffer = cv2.imencode('.jpg', final_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Frontend Routes ---
@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint for the video stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API Endpoints ---
@app.route('/api/refresh_sources', methods=['GET'])
def refresh_sources():
    """Get initial lists of cameras, ports, and agents."""
    cameras = refresh_cameras()
    ports = refresh_serial_ports()
    agents = get_agent_folders()
    agents.append("Create New")
    return jsonify({
        "cameras": cameras,
        "ports": ports,
        "agents": agents
    })
    
@app.route('/api/set_camera', methods=['POST'])
def set_camera():
    """Set the active camera for the video stream."""
    data = request.json
    cam_index = int(data.get('camera_index', 0))
    app_state.set_camera(cam_index)
    print(f"Switched to camera index: {cam_index}")
    return jsonify({"status": "success", "message": f"Camera set to index {cam_index}"})

@app.route('/api/run_action', methods=['POST'])
def run_action():
    """Endpoint to run calibrate or run tasks."""
    data = request.json
    action = data.get('action')
    response = {"status": "failed", "output": ""}

    if action == 'Calibrate':
        # NOTE: Tasks must be adapted to return data, not update GUI.
        # This is a simplified example.
        threading.Thread(target=callibrate_task, args=(None,)).start() # Pass dummy 'self' if needed
        response = {"status": "started", "output": "Calibration started."}
    
    elif action == 'Run':
        # Example of getting params and running in a thread
        input_val = data.get('inputValue')
        mode_val = data.get('modeValue')
        # The run_task function would need to be adapted to not require 'self' or a textbox.
        # It could write its output to a file that another endpoint could poll.
        threading.Thread(target=run_task, args=(None, None, input_val, mode_val)).start()
        response = {"status": "started", "output": f"Run task started for agent '{mode_val}'."}
        
    return jsonify(response)

# You would continue to add endpoints for every other piece of functionality:
# - /api/toggle_execute
# - /api/get_agent_details
# - /api/save_agent
# - /api/settings

if __name__ == '__main__':
    # Initialize default camera on startup
    app_state.set_camera(0) 
    app.run(debug=True, threaded=True) # `threaded=True` is important for handling requests during long tasks
