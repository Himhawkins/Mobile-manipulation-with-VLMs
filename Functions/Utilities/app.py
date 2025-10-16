# app.py
import cv2
import io
from PIL import Image
from flask import Flask, render_template, Response, jsonify, request

# Local application imports (ensure these are in your Python path)
from rpc_system import RPCClient
from utils import ports, agents

app = Flask(__name__)

# --- State & Backend Connection ---
# Global state management for the web server
app_state = {
    'is_executing': False,
    'rpc_client': None
}

# Try to connect to the RPC server on startup
try:
    app_state['rpc_client'] = RPCClient()
    print("Successfully connected to RPC Server.")
except Exception as e:
    print(f"FATAL: Could not connect to RPC Server. Is it running? Error: {e}")
    # The server will still run, but API calls will fail gracefully.

# --- HTML Page Routes ---
@app.route('/')
def index():
    """Serves the main dashboard page."""
    return render_template('index.html')

# --- Video Streaming ---
def generate_video_frames():
    """
    Generator function to fetch, process, and stream video frames.
    This replaces the update_display() loop.
    """
    client = app_state.get('rpc_client')
    if not client:
        # Handle case where RPC connection failed
        # You could generate a static "Disconnected" image here
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
        return

    while True:
        try:
            # Get the latest data packet from the server
            state = client.Data.get_full_state()
            frame = state['stitched_frame']

            # Draw robot poses
            for mid, pose in state['robot_poses'].items():
                cv2.circle(frame, (pose['x'], pose['y']), 12, (0, 255, 0), 2)
                cv2.putText(frame, f"ID {mid}", (pose['x'], pose['y'] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw static obstacles
            for x, y, w, h in state['static_obstacles']:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            # Yield the frame in the multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error fetching frame from server: {e}")
            # If an error occurs, break the loop to prevent crashing the server thread
            break

@app.route('/video_feed')
def video_feed():
    """Returns the video stream."""
    return Response(generate_video_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- API Endpoints for UI Actions ---
@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    """API endpoint to trigger calibration."""
    client = app_state.get('rpc_client')
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'status': 'error', 'message': 'Prompt cannot be empty.'}), 400
    if not client:
        return jsonify({'status': 'error', 'message': 'RPC Server not connected.'}), 503

    try:
        client.Tasks.calibrate(prompt)
        return jsonify({'status': 'success', 'message': 'Calibration task started.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Server error: {e}'}), 500

@app.route('/api/execute', methods=['POST'])
def api_execute():
    """API endpoint to start/stop robot execution."""
    client = app_state.get('rpc_client')
    if not client:
        return jsonify({'status': 'error', 'message': 'RPC Server not connected.'}), 503

    try:
        if not app_state['is_executing']:
            client.Robot.start_execution()
            app_state['is_executing'] = True
            return jsonify({'status': 'started', 'is_executing': True})
        else:
            client.Robot.stop_execution()
            app_state['is_executing'] = False
            return jsonify({'status': 'stopped', 'is_executing': False})
    except Exception as e:
        app_state['is_executing'] = False # Reset state on error
        return jsonify({'status': 'error', 'message': f'Execution command failed: {e}'}), 500

@app.route('/api/ports', methods=['GET'])
def api_get_ports():
    """Refreshes and returns a list of serial ports."""
    port_list = ports.refresh_serial_ports() or []
    return jsonify(port_list)

# You can expand the API with endpoints for managing agents
# (listing, creating, editing) using the functions in `utils/agents.py`

if __name__ == '__main__':
    # Use threaded=True to handle multiple requests (e.g., video stream + API calls)
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)