import cv2
import pyvirtualcam
import subprocess
import sys # Import sys for exiting gracefully

def load_v4l2loopback():
    """
    Ensures the v4l2loopback kernel module is loaded with the specified parameters.
    This function requires sudo privileges to run.
    """
    try:
        # Attempt to remove the module first to ensure a clean load
        print("Attempting to unload v4l2loopback (if loaded)...")
        subprocess.run([
            "sudo", "modprobe", "-r", "v4l2loopback"
        ], check=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE) # Use check=False as it might not be loaded

        # Load the module with desired parameters
        print("Loading v4l2loopback module...")
        subprocess.run([
            "sudo", "modprobe", "v4l2loopback",
            "devices=1", "video_nr=10", "card_label=VirtualCam", "exclusive_caps=1"
        ], check=True)
        print("v4l2loopback module loaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error loading v4l2loopback module: {e}", file=sys.stderr)
        print("Please ensure you have v4l2loopback installed and have sudo privileges.", file=sys.stderr)
        sys.exit(1) # Exit if module loading fails
    except FileNotFoundError:
        print("Error: 'sudo' command not found. Please ensure it's in your PATH.", file=sys.stderr)
        sys.exit(1) # Exit if sudo is not found

def initialize_camera(camera_index=0):
    """
    Initializes and returns a cv2.VideoCapture object for the specified camera index.
    """
    print(f"Opening physical camera at index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open physical camera at index {camera_index}.", file=sys.stderr)
        sys.exit(1) # Exit if camera cannot be opened
    print("Physical camera opened successfully.")
    return cap

def get_camera_properties(cap):
    """
    Retrieves and returns the width, height, and FPS of the camera.
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Fallback in case FPS is 0
        fps = 30
        print(f"Warning: Camera reported 0 FPS. Defaulting to {fps} FPS.", file=sys.stderr)
    print(f"Camera properties: Width={width}, Height={height}, FPS={fps}")
    return width, height, fps

def run_virtual_camera_stream(cap, width, height, fps, virtual_device='/dev/video10'):
    """
    Sets up the virtual camera and streams frames from the physical camera to it.
    """
    print(f"Creating virtual camera on device: {virtual_device}...")
    try:
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, device=virtual_device) as cam:
            print(f"Using virtual camera: {cam.device}")
            print("Streaming started. Press Ctrl+C to stop.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from physical camera. Exiting.", file=sys.stderr)
                    break

                # Convert BGR frame from OpenCV to RGB for pyvirtualcam
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam.send(frame_rgb)
                cam.sleep_until_next_frame()
    except Exception as e:
        print(f"An error occurred during streaming: {e}", file=sys.stderr)
    finally:
        print("Releasing physical camera resource.")
        cap.release()
        print("Streaming stopped.")

def main_virtual_camera():
    """
    Main function to orchestrate the virtual camera application.
    """
    load_v4l2loopback()
    cap = initialize_camera()
    width, height, fps = get_camera_properties(cap)
    run_virtual_camera_stream(cap, width, height, fps)

if __name__ == "__main__":
    main_virtual_camera()
