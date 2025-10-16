import numpy as np
import cv2
import os
import sys

# --- Configuration ---
CAMERA_ID = 4
CAMERA_INDEX = 4 # The actual index for cv2.VideoCapture
CALIBRATION_FILE = f"camera_calibration_data_{CAMERA_ID}.npz"
# ---------------------

def run_camera_stream_reliable_undistort():
    # 1. Load Camera Calibration
    if not os.path.exists(CALIBRATION_FILE):
        print(f"Error: Calibration file '{CALIBRATION_FILE}' not found.")
        print("Please ensure the file is in the correct directory.")
        sys.exit(1)

    print(f"Loading calibration data from {CALIBRATION_FILE}...")
    try:
        data = np.load(CALIBRATION_FILE)
        # Assuming your file keys are 'camera_matrix' and 'dist_coeffs'
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        print("Calibration data loaded successfully.")
    except KeyError:
        print("Error: The .npz file must contain 'camera_matrix' and 'dist_coeffs' keys.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        sys.exit(1)

    # 2. Open Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {CAMERA_INDEX}. Check connection.")
        sys.exit(1)

    # Note: Using None for the last argument of cv2.undistort (newCameraMatrix)
    # means it will use the original camera_matrix, which generally results in a
    # slightly larger, uncropped undistorted image. If you prefer to crop to
    # the optimal region, you can compute a new camera matrix here.

    print("Starting video stream. Press 'q' to exit.")

    # 3. Continuous Stream and Undistortion Loop
    while True:
        ret, frame_raw = cap.read() # Read the current raw frame

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # --- Undistortion and Grayscale Processing ---
        
        # Undistort the frame using cv2.undistort()
        # It takes the raw frame, the calibration matrix, and distortion coefficients.
        # The last two 'None' arguments mean: 
        #   - No optional rectification transform (R)
        #   - Use the original camera_matrix for the output (newCameraMatrix)
        undistorted_frame = cv2.undistort(frame_raw, camera_matrix, dist_coeffs, None, None)
        
        # Convert the undistorted frame to grayscale
        gray_undistorted = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        
        # --- Display Images ---
        
        # Display the raw color image
        cv2.imshow('Raw Image (Camera 4)', frame_raw)
        
        # Display the undistorted color image
        cv2.imshow('Undistorted Color', undistorted_frame)

        # Display the undistorted grayscale image (for further processing)
        cv2.imshow('Undistorted Grayscale', gray_undistorted)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_stream_reliable_undistort()