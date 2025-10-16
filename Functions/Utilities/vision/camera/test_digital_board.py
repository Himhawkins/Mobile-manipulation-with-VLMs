import cv2
from cv2 import aruco
import numpy as np
import time

# --- ChArUco Board Details ---
CHARUCO_SQUARES_X = 6       # How many squares wide is your board?
CHARUCO_SQUARES_Y = 4       # How many squares tall is your board?
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# IMPORTANT: You MUST measure your new printed board and update these values
# The size of a single square's side that you measured in millimeters
SQUARE_SIZE_MM = 40.0
# The size of the ArUco marker's side that you measured in millimeters
MARKER_SIZE_MM = 30.0

# --- Camera Details ---
CAMERA_ID = 2               # The ID of the camera to use

def main():
    """
    Performs camera calibration using a ChArUco board.
    The user can interactively capture frames for calibration.
    """
    # Initialize camera capture
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {CAMERA_ID}.")
        return

    # Set a higher resolution if possible (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(2) # Give camera time to initialize

    # Create the ChArUco board object
    board = aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        SQUARE_SIZE_MM,
        MARKER_SIZE_MM,
        ARUCO_DICT
    )

    # Create the modern Aruco detector and Charuco detector
    detector_params = aruco.DetectorParameters()
    charuco_params = aruco.CharucoParameters()
    # You can refine the corners for better accuracy
    charuco_params.tryRefineMarkers = True 
    
    detector = aruco.ArucoDetector(ARUCO_DICT, detector_params)
    charuco_detector = aruco.CharucoDetector(board, charuco_params, detector_params)

    # Lists to store points from all calibration images
    all_charuco_corners = []
    all_charuco_ids = []
    
    img_size = None # To be determined from the first frame

    # --- Create a window and trackbars for threshold adjustment ---
    WINDOW_NAME = "ChArUco Calibration"
    cv2.namedWindow(WINDOW_NAME)

    def on_trackbar(val):
        # This function is a placeholder for the trackbar callback.
        pass

    # Create trackbars for adaptive thresholding parameters
    # Block Size will be calculated as (val * 2 + 3) to ensure it's always odd and >= 3
    cv2.createTrackbar('Block Size', WINDOW_NAME, 5, 50, on_trackbar)
    cv2.createTrackbar('C Constant', WINDOW_NAME, 2, 50, on_trackbar)

    print("Starting camera feed. Adjust sliders for best thresholding.")
    print("Press 'SPACE' to capture a valid board view.")
    print("Press 'q' to quit and perform calibration.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        if img_size is None:
            img_size = frame.shape[:2]

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- Pre-processing for better detection ---
        # Get current trackbar positions
        block_size_val = cv2.getTrackbarPos('Block Size', WINDOW_NAME)
        c_constant_val = cv2.getTrackbarPos('C Constant', WINDOW_NAME)

        # Block size must be an odd number >= 3
        block_size = block_size_val * 2 + 3

        # Apply a Gaussian blur to reduce noise and improve thresholding
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Apply adaptive thresholding to binarize the image, helping with lighting variations
        thresh_frame = cv2.adaptiveThreshold(
            blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c_constant_val
        )

        # Detect the board on the thresholded image
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(thresh_frame)

        # --- Create display frame from the processed image ---
        # Convert the single-channel threshold image back to BGR to draw colored overlays
        display_frame = cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR)

        # Draw detections on the display frame
        if charuco_ids is not None and len(charuco_ids) > 0:
            aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids)

        # Display instructions and capture count
        capture_text = f"Captures: {len(all_charuco_corners)}"
        cv2.putText(display_frame, capture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 'SPACE' to capture, 'q' to calibrate", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            if charuco_ids is not None and len(charuco_ids) > 3:
                print(f"Capture {len(all_charuco_corners) + 1} successful!")
                # Add the detected points to our lists
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
            else:
                print("Capture failed: Not enough corners detected.")

    cap.release()
    cv2.destroyAllWindows()

    # --- Perform Calibration ---
    print("\nStarting calibration process...")
    if len(all_charuco_corners) < 5:
        print("Calibration failed: Need at least 5 good captures.")
        return

    # Get object points for the board
    obj_points = board.getObjPoints()
    all_obj_points = [obj_points for _ in all_charuco_ids]

    # Calibrate the camera
    calibration_flags = cv2.CALIB_RATIONAL_MODEL
    ret, camera_matrix, dist_coeffs, rvecs, tvecs, std_devs_intrinsics, std_devs_extrinsics, per_view_errors = cv2.calibrateCameraExtended(
        objectPoints=all_obj_points,
        imagePoints=all_charuco_corners,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=calibration_flags
    )
    
    # Calculate the mean reprojection error
    mean_error = np.mean(per_view_errors)

    print("\n--- Calibration Results ---")
    if ret:
        print(f"Calibration successful with a reprojection error of: {mean_error:.4f} pixels")
        print("\nCamera Matrix:\n", camera_matrix)
        print("\nDistortion Coefficients:\n", dist_coeffs)
    else:
        print("Calibration failed.")

if __name__ == "__main__":
    main()

