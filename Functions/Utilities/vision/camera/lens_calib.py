import numpy as np
import cv2
import cv2.aruco as aruco
import os

# --- CONFIGURATION ---
# --- Board Details ---
# IMPORTANT: These MUST match your printed board's properties
CHARUCO_SQUARES_X = 5       # How many squares wide is your board?
CHARUCO_SQUARES_Y = 7       # How many squares tall is your board?
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# IMPORTANT: You MUST measure your new printed board and update these values
# The size of a single square's side that you measured in millimeters
SQUARE_SIZE_MM = 34 
# The size of the ArUco marker's side that you measured in millimeters
MARKER_SIZE_MM = 26 

# --- Camera Details ---
CAMERA_ID = 2               # Use 2 for the camera
NUM_CALIBRATION_IMAGES = 20
MAX_REPROJECTION_ERROR = 1.0 # The highest allowed error to save the calibration

def save_camera_calibration(camera_matrix, dist_coeffs, filename=f"camera_calibration_data_{CAMERA_ID}.npz"):
    """Saves the camera matrix and distortion coefficients to a .npz file."""
    print(f"Saving camera calibration data to {filename}...")
    np.savez(filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print("Data saved successfully.")

def nothing(x):
    """Callback function for trackbars."""
    pass

def calibrate_camera_interactive():
    """Performs camera calibration interactively using a ChArUco board."""
    print(f"Using OpenCV version: {cv2.__version__}")

    detector_params = aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX 
    refine_params = aruco.RefineParameters()

    charuco_board = aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        SQUARE_SIZE_MM,
        MARKER_SIZE_MM,
        ARUCO_DICT
    )
    charuco_detector = aruco.CharucoDetector(charuco_board, detectorParams=detector_params, refineParams=refine_params)

    all_charuco_corners = []
    all_charuco_ids = []
    
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {CAMERA_ID}.")
        return None, None

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
    cap.set(cv2.CAP_PROP_FOCUS, 255) 

    cv2.namedWindow('Camera Calibration')
    cv2.namedWindow('Threshold Settings')
    cv2.createTrackbar('Threshold Value', 'Threshold Settings', 127, 255, nothing)
    cv2.createTrackbar('Blur Kernel', 'Threshold Settings', 2, 10, nothing) # k=2 -> (5,5) kernel

    print("\n--- Interactive ChArUco Calibration ---")
    print("Use the sliders to make markers solid black in the 'Live Threshold' window.")
    print("Press 'c' to capture a view. Press 'q' to quit.")

    while len(all_charuco_ids) < NUM_CALIBRATION_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur_k = cv2.getTrackbarPos('Blur Kernel', 'Threshold Settings')
        kernel_size = 2 * blur_k + 1
        
        blurred_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        thresh_val = cv2.getTrackbarPos('Threshold Value', 'Threshold Settings')
        
        _, processed_gray = cv2.threshold(blurred_gray, thresh_val, 255, cv2.THRESH_BINARY)
        
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(processed_gray)
        
        found_corners = False
        if charuco_ids is not None and len(charuco_ids) > 4:
            found_corners = True
            aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

        status_text = f"Captured: {len(all_charuco_ids)}/{NUM_CALIBRATION_IMAGES}"
        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if found_corners:
            cv2.putText(frame, "Ready! Press 'c' to capture.", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Aim at ChArUco board...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Camera Calibration', frame)
        cv2.imshow('Live Threshold', processed_gray) # Show the live thresholded image
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Calibration cancelled by user.")
            break
            
        elif key == ord('c') and found_corners:
            print(f"Capturing points... ({len(all_charuco_ids) + 1}/{NUM_CALIBRATION_IMAGES})")
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

    cap.release()
    cv2.destroyAllWindows()

    if len(all_charuco_ids) < 5:
        print("\nCalibration failed. Not enough points were captured.")
        return None, None
        
    print(f"\nCollected {len(all_charuco_ids)} views. Calibrating camera...")
    
    image_size = gray.shape[::-1]

    # --- Manually prepare object points for the generic calibrateCamera function ---
    all_object_points = []
    all_board_corners_3d = charuco_board.getChessboardCorners()
    for ids in all_charuco_ids:
        obj_points_for_view = all_board_corners_3d[ids.flatten()]
        all_object_points.append(obj_points_for_view)

    # --- FIXED: Ensure correct data types for calibration function ---
    # The cv2.calibrateCamera function is very specific about requiring float32 data types.
    # We convert each array in the list to the correct type to prevent errors.
    all_object_points = [op.astype(np.float32) for op in all_object_points]
    all_charuco_corners = [cc.astype(np.float32) for cc in all_charuco_corners]

    # --- Using the core cv2.calibrateCamera for better compatibility ---
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points,
        all_charuco_corners,
        image_size,
        None,
        None
    )

    if ret:
        print("Camera calibrated!")
        # The return value from cv2.calibrateCamera is the total reprojection error.
        print(f"Total Reprojection Error: {ret}")
        if ret <= MAX_REPROJECTION_ERROR:
            print("Reprojection error is low. This is a good calibration!")
            save_camera_calibration(mtx, dist)
        else:
            print(f"Error is too high! (>{MAX_REPROJECTION_ERROR}). Calibration file NOT saved.")
        return mtx, dist
    else:
        print("Camera calibration failed.")
        return None, None

if __name__ == '__main__':
    calibrate_camera_interactive()

