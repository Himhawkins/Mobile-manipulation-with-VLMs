import numpy as np
import cv2
import cv2.aruco as aruco
import os

# --- CONFIGURATION (MUST MATCH calibrate_camera.py) ---
# --- Board Details ---

# --- Camera Details ---
CAMERA_ID = 4

         # Use 2 for the camera

# --- ArUco Dictionary ---
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

k=305
actual_objs={0:[0,0,0],
             1:[-3,0,0],
             2:[-7,0,0],
             3:[-7,3,0],
             4:[-3,3,0],
             5:[6,0,0],
             6:[3,0,0],
             7:[3,2,0],
             8:[6,2,0]}

for i in actual_objs.keys():
    t=actual_objs[i]
    u=[]
    u.append([k*t[0]+30,k*t[1]-20,k*t[2]])
    u.append([k*t[0]+30+143,k*t[1]-20,k*t[2]])
    u.append([k*t[0]+30+143,k*t[1]-20-143,k*t[2]])
    u.append([k*t[0]+30,k*t[1]-20-143,k*t[2]])
    
    actual_objs[i]=np.array(u,dtype=np.float32)
print(actual_objs)

def find_homography_live():
    """
    Calculates the homography from the camera to the floor plane using a live feed.
    Returns a function that can transform image coordinates to real-world coordinates.
    """
    # --- Load Camera Calibration ---
    calib_file = f"camera_calibration_data_{CAMERA_ID}.npz"
    if not os.path.exists(calib_file):
        print(f"Error: Calibration file '{calib_file}' not found.")
        print("Please run the calibration script first.")
        return None
        
    print(f"Loading calibration data from {calib_file}...")
    data = np.load(calib_file)
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    print("Calibration data loaded successfully.")

    # --- Setup Board and Detector ---
    detector_params = aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    

    aruco_detector = aruco.ArucoDetector(ARUCO_DICT, detectorParams=detector_params)

    # --- Start Camera ---
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {CAMERA_ID}.")
        return None
        
    # Set camera focus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
    cap.set(cv2.CAP_PROP_FOCUS, 0)

    print("\n--- Live Homography Setup ---")
    print("Place the ChArUco board flat on the floor.")
    print("Position your camera in its final, fixed location.")
    print("Press 's' to save the transformation. Press 'q' to quit.")

    homography_matrix = None
    last_seen_corners = None
    last_seen_ids = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
            
        # Undistort the live view for accurate positioning
        undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
        
        if marker_ids is not None and len(marker_ids) > 3:
            aruco.drawDetectedMarkers(undistorted_frame, marker_corners, marker_ids)
            last_seen_corners = marker_corners
            last_seen_ids = marker_ids
        
        cv2.putText(undistorted_frame, "Press 's' to set homography. 'q' to quit.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Homography Setup", undistorted_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if last_seen_ids is None:
                print("Error: No markers detected. Cannot calculate homography.")
                continue

            print("\nCalculating homography...")
            
            # --- Match 2D image points with 3D real-world points ---
            obj_points = []
            img_points = []

            # board_obj_points = charuco_board.getObjPoints()
            # print(board_obj_points)
            for i, marker_id in enumerate(last_seen_ids.flatten()):
                if marker_id in actual_objs.keys():
                    obj_points.append(actual_objs[marker_id])
                    img_points.append(last_seen_corners[i])
            
            if len(obj_points) < 4:
                print("Error: Not enough markers found to create a reliable homography.")
                continue

            # --- CORRECTED: Reshape arrays to the simple 2D format findHomography expects ---
            # The error message indicates the shape is the problem.
            obj_points_np = np.concatenate(obj_points)
            img_points_np = np.concatenate(img_points).reshape(-1, 2) # Reshape to (N, 2)

            # We use only the (x, y) coordinates for the planar homography
            homography_matrix, _ = cv2.findHomography(img_points_np, obj_points_np[:, :2])
            
            if homography_matrix is not None:
                print("Homography calculated successfully!")

                # --- ADDED: Reprojection Error Calculation ---
                # Project the real-world points back to the image plane using the homography's inverse
                reprojected_img_points = cv2.perspectiveTransform(
                    obj_points_np[:, :2].reshape(-1, 1, 2), 
                    np.linalg.inv(homography_matrix)
                )
                
                # Calculate the error (distance between original and reprojected points)
                error = cv2.norm(
                    img_points_np.reshape(-1, 1, 2), 
                    reprojected_img_points, 
                    cv2.NORM_L2
                )
                
                # Calculate the average error per point
                mean_error_per_point = error / len(img_points_np)
                print(f"Mean Reprojection Error: {mean_error_per_point:.4f} pixels")
                # --- END: Error Calculation ---

                print(homography_matrix)
                save_homography(homography_matrix)
                break
            else:
                print("Could not calculate homography.")

    cap.release()
    cv2.destroyAllWindows()

    if homography_matrix is None:
        print("Homography was not set.")
        return None
    
def save_homography(homography_matrix, camera_id=CAMERA_ID):
    """
    Saves the calculated homography matrix to an NPZ file.
    """
    filepath = f"homography_matrix_{camera_id}.npz"
    print(f"Saving homography matrix to {filepath}...")
    try:
        np.savez_compressed(filepath, homography=homography_matrix)
        print("Homography matrix saved successfully.")
    except Exception as e:
        print(f"Error saving homography matrix: {e}")

def load_homography(camera_id):
    """
    Loads a previously saved homography matrix from an NPZ file.
    Returns the matrix or None if the file is not found or corrupted.
    """
    filepath = f"homography_matrix_{camera_id}.npz"
    if not os.path.exists(filepath):
        print(f"Homography file '{filepath}' not found.")
        return None
    
    print(f"Loading homography matrix from {filepath}...")
    try:
        data = np.load(filepath)
        homography_matrix = data['homography']
        print("Homography matrix loaded successfully.")
        return homography_matrix
    except Exception as e:
        print(f"Error loading homography matrix: {e}")
        return None


    # --- Define the final transformation function ---
    def get_planar_coords(pixel_coord):
        """
        Transforms a single (u, v) pixel coordinate from the raw image
        into a real-world (x, y) coordinate in millimeters.
        """
        # Create a point in the format required by perspectiveTransform
        pixel_point = np.array([[pixel_coord]], dtype=np.float32)
        
        # First, we must undistort the pixel coordinate
        undistorted_pixel = cv2.undistortPoints(pixel_point, camera_matrix, dist_coeffs, P=camera_matrix)
        
        # Then, apply the homography
        real_world_point = cv2.perspectiveTransform(undistorted_pixel, homography_matrix)
        
        return real_world_point[0][0]

    # --- DEMONSTRATION ---
    # Take the first corner of the first detected marker as an example
    if last_seen_corners:
        example_pixel = tuple(last_seen_corners[0].flatten()[:2])
        planar_coords = get_planar_coords(example_pixel)
        print(f"\n--- DEMO ---")
        print(f"Pixel coordinate {example_pixel} corresponds to...")
        print(f"Real-world coordinate: ({planar_coords[0]:.2f}, {planar_coords[1]:.2f}) mm")
        
    return get_planar_coords

if __name__ == '__main__':
    transform_function = find_homography_live()

    if transform_function:
        print("\nTransformation function is ready to use.")
        # Example of how you would use it:
        # my_pixel = (320, 240) 
        # real_coords = transform_function(my_pixel)
        # print(f"The real world coordinate for {my_pixel} is {real_coords}")

