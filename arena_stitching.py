import cv2
import numpy as np
import json

from regex import sub
from arena_utils import warp_arena_frame, load_arena_settings

overlap = 0

def open_all_cameras(settings):
    caps = {}
    for cell_key, cell in settings["cells"].items():
        cam_id = int(cell["camera"])
        if cam_id not in caps:
            cap = cv2.VideoCapture(cam_id)
            if cap.isOpened():
                caps[cam_id] = cap
            else:
                print(f"Failed to open camera {cam_id}")
    return caps

def get_frame_from_camera(caps, cam_id):
    cap = caps.get(cam_id)
    if cap:
        ret, frame = cap.read()
        if ret:
            return frame
    return None

def draw_inset_dashed_border(frame, offset):
    """
    Draws a dashed rectangle with a hardcoded 30px internal offset on a frame.

    Args:
        frame: The input image/frame to draw on.

    Returns:
        A new frame with the dashed border drawn on it.
    """
    # Create a copy to avoid modifying the original frame passed to the function
    drawn_frame = frame.copy()

    # --- Hardcoded Parameters ---
    color = (0, 255, 255)  # Yellow color in BGR
    thickness = 2
    dash_length = 15
    gap_length = 7
    # --------------------------

    # Get frame dimensions
    height, width, _ = drawn_frame.shape

    # Calculate the rectangle's corners based on the internal offset
    top_left = (offset, offset)
    bottom_right = (width - offset, height - offset)
    
    # Get the four corner points
    points = [
        top_left, 
        (bottom_right[0], top_left[1]), 
        bottom_right, 
        (top_left[0], bottom_right[1])
    ]

    # Draw each of the four sides
    for i in range(4):
        p1 = points[i]
        p2 = points[(i + 1) % 4]
        
        line_length = np.linalg.norm(np.array(p2) - np.array(p1))
        if line_length == 0: continue
        unit_vector = (np.array(p2) - np.array(p1)) / line_length
        
        current_pos = np.array(p1, dtype=float)
        dist_covered = 0
        
        while dist_covered < line_length:
            dash_end = current_pos + unit_vector * dash_length
            if dist_covered + dash_length > line_length:
                dash_end = np.array(p2)
            
            # Draw the dash on the copied frame
            cv2.line(drawn_frame, tuple(current_pos.astype(int)), tuple(dash_end.astype(int)), color, thickness)
            
            current_pos += unit_vector * (dash_length + gap_length)
            dist_covered += dash_length + gap_length
            
    return drawn_frame

def stitch_arena(settings, caps):
    global overlap
    rows = settings["rows"]
    cols = settings["cols"]
    full_rows = []  # This list will still be used to create the final stitched image
    
    # 1. Initialize the new list of lists to hold the individual images
    grid_of_images = []
    subgrid_images = []

    for r in range(rows):
        row_images = []
        for c in range(cols):
            key = f"{r},{c}"
            cell = settings["cells"].get(key)
            overlap = int(cell.get("overlap", 0))

            if not cell:
                print(f"[WARN] No config for cell {key}. Using blank image.")
                width = int(settings.get("default_width", 640))
                height = int(settings.get("default_height", 480))
                blank = np.zeros((height, width, 3), dtype=np.uint8)
                row_images.append(blank)
                continue

            cam_id = int(cell.get("camera", -1))
            raw_frame = get_frame_from_camera(caps, cam_id)

            if raw_frame is not None:
                warped = warp_arena_frame(raw_frame, cell_key=key)
            else:
                print(f"[WARN] Camera {cam_id} unavailable for cell {key}. Using blank image.")
                width = int(cell.get("width", 640))
                height = int(cell.get("height", 480))
                warped = np.zeros((height, width, 3), dtype=np.uint8)

            # warped = draw_inset_dashed_border(warped, overlap)  # Draw dashed border
            row_images.append(warped)
            subgrid_images.append(warped)
        
        grid_of_images.append(subgrid_images)
        # The existing logic to stitch the full row can remain
        if row_images:
            base_h = row_images[0].shape[0]
            # Ensure all images in the row have the same height for hstack
            resized_row_images = [cv2.resize(img, (img.shape[1], base_h)) for img in row_images]
            stitched_row = np.hstack(resized_row_images)
            full_rows.append(stitched_row)

    # The existing logic to stack rows vertically can also remain
    if full_rows:
        # Before stacking, ensure all stitched rows have the same width
        base_w = max(row.shape[1] for row in full_rows)
        resized_full_rows = [cv2.resize(img, (base_w, img.shape[0])) for img in full_rows]
        stitched_full = np.vstack(resized_full_rows)
        
        # 3. Return the fully stitched image AND the new grid of images
        return stitched_full, grid_of_images
    else:
        # Return a consistent type if no images were processed
        return None, []

def detect_robot_pos(frame, overlap, r_idx, c_idx):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(frame)

    # --- 3. Process and display the results ---
    if ids is not None:
        # print(f"Detected {len(ids)} marker(s).")
        # Loop over the detected ArUco corners
        for i, marker_id in enumerate(ids):
            # The returned corners object is a list of arrays.
            # Extract the corners for the current marker.
            marker_corners = corners[i].reshape((4, 2))
            
            # Extract the individual corner points (top-left, top-right, etc.)
            (topLeft, topRight, bottomRight, bottomLeft) = marker_corners
            
            # --- 4. Calculate the center of the marker ---
            # Calculate the center (x, y)-coordinates of the ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            
            # Calculate the coordinates based on the overlap and row/column indices
            x_coord = (cX - overlap) + (c_idx * (frame.shape[1] - (2 * overlap)))
            y_coord = (cY - overlap) + (r_idx * (frame.shape[0] - (2 * overlap)))

            print(f"-> Marker ID: {marker_id[0]} | Center Coordinates: ({x_coord}, {y_coord})")

            # --- 5. Draw on the image for verification ---
            # Draw the bounding box of the ArUco detection
            cv2.polylines(frame, [marker_corners.astype(int)], True, (0, 255, 0), 2)
            
            # Draw the center of the marker
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

            # Draw the marker ID and coordinates
            text = f"ID: {marker_id[0]} ({x_coord}, {y_coord})"
            cv2.putText(frame, text, (int(topLeft[0]), int(topLeft[1]) - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # Placeholder for robot detection logic
    # For now, just return a dummy position
    return frame


# --- Main Loop ---
if __name__ == "__main__":
    settings = load_arena_settings()
    caps = open_all_cameras(settings)

    if not caps:
        print("No cameras available")
        exit()

    try:
        while True:
            stitched, full_images = stitch_arena(settings, caps)
            
            for r_idx, row_list in enumerate(full_images):
                # 2. The inner loop gets each individual image from the row_list
                for c_idx, img in enumerate(row_list):
                    
                    # Now, 'img' is a single image and your functions will work correctly
                    processed_img = detect_robot_pos(img, overlap, r_idx, c_idx)
                    processed_img = draw_inset_dashed_border(processed_img, overlap)

                    # Create a unique window name using both row and column index
                    window_name = f"Cell Image [{r_idx}][{c_idx}]"
                    cv2.imshow(window_name, processed_img)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
