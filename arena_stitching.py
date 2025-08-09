import cv2
import numpy as np
from arena_utils import warp_arena_frame, warp_arena_frame_extended, load_arena_settings, open_all_cameras

current_overlap = 0

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
    global current_overlap
    rows = settings["rows"]
    cols = settings["cols"]
    full_rows = []
    
    grid_of_images = []

    for r in range(rows):
        row_images = []
        sub_grid = []
        for c in range(cols):
            key = f"{r},{c}"
            cell = settings["cells"].get(key)
            current_overlap = int(cell.get("overlap", 0))

            # ... (logic to get 'warped' image) ...
            cam_id = int(cell.get("camera", -1))
            raw_frame = get_frame_from_camera(caps, cam_id)
            if raw_frame is not None:
                warped_ex = warp_arena_frame_extended(raw_frame, cell_key=key)
                warped = warp_arena_frame(raw_frame, cell_key=key)
            else:
                width = int(cell.get("width", 640))
                height = int(cell.get("height", 480))
                warped_ex = np.zeros((height, width, 3), dtype=np.uint8)
            
            row_images.append(warped)
            sub_grid.append(warped_ex)
        
        grid_of_images.append(sub_grid)
        
        if row_images:
            base_h = row_images[0].shape[0]
            resized_row_images = [cv2.resize(img, (img.shape[1], base_h)) for img in row_images]
            stitched_row = np.hstack(resized_row_images)
            full_rows.append(stitched_row)

    if full_rows:
        base_w = max(row.shape[1] for row in full_rows)
        resized_full_rows = [cv2.resize(img, (base_w, img.shape[0])) for img in full_rows]
        stitched_full = np.vstack(resized_full_rows)
        return stitched_full, grid_of_images
    else:
        return None, []

def detect_robot_pos(frame, target_id, overlap, r_idx, c_idx):
    """
    Detect a specific ArUco marker and calculate global (x,y) and orientation (theta, radians).
    Returns (processed_frame, x, y, theta). If not found, x/y/theta are None.
    """
    x_coord, y_coord, theta_rad = None, None, None

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        for i, marker_id in enumerate(ids):
            if marker_id[0] == target_id:
                marker_corners = corners[i].reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = marker_corners

                # center in the cell image
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                # global coordinates in the stitched layout (accounting for overlap)
                x_coord = (cX - overlap) + (c_idx * (frame.shape[1] - (2 * overlap)))
                y_coord = (cY - overlap) + (r_idx * (frame.shape[0] - (2 * overlap)))

                # --- orientation (radians), image x points right, y points down ---
                # Vector along marker's x-axis: topLeft -> topRight
                v = topRight - topLeft
                theta_rad = float(np.arctan2(v[1], v[0]))  # in [-pi, pi]

                # Draw
                cv2.polylines(frame, [marker_corners.astype(int)], True, (0, 255, 0), 2)
                cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

                # Optional: draw a small heading arrow
                arrow_len = 40
                tip = (int(cX + arrow_len * np.cos(theta_rad)),
                       int(cY + arrow_len * np.sin(theta_rad)))
                cv2.arrowedLine(frame, (cX, cY), tip, (255, 0, 255), 2, tipLength=0.3)

                text = f"ID:{marker_id[0]} ({x_coord},{y_coord}) θ:{theta_rad:.2f}rad"
                cv2.putText(frame, text, (int(topLeft[0]), int(topLeft[1]) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                break

    return frame, x_coord, y_coord, theta_rad

def find_robot_in_arena(target_id, settings, caps, save_path=None):
    global current_overlap
    stitched_img, image_grid = stitch_arena(settings, caps)
    if not image_grid:
        return None, {}, None, None, None

    processed_frames = {}
    x_coordinates, y_coordinates = [], []
    thetas = []  # collect per-cell theta

    for r_idx, row_list in enumerate(image_grid):
        for c_idx, img in enumerate(row_list):
            processed_img, x_pos, y_pos, theta_rad = detect_robot_pos(
                img, target_id, current_overlap, r_idx, c_idx
            )

            if x_pos is not None and y_pos is not None:
                x_coordinates.append(x_pos)
                y_coordinates.append(y_pos)
            if theta_rad is not None:
                thetas.append(theta_rad)

            processed_img = draw_inset_dashed_border(processed_img, current_overlap)
            cell_key = f"{r_idx},{c_idx}"
            processed_frames[cell_key] = processed_img

    final_x = sum(x_coordinates) / len(x_coordinates) if x_coordinates else None
    final_y = sum(y_coordinates) / len(y_coordinates) if y_coordinates else None

    # Circular mean for orientation (handle wrap-around)
    rotated_theta = None
    if thetas:
        c = np.mean(np.cos(thetas))
        s = np.mean(np.sin(thetas))
        final_theta = float(np.arctan2(s, c))  # still in [-pi, pi]
        # Rotate left by 90° (π/2 radians)
        rotated_theta = final_theta - np.pi / 2
        rotated_theta = (rotated_theta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    final_pose = (int(final_x), int(final_y), rotated_theta) if final_x is not None and final_y is not None else None

    if save_path is not None:
        with open(save_path, "w") as f:
            if final_pose:
                f.write(f"{int(final_x)},{int(final_y)},{rotated_theta}\n")

    return stitched_img, processed_frames, final_pose


# --- Main Loop ---
if __name__ == "__main__":
    settings = load_arena_settings()
    caps = open_all_cameras(settings)

    if not caps:
        print("No cameras available")
        exit()

    robot_marker_id = 782

    try:
        while True:
            stitched, processed_frames, pose = find_robot_in_arena(robot_marker_id, settings, caps)
            final_x, final_y, final_theta = pose 

            if final_x is not None and final_y is not None:
                center_point = (int(final_x), int(final_y))
                cv2.circle(stitched, center_point, radius=8, color=(0, 0, 255), thickness=-1)

                # Optional heading arrow on the stitched image if theta available
                if final_theta is not None:
                    L = 60
                    tip = (int(final_x + L * np.cos(final_theta)),
                        int(final_y + L * np.sin(final_theta)))
                    cv2.arrowedLine(stitched, center_point, tip, (0, 255, 0), 2, tipLength=0.3)


                if final_theta is not None:
                    print(f"Robot Position: ({final_x:.2f}, {final_y:.2f}), θ={final_theta:.3f} rad")
                else:
                    print(f"Robot Position: ({final_x:.2f}, {final_y:.2f}), θ=NA")
            else:
                print("Robot not detected in any cell.")

            if stitched is not None:
                cv2.imshow("Stitched Arena", stitched)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
