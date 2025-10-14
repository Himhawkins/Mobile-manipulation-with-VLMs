import cv2
import os
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

            cam_id = int(cell.get("camera", -1))
            width = int(cell.get("width", 640))
            height = int(cell.get("height", 480))

            # Default black images so 'warped' and 'warped_ex' always exist
            warped = np.zeros((height, width, 3), dtype=np.uint8)
            warped_ex = np.zeros((height, width, 3), dtype=np.uint8)

            raw_frame = get_frame_from_camera(caps, cam_id)
            if raw_frame is not None:
                warped_ex = warp_arena_frame_extended(raw_frame, cell_key=key)
                warped = warp_arena_frame(raw_frame, cell_key=key)
            else:
                print(f"[WARN] Camera {cam_id} failed to grab a frame, removing from caps.")
                if cam_id in caps:
                    caps[cam_id].release()
                    del caps[cam_id]

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


def detect_robot_poses(frame, overlap, r_idx, c_idx):
    """
    Detect ALL ArUco markers in a single cell `frame`.
    Returns:
        processed_frame,
        detections: list of dicts [{id:int, x:int, y:int, theta:float (radians)}]
                    (x,y) are global arena coords accounting for overlap/grid.
                    theta is marker x-axis heading in image coords, rotated -90° to
                    match your prior convention, and wrapped to [-pi, pi].
    """
    detections = []

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None and len(ids) > 0:
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i].reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = marker_corners

            # center in the cell image
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)

            # global coordinates in the stitched layout (accounting for overlap)
            x_coord = (cX - overlap) + (c_idx * (frame.shape[1] - (2 * overlap)))
            y_coord = (cY - overlap) + (r_idx * (frame.shape[0] - (2 * overlap)))

            # orientation (radians) from marker x-axis: topLeft -> topRight
            v = topRight - topLeft
            theta = float(np.arctan2(v[1], v[0]))  # [-pi, pi]

            # rotate left by 90° (π/2) to match prior convention
            theta = theta - np.pi / 2.0
            theta = (theta + np.pi) % (2 * np.pi) - np.pi  # wrap [-pi, pi]

            # Draw annotations
            cv2.polylines(frame, [marker_corners.astype(int)], True, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            arrow_len = 40
            tip = (int(cX + arrow_len * np.cos(theta)),
                   int(cY + arrow_len * np.sin(theta)))
            cv2.arrowedLine(frame, (cX, cY), tip, (255, 0, 255), 2, tipLength=0.3)

            text = f"ID:{marker_id} ({x_coord},{y_coord}) θ:{theta:.2f}rad"
            cv2.putText(frame, text, (int(topLeft[0]), int(topLeft[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            detections.append({
                "id": int(marker_id),
                "x": int(x_coord),
                "y": int(y_coord),
                "theta": float(theta),
            })

    return frame, detections

def find_robots_in_arena(settings, caps, save_path=None, append=False):
    """
    Stitches the arena, scans ALL cells, detects ALL ArUco markers,
    merges multi-cell sightings per marker ID, and optionally appends to file.

    Returns:
        stitched_img,
        processed_frames: dict["r,c"] -> annotated cell image,
        merged: dict[id] -> {"x": int, "y": int, "theta": float}
               (averaged across sightings using circular mean for theta)
    """
    global current_overlap
    stitched_img, image_grid = stitch_arena(settings, caps)
    if not image_grid:
        return None, {}, {}

    processed_frames = {}
    pos_acc, ang_acc = {}, {}

    for r_idx, row_list in enumerate(image_grid):
        for c_idx, img in enumerate(row_list):
            processed_img, dets = detect_robot_poses(img, current_overlap, r_idx, c_idx)
            processed_img = draw_inset_dashed_border(processed_img, current_overlap)
            processed_frames[f"{r_idx},{c_idx}"] = processed_img

            for d in dets:
                mid = d["id"]
                if mid == 0:
                    continue
                pos_acc.setdefault(mid, []).append((d["x"], d["y"]))
                ang_acc.setdefault(mid, []).append(d["theta"])

    merged = {}
    for mid in pos_acc.keys():
        xs = [p[0] for p in pos_acc[mid]]
        ys = [p[1] for p in pos_acc[mid]]
        cs = np.mean(np.cos(ang_acc[mid]))
        ss = np.mean(np.sin(ang_acc[mid]))
        theta_mean = float(np.arctan2(ss, cs))
        theta_mean = (theta_mean + np.pi) % (2 * np.pi) - np.pi
        merged[mid] = {
            "x": int(round(np.mean(xs))),
            "y": int(round(np.mean(ys))),
            "theta": theta_mean,
        }

    # ---- SAVE DETECTIONS ----
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

        if len(merged) == 0:
            # Clear file when no robots detected
            open(save_path, "w").close()
        else:
            mode = "a" if append else "w"
            with open(save_path, mode) as f:
                for mid in sorted(merged.keys()):
                    m = merged[mid]
                    f.write(f"{mid},{m['x']},{m['y']},{m['theta']}\n")

    return stitched_img, processed_frames, merged

# --- Main Loop ---
if __name__ == "__main__":
    settings = load_arena_settings()
    caps = open_all_cameras(settings)

    if not caps:
        print("No cameras available")
        exit()

    robot_marker_id = 3

    try:
        while True:
            stitched, processed_frames_list, pose = find_robots_in_arena(settings, caps)
            if pose is None:
                print("Robot not detected in any cell.")
                final_x, final_y, final_theta = None, None, None
            else:
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
            
            for key, frame in processed_frames_list.items():
                cv2.imshow(f"Cell {key}", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                break
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
