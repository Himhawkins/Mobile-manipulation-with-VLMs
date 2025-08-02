import cv2
import cv2.aruco as aruco
import numpy as np
import time

# ----- CONFIGURATION -----
MARKER_ID_TO_TRACK = 782
MARKER_DICT = aruco.DICT_ARUCO_ORIGINAL
CAMERA_INDEX = 0
FLOOR_LOWER = np.array([0, 0, 150])
FLOOR_UPPER = np.array([180, 60, 255])
CELL_PIXEL_SIZE = 30
PRINT_INTERVAL = 5  # seconds

# ----- INITIALIZE -----
cap = cv2.VideoCapture(CAMERA_INDEX)
aruco_dict = aruco.getPredefinedDictionary(MARKER_DICT)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
last_print_time = time.time()

print("Tracking ArUco marker... Press 'q' to quit.")

# Will store robot center each frame
cx, cy = None, None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width = frame.shape[:2]
    marker_mask = np.zeros((height, width), dtype=np.uint8)
    cx, cy = None, None  # reset per frame

    # --- ArUco Detection ---
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == MARKER_ID_TO_TRACK:
                pts = corners[i][0]
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())

                top_left = pts[0]
                top_right = pts[1]
                delta_x = top_right[0] - top_left[0]
                delta_y = top_right[1] - top_left[1]
                angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi

                cv2.polylines(frame, [pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{marker_id} ({cx},{cy})", (cx+10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Angle: {angle:.2f}", (cx + 10, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (238, 244, 44), 2)

                cv2.fillConvexPoly(marker_mask, pts.astype(int), 255)
                MARGIN = 60
                x, y, w, h = cv2.boundingRect(pts.astype(int))
                x1 = max(x - MARGIN, 0)
                y1 = max(y - MARGIN, 0)
                x2 = min(x + w + MARGIN, width)
                y2 = min(y + h + MARGIN, height)
                marker_mask[y1:y2, x1:x2] = 255

    # --- Floor Detection ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    floor_mask = cv2.inRange(hsv, FLOOR_LOWER, FLOOR_UPPER)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # --- Obstacle Detection ---
    combined_mask = cv2.bitwise_or(floor_mask, marker_mask)
    obstacle_mask = cv2.bitwise_not(combined_mask)

    # --- Draw red overlay on obstacle regions ---
    obstacle_pixels = obstacle_mask > 0
    overlay = np.zeros_like(frame)
    overlay[:, :] = (0, 0, 255)
    frame[obstacle_pixels] = cv2.addWeighted(frame[obstacle_pixels], 0.5,
                                             overlay[obstacle_pixels], 0.5, 0)

    # --- Dynamic Grid Generation ---
    grid_cols = max(1, width // CELL_PIXEL_SIZE)
    grid_rows = max(1, height // CELL_PIXEL_SIZE)
    resized_obstacle = cv2.resize(obstacle_mask, (grid_cols, grid_rows), interpolation=cv2.INTER_NEAREST)
    obstacle_grid = (resized_obstacle > 0).astype(int)

    # --- Add Robot Position to Grid (mark as 2) ---
    if cx is not None and cy is not None:
        grid_x = min(cx * grid_cols // width, grid_cols - 1)
        grid_y = min(cy * grid_rows // height, grid_rows - 1)
        obstacle_grid[grid_y, grid_x] = 2

    # --- Print Grid Every PRINT_INTERVAL Seconds ---
    now = time.time()
    if now - last_print_time > PRINT_INTERVAL:
        print(f"\nObstacle Grid ({grid_rows} rows × {grid_cols} cols):")
        print("Legend: 0 = free, 1 = obstacle, 2 = robot")
        for row in obstacle_grid:
            print(''.join(str(cell) for cell in row))
        last_print_time = now

    # --- Show Frame ---
    cv2.imshow("ArUco Tracker with Obstacles", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
