import cv2
import cv2.aruco as aruco
import numpy as np

# ----- CONFIGURATION -----
MARKER_ID_TO_TRACK = 782  # Set this to the ID of your ArUco marker
MARKER_DICT = aruco.DICT_ARUCO_ORIGINAL  # Can be changed depending on what you printed
CAMERA_INDEX = 2  # Change to video file path if using footage

# ----- INITIALIZE -----
cap = cv2.VideoCapture(CAMERA_INDEX)
aruco_dict = aruco.getPredefinedDictionary(MARKER_DICT)
parameters = aruco.DetectorParameters()

detector = aruco.ArucoDetector(aruco_dict, parameters)

print("Tracking ArUco marker... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect markers
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == MARKER_ID_TO_TRACK:
                pts = corners[i][0]  # 4 corners of the marker
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())

                # Get top-left and top-right corners
                top_left = pts[0]
                top_right = pts[1]

                # Compute angle in degrees
                delta_x = top_right[0] - top_left[0]
                delta_y = top_right[1] - top_left[1]
                angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi

                # Draw marker outline and center
                cv2.polylines(frame, [pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{marker_id} ({cx},{cy})", (cx+10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Angle: {angle:.2f}", (cx + 10, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (238, 244, 44), 2)
                print(f"Marker ID {marker_id}: Center = ({cx}, {cy}), Angle = {angle:.2f}°")

    cv2.imshow("ArUco Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
