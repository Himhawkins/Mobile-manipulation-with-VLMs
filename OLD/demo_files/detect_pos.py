import cv2
import cv2.aruco as aruco
import numpy as np

# ----- CONFIGURATION -----
MARKER_ID_TO_TRACK = 782      # Set this to the ID of your ArUco marker
MARKER_DICT = aruco.DICT_ARUCO_ORIGINAL  # Can be changed depending on what you printed
CAMERA_INDEX = 4             # Change to a video file path if using footage
OUTPUT_FILENAME = "robot_pos.txt"

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

    # If markers are detected, and our specific marker is among them
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == MARKER_ID_TO_TRACK:
                # Get the four corner points of the marker
                pts = corners[i][0]

                # Calculate the center (x, y) of the marker
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())

                # Get top-left and top-right corners for angle calculation
                top_left = pts[0]
                top_right = pts[1]

                # Calculate the angle (theta) in radians
                delta_x = top_right[0] - top_left[0]
                delta_y = top_right[1] - top_left[1]
                angle_rad = np.arctan2(delta_y, delta_x) + (np.pi / 2)

                # --- MODIFICATION: Write x, y, theta to file ---
                try:
                    with open(OUTPUT_FILENAME, "w") as f:
                        f.write(f"{cx},{cy},{angle_rad}")
                except IOError as e:
                    print(f"Error writing to file {OUTPUT_FILENAME}: {e}")
                # ----------------------------------------------------

                # --- Draw visualizations on the frame ---
                # Draw marker outline
                cv2.polylines(frame, [pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
                # Draw marker center
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                # Display marker ID and coordinates
                cv2.putText(frame, f"ID:{marker_id} ({cx},{cy})", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Display angle in radians
                cv2.putText(frame, f"Angle: {angle_rad:.2f} rad", (cx + 10, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (238, 244, 44), 2)

                # Print to console for debugging
                print(f"Marker ID {marker_id}: Center = ({cx}, {cy}), Angle = {angle_rad:.4f} radians")

    # Display the resulting frame
    cv2.imshow("ArUco Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print(f"Tracking stopped. Final position saved to {OUTPUT_FILENAME}.")
