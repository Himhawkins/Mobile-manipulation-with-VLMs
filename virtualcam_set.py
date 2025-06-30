import cv2
import pyvirtualcam

# Open the physical camera
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise RuntimeError("Could not start camera.")

# Get camera properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a virtual camera
with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
    print(f"Using virtual camera: {cam.device}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Send the frame to the virtual camera
        cam.send(frame_rgb)

        # Wait for the next frame
        cam.sleep_until_next_frame()

cap.release()
