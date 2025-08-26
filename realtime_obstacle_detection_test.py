import cv2
import numpy as np

# ------ Tunables ------
CAM_INDEX = 2                 # camera id
BLUR_KSIZE = (5, 5)           # Gaussian blur kernel
DIFF_THRESH = 40 #25          # pixel intensity threshold for motion
MIN_AREA = 300                # min contour area to count as obstacle (px)
MORPH_KERNEL = (3, 3)         # morphology kernel size
ADAPTIVE_BG = False #True     # slowly adapt background to scene
ADAPT_ALPHA = 0.01            # speed of adaptation (0..1); higher = faster
# ----------------------

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera")

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
    return gray

base = None
print("Press 'b' to (re)capture base background; 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_disp = frame.copy()
    gray = preprocess(frame)

    # Initialize or recapture base when requested
    if base is None:
        base = gray.copy().astype("float")
        cv2.putText(frame_disp, "Capturing base... press 'b' to recapture",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("frame", frame_disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            base = gray.copy().astype("float")
        continue

    # Background model as float for accumulateWeighted
    base_uint8 = cv2.convertScaleAbs(base)

    # Absolute difference
    diff = cv2.absdiff(gray, base_uint8)

    # Threshold
    _, mask = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)

    # Morphology to clean noise
    k = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)

    # Find moving blobs (possible "unknown obstacles")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame_disp, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame_disp, f"Obstacle (area={int(area)})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        detected = True

    # Show diagnostics
    cv2.putText(frame_disp, "Press 'b' to reset background | 'q' to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if detected:
        cv2.putText(frame_disp, "UNKNOWN OBSTACLE DETECTED", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("frame", frame_disp)
    cv2.imshow("diff", diff)
    cv2.imshow("mask", mask)

    # Slowly adapt background to handle small lighting changes
    if ADAPTIVE_BG:
        cv2.accumulateWeighted(gray, base, ADAPT_ALPHA)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        base = gray.copy().astype("float")

cap.release()
cv2.destroyAllWindows()
