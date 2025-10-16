import cv2
import numpy as np
import sys
import time

# --- Configuration ---
CAM_ID_A = 2 
CAM_ID_B = 4 

# TROUBLESHOOTING STEP: Lower resolution for stability
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

FINAL_FIXED_SIZE = (1280, 720) 

# --- Initialization ---
# TROUBLESHOOTING STEP: Explicitly set the backend API for stability (e.g., cv2.CAP_DSHOW for Windows)
# If on Linux, try cv2.CAP_V4L2.
API_PREFERENCE = cv2.CAP_DSHOW # Change this based on your OS if needed

# Open cameras with the explicit API preference
cap_A = cv2.VideoCapture(CAM_ID_A)
cap_B = cv2.VideoCapture(CAM_ID_B)

if not cap_A.isOpened() or not cap_B.isOpened():
    print(f"[ERROR] Could not open one or both cameras (IDs {CAM_ID_A} and {CAM_ID_B}).")
    print("Please check camera IDs and connections.")
    sys.exit(1)

# Set lower resolution for stability
cap_A.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_A.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap_B.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_B.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Optional: Try setting a lower frame rate
cap_B.set(cv2.CAP_PROP_FPS, 15) # For Cam 4, try forcing a lower FPS

print(f"[INFO] Cameras capturing at: {cap_A.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap_A.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

stitcher = cv2.Stitcher_create()
print("[INFO] Stitcher initialized. Starting live feed...")

# --- Main Loop ---
while True:
    retA, frameA = cap_A.read()
    retB, frameB = cap_B.read()
    time.sleep(2)
    if not retA or not retB:
        # If read fails, try again without crashing
        time.sleep(0.05)
        continue

    images_to_stitch = [frameA, frameB]

    # 1. Perform Stitching
    (status, stitched) = stitcher.stitch(images_to_stitch)

    # 2. Process and Display Result
    if status == cv2.Stitcher_OK:
        
        # Resize to Final Fixed Output Size
        final_image = cv2.resize(
            stitched, 
            FINAL_FIXED_SIZE, 
            interpolation=cv2.INTER_LINEAR
        )
        
        cv2.imshow("Cam 2", frameA)
        cv2.imshow("Cam 4", frameB)
        cv2.imshow("LIVE STITCHED (Fixed Size: {})".format(FINAL_FIXED_SIZE), final_image)

    else:
        # Display error state
        error_text = f"Stitching Failed! Status: {status}"
        error_frame = np.zeros(
            (FINAL_FIXED_SIZE[1], FINAL_FIXED_SIZE[0], 3), 
            dtype=np.uint8
        )
        cv2.putText(error_frame, error_text, (50, FINAL_FIXED_SIZE[1] // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("LIVE STITCHED (Fixed Size: {})".format(FINAL_FIXED_SIZE), error_frame)
        cv2.imshow("Cam 2", frameA)
        cv2.imshow("Cam 4", frameB)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap_A.release()
cap_B.release()
cv2.destroyAllWindows()