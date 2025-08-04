import cv2
import numpy as np
import json
from arena_utils import warp_arena_frame, load_arena_settings

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

def stitch_arena(settings, caps):
    rows = settings["rows"]
    cols = settings["cols"]
    full_rows = []

    for r in range(rows):
        row_images = []
        for c in range(cols):
            key = f"{r},{c}"
            cell = settings["cells"].get(key)

            if not cell:
                print(f"[WARN] No config for cell {key}. Using blank image.")
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
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

            row_images.append(warped)

        # Stitch the full row (even if blanks are included)
        if row_images:
            base_h = row_images[0].shape[0]
            row_images = [cv2.resize(img, (img.shape[1], base_h)) for img in row_images]
            stitched_row = np.hstack(row_images)
            full_rows.append(stitched_row)

    # Now stack all rows vertically
    if full_rows:
        base_w = full_rows[0].shape[1]
        full_rows = [cv2.resize(img, (base_w, img.shape[0])) for img in full_rows]
        stitched_full = np.vstack(full_rows)
        return stitched_full
    else:
        return None

# --- Main Loop ---
if __name__ == "__main__":
    settings = load_arena_settings()
    caps = open_all_cameras(settings)

    if not caps:
        print("No cameras available")
        exit()

    try:
        while True:
            stitched = stitch_arena(settings, caps)
            if stitched is not None:
                cv2.imshow("Live Stitched Arena View", stitched)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
