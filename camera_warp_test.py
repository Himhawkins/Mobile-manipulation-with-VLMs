#!/usr/bin/env python3
import cv2
import numpy as np

# If your project keeps arena settings helpers elsewhere, import it here:
# from arena_utils import load_arena_settings
# For a self-contained file, provide a minimal stub that reads Settings/arena_settings.json.
# Delete this stub if you already have load_arena_settings().
import json, os
def load_arena_settings(path="Settings/arena_settings.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}.")
    with open(path, "r") as f:
        return json.load(f)

# ------------------ CONFIG ------------------
CAM_INDEX = 6
DESIRED_W, DESIRED_H = 800, 600   # requested capture size
WARP_W, WARP_H = 500, 500         # output size for manual perspective warp (4-click)
WINDOW_LIVE = "Live (cropped)"
WINDOW_WARP = "Warped"
WINDOW_WARP_EXP = "Warped (settings)"  # window for settings-based warp

# Arena-settings warp selection
CELL_KEY = "0,2"  # which cell from arena_settings.json to use

# ------------------ HELPERS ------------------
def order_quad(pts):
    """
    Order 4 points as TL, TR, BR, BL regardless of click order.
    pts: (4,2) float32
    """
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)            # x + y
    diff = np.diff(pts, axis=1)    # x - y

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

class ClickCollector:
    """Mouse callback to collect 4 points on the live view."""
    def __init__(self):
        self.points = []
        self.enabled = False

    def reset(self):
        self.points = []

    def enable(self, en=True):
        self.enabled = en
        if en:
            self.reset()

    def __call__(self, event, x, y, flags, param):
        if not self.enabled:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()

def _rotate_points(pts, width, height, rotation_deg):
    """Rotate points by 0/90/180/270 CW around the image center, matching cv2.rotate."""
    pts = np.asarray(pts, dtype=np.float32)
    if rotation_deg % 360 == 0:
        return pts, width, height
    if rotation_deg % 360 == 90:
        # cv2.ROTATE_90_CLOCKWISE
        x, y = pts[:, 0], pts[:, 1]
        x2 = height - 1 - y
        y2 = x
        return np.stack([x2, y2], axis=1).astype(np.float32), height, width
    if rotation_deg % 360 == 180:
        x, y = pts[:, 0], pts[:, 1]
        x2 = width - 1 - x
        y2 = height - 1 - y
        return np.stack([x2, y2], axis=1).astype(np.float32), width, height
    if rotation_deg % 360 == 270:
        # cv2.ROTATE_90_COUNTERCLOCKWISE
        x, y = pts[:, 0], pts[:, 1]
        x2 = y
        y2 = width - 1 - x
        return np.stack([x2, y2], axis=1).astype(np.float32), height, width
    # Fallback (shouldn't happen)
    return pts, width, height

def warp_arena_frame(frame, cell_key="0,0"):
    arena_settings = load_arena_settings()
    cell = arena_settings.get("cells", {}).get(cell_key)
    if cell is None:
        raise ValueError(f"No cell configuration found for {cell_key}")

    # 1) Reference geometry in which corners were measured
    REF_WIDTH, REF_HEIGHT = 800, 600

    # 2) Output content size and optional border ("overlap")
    out_w = int(cell.get("width", 800))
    out_h = int(cell.get("height", 600))
    overlap = int(cell.get("overlap", 0))  # extra black border on all sides
    canvas_w = out_w + 2 * overlap
    canvas_h = out_h + 2 * overlap

    # 3) Read corners (assumed measured in REF_WIDTH x REF_HEIGHT BEFORE rotation)
    src_pts_ref = np.array([
        cell.get("topLeft", [0, 0]),
        cell.get("topRight", [0, 0]),
        cell.get("bottomRight", [0, 0]),
        cell.get("bottomLeft", [0, 0]),
    ], dtype=np.float32)

    # 4) Bring the current frame to the same reference scale first
    frame_ref = cv2.resize(frame, (REF_WIDTH, REF_HEIGHT), interpolation=cv2.INTER_LINEAR)
    # frame_ref = frame

    # 5) Apply rotation both to the image and to the points (to stay consistent)
    # rotation = int(cell.get("rotation", 0)) % 360
    # if rotation == 90:
    #     frame_ref = cv2.rotate(frame_ref, cv2.ROTATE_90_CLOCKWISE)
    # elif rotation == 180:
    #     frame_ref = cv2.rotate(frame_ref, cv2.ROTATE_180)
    # elif rotation == 270:
    #     frame_ref = cv2.rotate(frame_ref, cv2.ROTATE_90_COUNTERCLOCKWISE)

    dst = np.array([[0, 0],
                    [out_w-1, 0],
                    [out_w-1, out_h-1],
                    [0, out_h-1]], dtype=np.float32)

    # 8) Compute transform and warp DIRECTLY to the final canvas size (no second resize)
    M = cv2.getPerspectiveTransform(src_pts_ref, dst)
    warped = cv2.warpPerspective(
        frame_ref, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0  # black; use 30 if you want a gray edge like your first snippet
    )

    return warped

# ------------------ MAIN ------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera {CAM_INDEX}")
        return

    # Request a specific resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  DESIRED_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_H)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Requested: {DESIRED_W}x{DESIRED_H} | Actual: {actual_w}x{actual_h}")

    cv2.namedWindow(WINDOW_LIVE, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_WARP, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_WARP_EXP, cv2.WINDOW_NORMAL)

    clicker = ClickCollector()
    cv2.setMouseCallback(WINDOW_LIVE, clicker)

    roi = None       # (x, y, w, h)
    warped = None    # last manual warped image (4-point)

    print("[Controls]")
    print("  c : choose rectangular crop (drag + Enter/Space to confirm, 'c' to replace)")
    print("  p : pick 4 points for perspective warp (left-click to add, right-click to undo)")
    print("  r : reset selected 4 points")
    print("  e : warp via arena_settings.json for CELL_KEY")
    print("  s : save manual warped image to 'warped_output.png'")
    print(" ESC: quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed.")
            break

        view = frame

        # Apply crop if available
        if roi is not None:
            x, y, w, h = roi
            # Guard against out-of-bounds
            x2 = x + w
            y2 = y + h
            x = max(0, min(x, frame.shape[1]-1))
            y = max(0, min(y, frame.shape[0]-1))
            x2 = max(1, min(x2, frame.shape[1]))
            y2 = max(1, min(y2, frame.shape[0]))
            if x2 > x and y2 > y:
                view = frame[y:y2, x:x2].copy()
            else:
                view = frame

        # Draw clicked points on the live (cropped) view
        disp = view.copy()
        for i, (px, py) in enumerate(clicker.points):
            cv2.circle(disp, (int(px), int(py)), 5, (0, 255, 0), -1)
            cv2.putText(disp, f"{i+1}", (int(px)+6, int(py)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # If we have 4 points, compute and show warp (manual)
        if len(clicker.points) == 4:
            src = order_quad(np.array(clicker.points, dtype=np.float32))
            dst = np.array([[0, 0],
                            [WARP_W-1, 0],
                            [WARP_W-1, WARP_H-1],
                            [0, WARP_H-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(view, M, (WARP_W, WARP_H),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=30)

        cv2.imshow(WINDOW_LIVE, disp)
        if warped is not None:
            cv2.imshow(WINDOW_WARP, warped)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:     # ESC
            break
        elif key == ord('c'):
            # Pause on the current frame (cropped or not) to select ROI
            tmp = view.copy()
            r = cv2.selectROI("Select ROI (press Enter/Space to confirm)", tmp, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select ROI (press Enter/Space to confirm)")
            if r is not None and r[2] > 0 and r[3] > 0:
                # Convert local ROI (within current 'view') to global coordinates in the original frame
                if roi is None:
                    base_x, base_y = 0, 0
                else:
                    base_x, base_y, _, _ = roi
                rx, ry, rw, rh = r
                roi = (base_x + int(rx), base_y + int(ry), int(rw), int(rh))
                clicker.reset()
                warped = None
        elif key == ord('p'):
            # Enable 4-point clicking on the live (cropped) window
            clicker.enable(True)
            warped = None
        elif key == ord('r'):
            clicker.reset()
            clicker.enable(True)
            warped = None
        elif key == ord('e'):
            # Settings-based warp — use the full original frame (not the cropped view)
            try:
                expanded = warp_arena_frame(frame, cell_key=CELL_KEY)
                cv2.imshow(WINDOW_WARP_EXP, expanded)
                print(f"Settings-based warp shown for cell {CELL_KEY}")
            except Exception as ex:
                print(f"[warp_arena_frame_extended] {ex}")
        elif key == ord('s'):
            if warped is not None:
                cv2.imwrite("warped_output.png", warped)
                print("Saved: warped_output.png")
            else:
                print("No warped image to save yet (press 'p' and click 4 points).")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
