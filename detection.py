#!/usr/bin/env python3
import os
import moondream as md
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from typing import List, Tuple, Union, Optional

# If you use these elsewhere in your project, keep them imported:
# from Functions.Library.warping import get_warped_image, unwarp_points

# ============================================================
# SECTION A: MoonDream detector utilities (from your detection.py)
# ============================================================

class ObjectDetector:
    """
    VLM (MoonDream) object detector that returns a list of obstacles with bboxes.
    """
    def __init__(self, prompt: str = "black rectangles") -> None:
        if md is None:
            raise ImportError("moondream not installed. `pip install moondream`")
        load_dotenv()
        self.api_key = os.getenv("MOONDREAM_API_KEY")
        self.model  = md.vl(api_key=self.api_key)
        self.prompt = prompt

    def detect_objects(self, image):
        # Normalize input to PIL
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            pil_img = Image.open(str(image)).convert("RGB")

        w, h = pil_img.size
        results = self.model.detect(pil_img, self.prompt)
        objs = results.get("objects", [])

        obstacles = []
        for o in objs:
            xmin, xmax = o["x_min"], o["x_max"]
            ymin, ymax = o["y_min"], o["y_max"]
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            bw, bh = x2 - x1, y2 - y1
            area   = bw * bh
            cx, cy = x1 + bw // 2, y1 + bh // 2

            obstacles.append({
                "bbox":     (x1, y1, bw, bh),
                "centroid": (cx, cy),
                "area":     float(area)
            })

        return obstacles

def detect_and_list(image, prompt):
    detector = ObjectDetector(prompt=prompt)
    obstacles = detector.detect_objects(image)
    count = len(obstacles)
    return obstacles, count

def detect_and_get_centroids(frame, prompt="Blue Circles", save_path=None):
    objects, _ = detect_and_list(frame, prompt)
    centroids = [o["centroid"] for o in objects]
    if save_path is not None:
        with open(save_path, "w") as f:
            for x, y in centroids:
                f.write(f"{x},{y}\n")
    return centroids

def detect_and_get_bbox(img_path="Data/frame_img.png", prompt="Blue Circles", save_path=None):
    frame = cv2.imread(img_path)
    objects, _ = detect_and_list(frame, prompt)
    obstacles = [o["bbox"] for o in objects]
    if save_path is not None:
        with open(save_path, "w") as f:
            for x, y, w, h in obstacles:
                f.write(f"{x},{y},{w},{h}\n")
    return obstacles

def detect_obstacles(
    img_path: str = "Data/frame_img.png",
    prompt: str = "Blue Circles",
    save_path: str | None = None,
    sections: Union[int, Tuple[int, int]] = 1,
):
    """
    Detect obstacles with the VLM by dividing the image into sections (optional).
    Returns list of 4-corner rectangles in full-image coords.
    """
    frame = cv2.imread(img_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image at '{img_path}'")

    H, W = frame.shape[:2]

    # Normalize sections to (rows, cols)
    if isinstance(sections, int):
        if sections < 1:
            raise ValueError("sections must be >= 1")
        rows, cols = 1, sections
    else:
        if not (isinstance(sections, (tuple, list)) and len(sections) == 2):
            raise ValueError("sections must be an int or a (rows, cols) tuple")
        rows, cols = int(sections[0]), int(sections[1])
        if rows < 1 or cols < 1:
            raise ValueError("rows and cols in sections must be >= 1")

    merged_obstacles: List[List[Tuple[int, int]]] = []

    # Compute tile bounds and run detection per tile
    for r in range(rows):
        y0 = (H * r) // rows
        y1 = (H * (r + 1)) // rows if r < rows - 1 else H
        for c in range(cols):
            x0 = (W * c) // cols
            x1 = (W * (c + 1)) // cols if c < cols - 1 else W

            tile = frame[y0:y1, x0:x1]
            if tile.size == 0:
                continue

            objects, _ = detect_and_list(tile, prompt)

            # Offset local tile bboxes into global image coords and store as corners
            for obj in objects:
                x, y, w, h = obj["bbox"]
                gx, gy = x + x0, y + y0
                corners = [
                    (gx, gy),             # TL
                    (gx + w, gy),         # TR
                    (gx + w, gy + h),     # BR
                    (gx, gy + h)          # BL
                ]
                merged_obstacles.append(corners)

    if save_path is not None:
        with open(save_path, "w") as f:
            for corners in merged_obstacles:
                line = ",".join(f"({x},{y})" for (x, y) in corners)
                f.write(line + "\n")

    return merged_obstacles

def detect_arena(img_path="Data/frame_img.png", save_path=None):
    frame = cv2.imread(img_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image at path: {img_path}")
    height, width = frame.shape[:2]

    top_left     = (0, 0)
    top_right    = (width - 1, 0)
    bottom_left  = (0, height - 1)
    bottom_right = (width - 1, height - 1)
    ordered_corners = [top_left, bottom_left, bottom_right, top_right]

    if save_path:
        with open(save_path, "w") as f:
            for x, y in ordered_corners:
                f.write(f"{x},{y}\n")
    return ordered_corners

def detect_objects(img_path="Data/frame_img.png", prompt_list=["A","B","C"], save_path=None):
    frame = cv2.imread(img_path)
    centroids = []
    for prompt in prompt_list:
        pts = detect_and_get_centroids(frame=frame, prompt=prompt)
        centroids.extend(pts)
    if save_path is not None:
        with open(save_path, "w") as f:
            for x, y in centroids:
                f.write(f"{x},{y}\n")
    return centroids

def detect_robot_pose(frame, aruco_id, save_path=None):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None
    for i, mid in enumerate(ids.flatten()):
        if mid == aruco_id:
            pts = corners[i][0]
            cx = int(pts[:, 0].mean())
            cy = int(pts[:, 1].mean())
            top_left, top_right = pts[0], pts[1]
            theta = np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0]) + (np.pi / 2)
            if save_path is not None:
                with open(save_path, "w") as f:
                    f.write(f"{cx},{cy},{theta}\n")
            return (cx, cy, theta, pts)
    return None

def save_img_to_path(frame, save_path="Data/frame_img.png"):
    cv2.imwrite(save_path, frame)

# ============================================================
# SECTION B: Background-difference realtime obstacle utilities
# ============================================================

BLUR_KSIZE_BG = (5, 5)
DIFF_THRESH_BG = 40
MIN_AREA_BG = 300
MORPH_KERNEL_BG = (3, 3)

class ObstacleDetectorBG:
    """
    Maintains a reference (background) image and detects 'unknown obstacles'
    by differencing frames against the reference.
    """
    def __init__(self,
                 blur_ksize: Tuple[int, int] = BLUR_KSIZE_BG,
                 diff_thresh: int = DIFF_THRESH_BG,
                 min_area: int = MIN_AREA_BG,
                 morph_kernel: Tuple[int, int] = MORPH_KERNEL_BG):
        self.blur_ksize = blur_ksize
        self.diff_thresh = diff_thresh
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
        self._base: Optional[np.ndarray] = None  # float32 background

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_ksize, 0)
        return gray

    def update_reference(self, frame_bgr: np.ndarray) -> None:
        """Set/refresh the background reference from a BGR frame."""
        gray = self._preprocess(frame_bgr)
        self._base = gray.astype(np.float32)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[Tuple[int,int,int,int]], np.ndarray, np.ndarray]:
        """
        Detect obstacles vs the stored reference.

        Returns:
            boxes: list of (x, y, w, h)
            diff:  absdiff image (uint8)
            mask:  binary mask after processing (uint8 {0,255})
        """
        if self._base is None:
            raise RuntimeError("Reference image not set. Call update_reference_image() first.")

        gray = self._preprocess(frame_bgr)
        base_uint8 = cv2.convertScaleAbs(self._base)

        # Difference and threshold
        diff = cv2.absdiff(gray, base_uint8)
        _, mask = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)

        # Morphology
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        # Contours -> boxes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int,int,int,int]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))

        return boxes, diff, mask

    @staticmethod
    def _ensure_dir(path: str) -> None:
        if path:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)

    @staticmethod
    def _rect_to_corners(x: int, y: int, w: int, h: int) -> List[Tuple[int,int]]:
        # TL, TR, BR, BL
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    def save_obstacles(self, boxes: List[Tuple[int,int,int,int]], save_path: str) -> None:
        """
        Overwrites save_path with one rectangle per line:
        (x1,y1),(x2,y2),(x3,y3),(x4,y4)
        """
        self._ensure_dir(save_path)
        with open(save_path, "w") as f:
            for (x, y, w, h) in boxes:
                corners = self._rect_to_corners(x, y, w, h)
                line = ",".join(f"({cx},{cy})" for (cx, cy) in corners)
                f.write(line + "\n")

# Singleton background detector instance
_bg_detector = ObstacleDetectorBG()

def _read_robot_xy(robot_path: str = "Data/robot_pos.txt"):
    """
    Reads robot pose from file formatted as 'x,y,theta'. Returns (x, y) as ints.
    If file is missing or malformed, returns None.
    """
    try:
        with open(robot_path, "r") as f:
            line = f.readline().strip()
        if not line:
            return None
        parts = line.replace("(", "").replace(")", "").split(",")
        if len(parts) < 2:
            return None
        x = int(float(parts[0].strip()))
        y = int(float(parts[1].strip()))
        return (x, y)
    except Exception:
        return None

def update_reference_image(frame_bgr: np.ndarray, ref_path: str = "Data/frame_img.png") -> None:
    """Save the current frame as the new reference image."""
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)
    cv2.imwrite(ref_path, frame_bgr)

def _read_robot_positions(robot_path: str):
    """
    Parse robot_pos.txt with one robot per line:
        id,x,y,theta
    Returns a list of dicts: [{"id": int, "x": float, "y": float, "theta": float}, ...]
    Tolerates spaces instead of commas; ignores blank/comment lines.
    """
    robots = []
    try:
        with open(robot_path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p for p in line.replace(" ", ",").split(",") if p != ""]
                if len(parts) < 4:
                    continue
                try:
                    rid = int(float(parts[0]))   # tolerate "7.0"
                    x   = float(parts[1])
                    y   = float(parts[2])
                    th  = float(parts[3])
                except ValueError:
                    continue
                robots.append({"id": rid, "x": x, "y": y, "theta": th})
    except FileNotFoundError:
        pass
    return robots

def detect_realtime_obstacles(frame_bgr: np.ndarray,
                              save_path: str = "Data/realtime_obstacles.txt",
                              ref_path: str = "Data/frame_img.png",
                              robot_path: str = "Data/robot_pos.txt",
                              robot_padding: int = 0):
    """
    Compare current frame against reference image on disk and save obstacles.
    Skips any obstacle whose bbox contains ANY robot (x,y) loaded from robot_pos.txt.

    Args:
        frame_bgr: BGR frame from OpenCV.
        save_path: where to write TL,TR,BR,BL per line.
        ref_path:  background/reference image path.
        robot_path: file containing multiple lines of 'id,x,y,theta'.
        robot_padding: optional pixels to expand the bbox by before testing
                       containment (helps avoid excluding near-robot blobs).

    Returns:
        boxes: kept boxes [(x,y,w,h), ...]
        diff:  absdiff image (uint8)
        mask:  binary mask after processing (uint8 {0,255})
    """
    # load reference
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    if ref is None:
        raise RuntimeError(f"No reference image found at {ref_path}. Call update_reference_image first.")
    ref = cv2.GaussianBlur(ref, BLUR_KSIZE_BG, 0)

    # preprocess current
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLUR_KSIZE_BG, 0)

    # difference + threshold
    diff = cv2.absdiff(gray, ref)
    _, mask = cv2.threshold(diff, DIFF_THRESH_BG, 255, cv2.THRESH_BINARY)

    # morphology
    k = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_BG)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)

    # read ALL robot positions
    robots = _read_robot_positions(robot_path)  # list[{"id","x","y","theta"}]

    def _bbox_contains_any_robot(x, y, w, h):
        if not robots:
            return False
        x0 = x - robot_padding
        y0 = y - robot_padding
        x1 = x + w + robot_padding
        y1 = y + h + robot_padding
        for r in robots:
            rx, ry = r["x"], r["y"]
            if x0 <= rx <= x1 and y0 <= ry <= y1:
                return True
        return False

    # contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kept_boxes = []
    dirn = os.path.dirname(save_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)

    with open(save_path, "w") as f:
        for c in contours:
            if cv2.contourArea(c) < MIN_AREA_BG:
                continue
            x, y, w, h = cv2.boundingRect(c)

            # Skip boxes that contain ANY robot position
            if _bbox_contains_any_robot(x, y, w, h):
                continue

            kept_boxes.append((x, y, w, h))
            corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            line = ",".join(f"({cx},{cy})" for (cx, cy) in corners)
            f.write(line + "\n")

    return kept_boxes, diff, mask

# ============================================================
# Optional demo (press 'b' to set reference, 'q' to quit)
# ============================================================
if __name__ == "__main__":
    CAM_INDEX = 2
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Press 'b' to set/update reference. Press 'q' to quit.")
    have_ref = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        disp = frame.copy()

        if have_ref:
            try:
                boxes, diff, mask = detect_realtime_obstacles(frame, "Data/realtime_obstacles.txt")
                for (x, y, w, h) in boxes:
                    cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.imshow("diff", diff)
                cv2.imshow("mask", mask)
            except RuntimeError as e:
                cv2.putText(disp, str(e), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.putText(disp, "b: set reference | q: quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("frame", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('b'):
            update_reference_image(frame)
            have_ref = True

    cap.release()
    cv2.destroyAllWindows()
