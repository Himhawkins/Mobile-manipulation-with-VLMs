#!/usr/bin/env python3
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional

# ---------------------------
# Tunables (same spirit as your script)
# ---------------------------
BLUR_KSIZE = (5, 5)
DIFF_THRESH = 40
MIN_AREA = 300
MORPH_KERNEL = (3, 3)

# ---------------------------
# Core Detector
# ---------------------------
class ObstacleDetector:
    """
    Maintains a reference (background) image and detects 'unknown' obstacles
    by differencing consecutive frames against the reference.

    Saving format (one obstacle per line):
      (x1,y1),(x2,y2),(x3,y3),(x4,y4)
    which are the rectangle corners (TL, TR, BR, BL) in pixel coords.
    """
    def __init__(self,
                 blur_ksize: Tuple[int, int] = BLUR_KSIZE,
                 diff_thresh: int = DIFF_THRESH,
                 min_area: int = MIN_AREA,
                 morph_kernel: Tuple[int, int] = MORPH_KERNEL):
        self.blur_ksize = blur_ksize
        self.diff_thresh = diff_thresh
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
        self._base: Optional[np.ndarray] = None  # float32 background

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_KSIZE, 0)
        return gray

    def update_reference(self, frame_bgr: np.ndarray) -> None:
        """Set/refresh the background reference from a BGR frame."""
        gray = self._preprocess(frame_bgr)
        self._base = gray.astype(np.float32)

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[Tuple[int,int,int,int]], np.ndarray, np.ndarray]:
        """
        Detect obstacles vs the stored reference.

        Returns:
            boxes: list of bounding boxes (x, y, w, h) for detected obstacles
            diff:  absolute difference image (uint8)
            mask:  binary mask after threshold + morphology (uint8 {0,255})
        """
        if self._base is None:
            raise RuntimeError("Reference image not set. Call update_reference() first.")

        gray = self._preprocess(frame_bgr)
        base_uint8 = cv2.convertScaleAbs(self._base)

        # Difference and threshold
        diff = cv2.absdiff(gray, base_uint8)
        _, mask = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)

        # Morphology to clean noise
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
        os.makedirs(os.path.dirname(path), exist_ok=True)

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

# ---------------------------
# Singleton-style instance + the two requested functions
# ---------------------------
_detector = ObstacleDetector()

def update_reference_image(frame_bgr: np.ndarray) -> None:
    """
    Function 2: Update reference image.
    Call this whenever you want to re-capture the baseline.
    """
    _detector.update_reference(frame_bgr)

def detect_realtime_obstacles(frame_bgr: np.ndarray, save_path: str = "Data/realtime_obstacles.txt"):
    """
    Function 1: Detect 'unknown' obstacles vs the stored reference and save them.

    Args:
        frame_bgr: current OpenCV frame (BGR).
        save_path: where to save the rectangles (default Data/realtime_obstacles.txt).

    Returns:
        boxes: list of (x, y, w, h)
        diff:  absdiff image (uint8)
        mask:  binary mask after processing (uint8 {0,255})
    """
    boxes, diff, mask = _detector.detect(frame_bgr)
    _detector.save_obstacles(boxes, save_path)
    return boxes, diff, mask

# ---------------------------
# Optional: demo loop (press 'b' to set reference, 'q' to exit)
# ---------------------------
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
                # draw boxes for visualization
                for (x, y, w, h) in boxes:
                    cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 0, 255), 2)

                cv2.imshow("diff", diff)
                cv2.imshow("mask", mask)
            except RuntimeError as e:
                cv2.putText(disp, str(e), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

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
# ---------------------------