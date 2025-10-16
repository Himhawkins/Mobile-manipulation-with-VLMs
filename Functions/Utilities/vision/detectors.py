# vision_dashboard/vision/detectors.py

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

class VLDetector:
    """Vision-Language Model (VLM) detector using the original Moondream v1 API."""
    def __init__(self, api_key=None):
        try:
            import moondream as md
        except ImportError:
            raise ImportError("moondream not installed. Run `pip install moondream`")
        
        # Use the original md.vl() function to load the model
        self.model = md.vl(api_key=api_key)

    def detect(self, image: np.ndarray, prompt: str) -> List[dict]:
        """Detects objects in an image based on a text prompt."""
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        w, h = pil_img.size
        
        # The original API for object detection
        results = self.model.detect(pil_img, prompt)
        objs = results.get("objects", [])

        obstacles = []
        for o in objs:
            # Coordinates are already normalized between 0 and 1
            x1, y1 = o["x_min"] * w, o["y_min"] * h
            x2, y2 = o["x_max"] * w, o["y_max"] * h
            bw, bh = x2 - x1, y2 - y1
            
            obstacles.append({
                "bbox": (int(x1), int(y1), int(bw), int(bh)),
                "centroid": (int(x1 + bw / 2), int(y1 + bh / 2))
            })
        return obstacles

class BGSubtractor:
    """Detects obstacles via background subtraction."""
    def __init__(self, diff_thresh=40, min_area=300, blur_ksize=(5,5)):
        self.diff_thresh = diff_thresh
        self.min_area = min_area
        self.blur_ksize = blur_ksize
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, self.blur_ksize, 0)

    def detect(self, frame: np.ndarray, ref_frame: np.ndarray) -> Tuple[List[Tuple], np.ndarray]:
        """Compares a frame to a reference image to find differences."""
        gray = self._preprocess(frame)
        ref_gray = self._preprocess(ref_frame)
        
        diff = cv2.absdiff(gray, ref_gray)
        _, mask = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > self.min_area]
        return boxes, mask

class ArucoDetector:
    """Detects ArUco markers for robot pose estimation."""
    def __init__(self, dictionary=cv2.aruco.DICT_ARUCO_ORIGINAL):
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    def detect_pose(self, frame: np.ndarray) -> dict:
        """Finds all ArUco markers and returns their poses."""
        corners, ids, _ = self.detector.detectMarkers(frame)
        poses = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                pts = corners[i][0]
                cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
                tl, tr = pts[0], pts[1]
                theta = np.arctan2(tr[1] - tl[1], tr[0] - tl[0])
                poses[int(mid)] = {"x": cx, "y": cy, "theta": float(theta), "corners": pts}
        return poses