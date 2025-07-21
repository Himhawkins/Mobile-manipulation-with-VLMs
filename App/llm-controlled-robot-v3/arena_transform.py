#!/usr/bin/env python3
"""
Arena Transformer Library

Provides classes and helper functions to detect circular markers in an image,
compute a perspective warp, and apply/unapply that warp.
"""
import logging
from pathlib import Path
from typing import List, Tuple, Union, Dict

import cv2
import moondream as md
import numpy as np
from PIL import Image
# import math
import pytesseract
from dotenv import load_dotenv
import os

class ArenaTransformer:
    """
    Handles detection of four arena markers and perspective warping.
    """

    def __init__(self, api_key: str, prompt: str) -> None:
        """
        :param api_key: MoonDream API key
        :param prompt: detection prompt for circular markers
        """
        self.model = md.vl(api_key=api_key)
        self.prompt = prompt

    def detect_markers(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> List[Tuple[float, float]]:
        """
        Detect exactly four circular markers in the image.

        :param image: file path, PIL.Image, or OpenCV BGR ndarray
        :returns: list of four (x, y) center points, in pixels
        :raises: ValueError if not exactly four are found
        """
        # support raw OpenCV frames
        if isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif not isinstance(image, Image.Image):
            img = Image.open(str(image)).convert("RGB")
        else:
            img = image.convert("RGB")

        w, h = img.size
        results = self.model.detect(img, self.prompt)
        objs = results.get("objects", [])
        centers: List[Tuple[float, float]] = []
        for o in objs:
            cx = ((o["x_min"] + o["x_max"]) / 2) * w
            cy = ((o["y_min"] + o["y_max"]) / 2) * h
            centers.append((cx, cy))

        if len(centers) != 4:
            raise ValueError(f"Expected 4 markers, but found {len(centers)}")

        logging.debug("Detected centers: %s", centers)
        return centers

    @staticmethod
    def sort_corners(
        corners: List[Tuple[float, float]]
    ) -> np.ndarray:
        """
        Sort points into top-left, top-right, bottom-right, bottom-left.

        :param corners: unsorted list of four (x, y) points
        :returns: 4×2 array of sorted float32 points
        """
        pts = np.array(corners, dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]

        sorted_pts = np.vstack([tl, tr, br, bl])
        logging.debug("Sorted corners: %s", sorted_pts.tolist())
        return sorted_pts

    def compute_transform(
        self,
        corners: List[Tuple[float, float]],
        dims: Tuple[float, float] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Compute a perspective transform matrix and output size.

        :param corners: list of four unsorted (x, y) points
        :param dims: optional (width, height) for output in pixels
        :returns: (3×3 matrix, (out_width, out_height))
        """
        src = self.sort_corners(corners)

        def _dist(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(a - b))

        if dims:
            out_w, out_h = map(int, map(round, dims))
            dst = np.array(
                [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                dtype=np.float32,
            )
        else:
            side = int(
                max(
                    _dist(src[0], src[1]),
                    _dist(src[1], src[2]),
                    _dist(src[2], src[3]),
                    _dist(src[3], src[0]),
                )
            )
            out_w = out_h = side
            dst = np.array(
                [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
                dtype=np.float32,
            )

        M = cv2.getPerspectiveTransform(src, dst)
        logging.info("Computed perspective matrix for output size %dx%d", out_w, out_h)
        return M, (out_w, out_h)

    def warp_image(
        self,
        image: Union[str, Path, np.ndarray],
        M: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Apply perspective warp to a top-down view.

        :param image: input image path or OpenCV BGR ndarray
        :param M: 3×3 perspective matrix
        :param size: (width, height) of output
        :returns: warped BGR image array
        """
        if isinstance(image, np.ndarray):
            orig = image
        else:
            orig = cv2.imread(str(image))
        w, h = size
        return cv2.warpPerspective(orig, M, (w, h))

    @staticmethod
    def invert_matrix(M: np.ndarray) -> np.ndarray:
        """Return the inverse of a 3×3 matrix."""
        return np.linalg.inv(M)

    @staticmethod
    def unwarp_points(
        points: List[Tuple[float, float]], M_inv: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Map points from warped space back to original image.
        """
        arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(arr, M_inv)
        return [tuple(map(int, pt[0])) for pt in out]

    @staticmethod
    def unwarp_image(
        warped: np.ndarray,
        M_inv: np.ndarray,
        original_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Warp an entire image back to the original perspective.
        """
        w, h = original_shape
        return cv2.warpPerspective(warped, M_inv, (w, h))

class ObstacleDetector:
    """
    Uses MoonDream VL to detect rectangular obstacles (e.g. "black rectangles")
    and returns their bounding‐boxes, centroids and areas.
    """

    def __init__(self, api_key: str, prompt: str = "black rectangles") -> None:
        self.model  = md.vl(api_key=api_key)
        self.prompt = prompt

    def detect_obstacles(
        self,
        image: Union[str, Path, Image.Image, np.ndarray]
    ) -> List[Dict[str, Union[Tuple[int,int,int,int], Tuple[int,int], float]]]:
        """
        :param image: file path, PIL.Image or OpenCV BGR ndarray
        :returns: list of dicts with "bbox", "centroid", "area"
        """
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

def detect_and_warp(
    image: Union[str, Path, np.ndarray],
    api_key: str,
    prompt: str,
    dims: Tuple[float, float] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray]:
    """
    Helper: detect markers, compute transform, warp image, and return integer arena corners.

    :returns: (warped_image, transform_matrix, output_size, arena_corners)
    """
    transformer = ArenaTransformer(api_key=api_key, prompt=prompt)

    # 1) detect raw float corners
    corners = transformer.detect_markers(image)

    # 2) compute perspective matrix & out size
    M, out_size = transformer.compute_transform(corners, dims)

    # 3) sort & quantize corners for masking later
    arena_corners = transformer.sort_corners(corners).astype(np.int32)

    # 4) warp the image
    warped = transformer.warp_image(image, M, out_size)

    return warped, M, out_size, arena_corners

def detect_and_list(
    image: Union[str, Path, Image.Image, np.ndarray],
    api_key: str,
    prompt: str = "black rectangles"
) -> List[Dict[str, Union[Tuple[int, int, int, int], Tuple[int, int], float]]]:
    """
    Helper: detect obstacles in the image and return their bounding boxes,
    centroids, and areas.

    :param image: file path, PIL.Image or OpenCV BGR ndarray
    :param api_key: your MoonDream API key
    :param prompt: object prompt (e.g. "black rectangles")
    :returns: list of dicts, each with keys "bbox", "centroid", and "area"
    """
    detector = ObstacleDetector(api_key=api_key, prompt=prompt)
    return detector.detect_obstacles(image)

# def detect_and_read_alphabets(frame, api_key):
#     """
#     Returns a list of (letter, centroid) for each detected letter.
#     """
#     prompt="Alphabets"
#     detections = detect_and_list(frame, api_key, prompt)
#     letter_info = []
#     for det in detections:
#         x, y, w, h = det["bbox"]
#         cx, cy     = det["centroid"]

#         # 2) crop the region (with a little padding)
#         pad = 4
#         x1 = max(0, x - pad)
#         y1 = max(0, y - pad)
#         x2 = min(frame.shape[1], x + w + pad)
#         y2 = min(frame.shape[0], y + h + pad)
#         crop = frame[y1:y2, x1:x2]

#         # 3) convert to grayscale & threshold (helps OCR)
#         gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 0, 255,
#                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#         # 4) run Tesseract, restricting to single characters
#         custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 10'
#         text = pytesseract.image_to_string(thresh,
#                                            config=custom_config).strip()

#         if text:
#             # grab first char if it detected more than one
#             letter = text[0].upper()
#             letter_info.append((letter, (cx, cy)))
#     return letter_info

def detect_and_read_alphabets(frame, api_key) -> List[Tuple[str, Tuple[int,int], Tuple[int,int,int,int]]]:
    """
    Returns a list of (letter, centroid, bbox) for each detected letter A–D,
    using detect_and_list rather than Tesseract OCR.
    Ensures each letter only appears once.
    """
    letters = ["A", "B", "C", "D"]
    letter_info: List[Tuple[str, Tuple[int,int], Tuple[int,int,int,int]]] = []
    seen = set()

    for letter in letters:
        dets = detect_and_list(frame, api_key, letter)
        for det in dets:
            if letter not in seen:
                cx, cy = det["centroid"]
                x, y, w, h = det["bbox"]
                letter_info.append((letter, (int(cx), int(cy)), (int(x), int(y), int(w), int(h))))
                seen.add(letter)
                break

    return letter_info


def detect_home(frame, api_key):
    """
    Returns a list of (letter, centroid) for each detected letter.
    """
    prompt="Blue Home Icon"
    detections = detect_and_list(frame, api_key, prompt)
    home = []
    for det in detections:
        cx, cy     = det["centroid"]
        x, y, w, h = det["bbox"]
        home.append(("Home", (int(cx), int(cy)), (int(x), int(y), int(w), int(h))))
    return home

def main():
    # 1) Get API key
    load_dotenv()
    api_key = os.getenv("MOONDREAM_API_KEY")
    if not api_key:
        api_key = input("Enter your MoonDream API key: ").strip()

    
    # 2) Open camera #2
    cam_idx = 2
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Error: could not open camera index {cam_idx}")
        return

    # 3) Stream and process frames
    ret, frame = cap.read()

    # 4) Detect letters A–D
    try:
        letters = detect_home(frame, api_key)
    except Exception as e:
        print(f"Detection error: {e}")
        letters = []

    # 5) Overlay results
    for letter, (x, y) in letters:
        cv2.putText(frame, letter, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # 6) Display
    cv2.imshow(f"Overlay Feed (cam {cam_idx})", frame)

    # 7) Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()