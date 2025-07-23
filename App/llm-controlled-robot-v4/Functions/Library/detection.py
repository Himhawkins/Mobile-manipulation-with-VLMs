#!/usr/bin/env python3
import cv2
import moondream as md
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os

class ObjectDetector:
    def __init__(self, prompt="black rectangles") -> None:
        load_dotenv()
        self.api_key = os.getenv("MOONDREAM_API_KEY")
        self.model  = md.vl(api_key=self.api_key)
        self.prompt = prompt

    def detect_objects(self, image):
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

def detect_and_get_centroids(img_path="Data/frame_img.png", prompt="Blue Cricles", save_path=None):
    frame = cv2.imread(img_path)
    objects, _ = detect_and_list(frame, prompt)
    centroids = [o["centroid"] for o in objects]
    if save_path is not None:
            with open(save_path, "w") as f:
                for x, y in centroids:
                    f.write(f"{x},{y}\n")
    return centroids

def detect_and_get_bbox(img_path="Data/frame_img.png", prompt="Blue Cricles", save_path=None):
    frame = cv2.imread(img_path)
    objects, _ = detect_and_list(frame, prompt)
    obstacles = [o["bbox"] for o in objects]
    if save_path is not None:
            with open(save_path, "w") as f:
                for x, y, w, h in obstacles:
                    f.write(f"{x},{y},{w},{h}\n")
    return obstacles

def detect_arena(img_path="Data/frame_img.png", prompt="Blue Cricles", save_path=None):
    frame = cv2.imread(img_path)
    corners, count = detect_and_list(frame, prompt)
    centroids = [o["centroid"] for o in corners]
    if save_path is not None:
            with open(save_path, "w") as f:
                for x, y in centroids:
                    f.write(f"{x},{y}\n")
    if len(corners) != 4:
        raise ValueError(f"Expected 4 markers, but found {len(corners)}")
    return centroids

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

def main():
    IMG_PATH = "../../Data/live.jpg"
    frame = cv2.imread(IMG_PATH)
    #save_img_to_path(frame, save_path="Data/frame_img.png")
    detect_arena(IMG_PATH, "Blue Circles", save_path="../../Data/arena_corners.txt")
    #detect_and_get_bbox(IMG_PATH, "Obstacles Black Rectangles", save_path="Data/obstacles.txt")

if __name__ == "__main__":
    main()