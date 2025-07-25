import numpy as np
import cv2
import json
from Functions.Library.Agent.load_data import read_data  # update path if needed

def warp_points(points, data_folder="Data", settings_path="Settings/settings.json"):
    """
    Transforms points from original image space to warped (rectified) space.
    Accepts a list of [x, y] or (x, y).
    """
    data = read_data(data_folder)
    if data is None:
        raise RuntimeError("Could not read data from folder:", data_folder)

    arena_corners = data["arena_corners"]
    if len(arena_corners) != 4:
        raise ValueError("arena_corners must contain exactly 4 points")

    with open(settings_path, "r") as f:
        settings = json.load(f)
    w = int(settings.get("arena_width", 800))
    h = int(settings.get("arena_height", 800))

    def order_points(pts):
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)]
        ], dtype=np.float32)

    src_pts = order_points(arena_corners)
    dst_pts = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    pts_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts_np, M).reshape(-1, 2)
    return [list(map(int, pt)) for pt in warped]

def unwarp_points(points, data_folder="Data", settings_path="Settings/settings.json"):
    """
    Transforms points from warped (rectified) space to original image space.
    Accepts a list of [x, y] or (x, y).
    """
    data = read_data(data_folder)
    if data is None:
        raise RuntimeError("Could not read data from folder:", data_folder)

    arena_corners = data["arena_corners"]
    if len(arena_corners) != 4:
        raise ValueError("arena_corners must contain exactly 4 points")

    with open(settings_path, "r") as f:
        settings = json.load(f)
    w = int(settings.get("arena_width", 800))
    h = int(settings.get("arena_height", 800))

    def order_points(pts):
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1)
        return np.array([
            pts[np.argmin(s)],
            pts[np.argmin(d)],
            pts[np.argmax(s)],
            pts[np.argmax(d)]
        ], dtype=np.float32)

    src_pts = order_points(arena_corners)
    dst_pts = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(dst_pts, src_pts)
    pts_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    unwarped = cv2.perspectiveTransform(pts_np, M).reshape(-1, 2)
    return [list(map(int, pt)) for pt in unwarped]


if __name__ == "__main__":
    points_in_warped = [[150, 200], [300, 400]]
    points_in_image = unwarp_points(points_in_warped)
    print(points_in_image)