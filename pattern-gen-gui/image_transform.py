import moondream as md
from PIL import Image
import cv2
import numpy as np

# ===============================
# STEP 0: Initialize MoonDream API
# ===============================
model = md.vl(api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiJiYjM0MmJiYi1kMGNkLTQ3ZTgtYmM4Yy1hM2RkZTljODhiZWIiLCJvcmdfaWQiOiJZMHh5dUF2bmhtMFExMDZnemhvYzlSYjA0TFpkOTdTbiIsImlhdCI6MTc1MTc1MDExNSwidmVyIjoxfQ.uJ_XyUinKLJK6zcndJmuJdVqdjxwwxCtKv3wziqigkM")  # Replace with your actual API key

# ===============================
# STEP 1: Detect 4 red circles in the image
# ===============================
def detect_arena_corners(image_path, prompt="blue circles"):
    """
    Detects 4 red circular markers in the image using MoonDream.

    Args:
        image_path (str): Path to the input image.
        prompt (str): Detection prompt for MoonDream.

    Returns:
        corners (list): List of 4 corner (x, y) points.
        base_image (PIL.Image): PIL image object (for visualization or annotation).
    """
    base_image = Image.open(image_path).convert("RGB")
    width, height = base_image.size

    detections = model.detect(base_image, prompt)["objects"]

    # Extract center of each detected object
    corners = []
    for det in detections:
        x = (det["x_min"] + det["x_max"]) / 2 * width
        y = (det["y_min"] + det["y_max"]) / 2 * height
        corners.append([x, y])

    if len(corners) != 4:
        raise ValueError("Exactly 4 red circles must be detected for perspective transform.")

    return corners, base_image

# ===============================
# STEP 2: Sort the 4 points to consistent order
# ===============================
def sort_corners(corners):
    """
    Sorts the 4 corner points in order: top-left, top-right, bottom-right, bottom-left.

    Args:
        corners (list): List of 4 (x, y) points.

    Returns:
        sorted_pts (np.ndarray): Sorted array of 4 points.
    """
    pts = np.array(corners)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    return np.array([
        pts[np.argmin(s)],        # top-left
        pts[np.argmin(diff)],     # top-right
        pts[np.argmax(s)],        # bottom-right
        pts[np.argmax(diff)]      # bottom-left
    ])

# ===============================
# STEP 3: Warp the image to a square top-down view
# ===============================
def warp_image(image_path, src_pts):
    """
    Warps the input image using the 4 source points to get a top-down square view.

    Args:
        image_path (str): Path to the input image.
        src_pts (list): List of 4 sorted source (x, y) points.

    Returns:
        warped (np.ndarray): Warped image (top-down view).
        M (np.ndarray): Perspective transform matrix.
        side_len (int): Side length of the square warp.
        original (np.ndarray): Original OpenCV image.
    """
    img = cv2.imread(image_path)
    src_pts = sort_corners(src_pts)

    def dist(p1, p2):
        return np.linalg.norm(p1 - p2)

    # Compute max side length to enforce a square warp
    side_len = int(max(
        dist(src_pts[0], src_pts[1]),
        dist(src_pts[1], src_pts[2]),
        dist(src_pts[2], src_pts[3]),
        dist(src_pts[3], src_pts[0])
    ))

    dst_pts = np.float32([
        [0, 0],
        [side_len - 1, 0],
        [side_len - 1, side_len - 1],
        [0, side_len - 1]
    ])

    # Perspective transform
    M = cv2.getPerspectiveTransform(np.float32(src_pts), dst_pts)
    warped = cv2.warpPerspective(img, M, (side_len, side_len))

    return warped, M, side_len, img

# ===============================
# STEP 4: Unwarp a list of points from warped space to original image space
# ===============================
def unwarp_points(points, M_inv):
    """
    Transforms a list of points from warped space back to original image space.

    Args:
        points (list): List of (x, y) points in warped image.
        M_inv (np.ndarray): Inverse perspective matrix.

    Returns:
        list: List of (x, y) points in original image coordinates.
    """
    pts_np = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    unwarped_np = cv2.perspectiveTransform(pts_np, M_inv)
    return [tuple(map(int, pt[0])) for pt in unwarped_np]

# ===============================
# STEP 5: Unwarp an entire image using inverse transform
# ===============================
def unwarp_image(warped_image, M_inv, original_shape):
    """
    Unwarps the entire image back to original perspective.

    Args:
        warped_image (np.ndarray): Image in warped (top-down) space.
        M_inv (np.ndarray): Inverse perspective matrix.
        original_shape (tuple): (width, height) of the original image.

    Returns:
        np.ndarray: Unwarped image.
    """
    return cv2.warpPerspective(warped_image, M_inv, original_shape)
