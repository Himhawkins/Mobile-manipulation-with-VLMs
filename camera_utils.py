import cv2
import hashlib
import numpy as np
from PIL import Image
import math
import customtkinter as ctk

# ---------------------------
# Camera helpers
# ---------------------------
def start_camera(cam_index):
    """
    Initializes and returns an opened camera capture object.
    Returns None if the camera fails to open.
    """
    cap = cv2.VideoCapture(cam_index)
    return cap if cap.isOpened() else None

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def display_frame(frame, target_w, target_h):
    """
    Converts a BGR frame to a centered, letterboxed CTkImage of size (target_w, target_h).
    """
    frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fh, fw = frame_copy.shape[:2]

    scale = min(target_w / fw, target_h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    frame_copy_resized = cv2.resize(frame_copy, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0, y0 = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = frame_copy_resized

    pil_img = Image.fromarray(canvas)
    return ctk.CTkImage(light_image=pil_img, size=(target_w, target_h))

# ---------------------------
# Drawing helpers (pose)
# ---------------------------
def draw_robot_pose(frame, x=None, y=None, theta=None, corners=None, 
                    box_color=(0, 0, 255), box_thickness=2,
                    center_color=(0, 0, 255), center_radius=4,
                    line_color=(0, 0, 255), line_thickness=2, line_len=20,
                    robot_pos_path=None, draw_ids=True, font_scale=0.5, font_thickness=1):
    """
    Modes:
      1) Single pose: if robot_pos_path is None, draws one robot using (x,y,theta[,corners]).
      2) Multi-robot: if robot_pos_path is provided, reads lines in 'id,x,y,theta' format and draws all.

    All robots are drawn in the same red color (BGR = (0,0,255)).
    """
    frame_copy = frame.copy()

    # --- Single-pose mode (original behavior) ---
    if robot_pos_path is None:
        if corners is not None:
            pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame_copy, [pts], isClosed=True, color=box_color, thickness=box_thickness)

        if x is not None and y is not None:
            cv2.circle(frame_copy, (int(x), int(y)), center_radius, center_color, -1)

        if x is not None and y is not None and theta is not None:
            x2 = int(x + line_len * math.cos(theta))
            y2 = int(y + line_len * math.sin(theta))
            cv2.line(frame_copy, (int(x), int(y)), (x2, y2), line_color, line_thickness)

        return frame_copy

    # --- Multi-robot mode (draw from file) ---
    try:
        with open(robot_pos_path, "r") as f:
            lines = f.readlines()
    except Exception:
        return frame_copy

    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = [p for p in ln.replace(" ", ",").split(",") if p != ""]
        if len(parts) < 4:
            continue
        try:
            mid   = int(float(parts[0]))
            xi    = float(parts[1])
            yi    = float(parts[2])
            thetai= float(parts[3])
        except ValueError:
            continue

        # All same red color
        bgr = (0, 0, 255)

        # Center
        cv2.circle(frame_copy, (int(xi), int(yi)), center_radius, bgr, -1)

        # Heading line
        x2 = int(xi + line_len * math.cos(thetai))
        y2 = int(yi + line_len * math.sin(thetai))
        cv2.line(frame_copy, (int(xi), int(yi)), (x2, y2), bgr, line_thickness)

        # Optional label
        if draw_ids:
            label = f"ID:{mid}"
            cv2.putText(frame_copy, label, (int(xi) + 6, int(yi) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, bgr, font_thickness, lineType=cv2.LINE_AA)

    return frame_copy

# ---------------------------
# Sprite rotation & overlay (artifact-free)
# ---------------------------
def rotate_sprite_rgba_premult(rgba, angle_deg, scale=1.0):
    """
    Rotate an RGBA sprite using pre-multiplied alpha to avoid halos/box artifacts.
    Expands the canvas (rotate-bound) so nothing gets cropped.
    Returns a new RGBA (uint8).
    """
    h, w = rgba.shape[:2]

    # Ensure 4 channels
    if rgba.shape[2] == 3:
        a = np.full((h, w, 1), 255, dtype=np.uint8)
        rgba = np.concatenate([rgba, a], axis=2)

    # Rotation matrix (about center)
    cX, cY = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D((cX, cY), angle_deg, scale)

    # New bounds
    cos, sin = abs(M[0,0]), abs(M[0,1])
    nW = int(h * sin + w * cos)
    nH = int(h * cos + w * sin)
    M[0,2] += (nW / 2.0) - cX
    M[1,2] += (nH / 2.0) - cY

    # Split & premultiply (RGB * A)
    bgr = rgba[..., :3].astype(np.float32) / 255.0            # (h,w,3)
    a2d = (rgba[..., 3].astype(np.float32) / 255.0)           # (h,w)   <-- 2D on purpose
    bgr_pm = bgr * a2d[..., None]                             # (h,w,3)

    # Rotate PM-RGB (3ch) and A (1ch) with zero border
    bgr_pm_rot = cv2.warpAffine(
        bgr_pm, M, (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    a_rot = cv2.warpAffine(
        a2d, M, (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    # Ensure alpha has shape (nH, nW, 1) for broadcasting
    a_rot = a_rot[..., None]

    # Un-premultiply safely
    eps = 1e-6
    bgr_rot = bgr_pm_rot / np.maximum(a_rot, eps)

    out = np.dstack((bgr_rot, a_rot))
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def _overlay_rgba_on_bgr(frame_bgr, rgba, center_xy):
    """
    Alpha-blend an RGBA sprite centered at center_xy onto a BGR frame.
    Returns modified frame (in-place safe).
    """
    fh, fw = frame_bgr.shape[:2]
    sh, sw = rgba.shape[:2]
    cx, cy = int(center_xy[0]), int(center_xy[1])

    # Top-left placement (centered)
    x0 = cx - sw // 2
    y0 = cy - sh // 2
    x1 = x0 + sw
    y1 = y0 + sh

    # Clip to frame bounds
    x0_clip = max(0, x0); y0_clip = max(0, y0)
    x1_clip = min(fw, x1); y1_clip = min(fh, y1)
    if x0_clip >= x1_clip or y0_clip >= y1_clip:
        return frame_bgr  # entirely outside

    # Corresponding region in the sprite
    sx0 = x0_clip - x0
    sy0 = y0_clip - y0
    sx1 = sx0 + (x1_clip - x0_clip)
    sy1 = sy0 + (y1_clip - y0_clip)

    roi = frame_bgr[y0_clip:y1_clip, x0_clip:x1_clip]
    sprite = rgba[sy0:sy1, sx0:sx1, :]

    # Split channels
    bgr = sprite[..., :3].astype(np.float32)
    alpha = sprite[..., 3:4].astype(np.float32) / 255.0  # (h,w,1)

    # Alpha blend
    roi[:] = (alpha * bgr + (1.0 - alpha) * roi.astype(np.float32)).astype(np.uint8)
    return frame_bgr

# ---------------------------
# Main draw w/ sprite
# ---------------------------
def draw_robot_pose_with_sprite(
    frame,
    x, y, theta,
    corners=None,
    sprite=None,               # either a numpy RGBA (H,W,4) or a string path to PNG with alpha
    sprite_scale=1.0,          # scale factor
    box_color=(255, 0, 0), box_thickness=2,
    center_color=(0, 255, 0), center_radius=4,
    line_color=(0, 0, 255), line_thickness=2, line_len=20
):
    """
    Overlays a rotated robot sprite centered at (x,y) aligned to theta,
    then draws the pose graphics (corners, center, heading).

    - theta in radians; 0 points toward +x (right).
    - We rotate by (-deg(theta) - 90) so the sprite is 90° CW from default.
    """
    out = frame.copy()

    # --- Load / prepare sprite (RGBA) ---
    rgba = None
    if sprite is not None:
        if isinstance(sprite, str):
            img = cv2.imread(sprite, cv2.IMREAD_UNCHANGED)  # keep alpha
            if img is None:
                raise FileNotFoundError(f"Could not load sprite image: {sprite}")
            rgba = img
        else:
            rgba = sprite

    if rgba is not None:
        # Optional scaling BEFORE rotation
        if sprite_scale != 1.0:
            new_w = max(1, int(round(rgba.shape[1] * sprite_scale)))
            new_h = max(1, int(round(rgba.shape[0] * sprite_scale)))
            rgba = cv2.resize(
                rgba, (new_w, new_h),
                interpolation=cv2.INTER_AREA if sprite_scale < 1.0 else cv2.INTER_LINEAR
            )

        # Rotate to align with theta (90° clockwise correction)
        angle_deg = -math.degrees(theta) - 90
        rgba_rot = rotate_sprite_rgba_premult(rgba, angle_deg, scale=1.0)

        # Composite on frame (centered at x,y)
        _overlay_rgba_on_bgr(out, rgba_rot, (x, y))

    # --- Draw extras on top ---
    if corners is not None:
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=box_color, thickness=box_thickness)

    # Center
    cv2.circle(out, (int(x), int(y)), center_radius, center_color, -1)

    # Heading line
    x2 = int(x + line_len * math.cos(theta))
    y2 = int(y + line_len * math.sin(theta))
    cv2.line(out, (int(x), int(y)), (x2, y2), line_color, line_thickness)

    return out
