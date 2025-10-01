import cv2
import os
import json
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
def _load_robot_names(json_path: str) -> dict[int, str]:
    try:
        if not os.path.exists(json_path):
            return {}
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        robots = data.get("robots", [])
        out = {}
        for r in robots:
            try:
                rid = int(r.get("id", -1))
                if rid > 0:  # ignore id=0
                    out[rid] = str(r.get("name", "")).strip()
            except Exception:
                continue
        return out
    except Exception:
        return {}

def _draw_label_box(img, text, anchor_xy, *,
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=0.5, font_thickness=1,
                    pad_x=4, pad_y=2,
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    offset=(6, -6)):
    """
    Draw a filled rectangle with text on top.
      - anchor_xy: (x,y) point near which the label should appear
      - offset: (dx, dy) from anchor to place the text box
    """
    x, y = int(anchor_xy[0]), int(anchor_xy[1])
    dx, dy = offset

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    box_w = tw + 2 * pad_x
    box_h = th + 2 * pad_y

    # Top-left of the rectangle
    tl_x = x + dx
    tl_y = y + dy - box_h  # place above anchor by default

    # Clamp box inside image bounds
    H, W = img.shape[:2]
    tl_x = max(0, min(tl_x, W - box_w))
    tl_y = max(0, min(tl_y, H - box_h))

    br_x = tl_x + box_w
    br_y = tl_y + box_h

    # Draw filled rectangle
    cv2.rectangle(img, (tl_x, tl_y), (br_x, br_y), bg_color, thickness=-1)

    # Put text (baseline aligned)
    text_x = tl_x + pad_x
    text_y = tl_y + box_h - pad_y - baseline
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

def draw_robot_pose(frame, x=None, y=None, theta=None, corners=None, 
                    box_color=(0, 0, 255), box_thickness=2,
                    center_color=(0, 0, 255), center_radius=4,
                    line_color=(0, 0, 255), line_thickness=2, line_len=20,
                    robot_pos_path="Data/robot_pos.txt", draw_ids=True,
                    font_scale=0.5, font_thickness=1,
                    robot_names_path="Data/robot_names.json"):
    """
    Modes:
      1) Single pose: if robot_pos_path is None, draws one robot using (x,y,theta[,corners]).
      2) Multi-robot: if robot_pos_path is provided, reads lines in 'id,x,y,theta' format and draws all.

    Labels:
      Uses Data/robot_names.json to show "Name(ID)" e.g., "Alpha(1)".
      Falls back to just "(ID)" if name is missing.
      Renders label on a black rectangle with white text.
    """
    frame_copy = frame.copy()

    # --- Single-pose mode ---
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

    # --- Multi-robot mode ---
    try:
        with open(robot_pos_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return frame_copy

    # Load names once per call
    id_to_name = _load_robot_names(robot_names_path)

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

        # Draw center & heading (red)
        bgr = (0, 0, 255)
        cv2.circle(frame_copy, (int(xi), int(yi)), center_radius, bgr, -1)
        x2 = int(xi + line_len * math.cos(thetai))
        y2 = int(yi + line_len * math.sin(thetai))
        cv2.line(frame_copy, (int(xi), int(yi)), (x2, y2), bgr, line_thickness)

        # Label box
        if draw_ids:
            name = id_to_name.get(mid, "").strip()
            label = f"{name}({mid})" if name else f"({mid})"
            _draw_label_box(
                frame_copy,
                label,
                anchor_xy=(xi, yi),
                font_scale=font_scale,
                font_thickness=font_thickness,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                pad_x=5,
                pad_y=3,
                offset=(-30, -30)  # slightly to the right and above the center
            )

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
