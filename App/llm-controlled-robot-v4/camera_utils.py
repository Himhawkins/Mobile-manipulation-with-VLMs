import cv2
import numpy as np
from PIL import Image
import math
import customtkinter as ctk

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
    Captures a frame from the given capture object,
    resizes it with letterboxing to fit the given dimensions,
    and returns a CTkImage suitable for CustomTkinter display.
    """
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    fh, fw = frame.shape[:2]

    scale = min(target_w / fw, target_h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    frame_resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0, y0 = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = frame_resized

    pil_img = Image.fromarray(canvas)
    return ctk.CTkImage(light_image=pil_img, size=(target_w, target_h))

def draw_robot_pose(frame, x, y, theta, corners=None, 
                    box_color=(255, 0, 0), box_thickness=2,
                    center_color=(0, 255, 0), center_radius=4,
                    line_color=(0, 0, 255), line_thickness=2, line_len=20):
    """
    Draws:
    - A polygon from 4 ArUco corners (if provided)
    - A center circle at (x, y)
    - A line indicating orientation from (x, y) in direction `theta` (radians)

    Parameters:
    - frame: OpenCV BGR image
    - x, y: center coordinates
    - theta: orientation in radians (0 = right)
    - corners: optional 4×2 list or np.array of ArUco marker corners
    - color: BGR color for drawing
    - thickness: polygon/line thickness
    - radius: circle radius at center
    - line_len: length of orientation line
    """
    if corners is not None:
        pts = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=box_color, thickness=box_thickness)

    # Draw center point
    cv2.circle(frame, (int(x), int(y)), center_radius, center_color, -1)

    # Draw heading line
    x2 = int(x + line_len * math.cos(theta))
    y2 = int(y + line_len * math.sin(theta))
    cv2.line(frame, (int(x), int(y)), (x2, y2), line_color, line_thickness)

    return frame
