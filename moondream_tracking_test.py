#!/usr/bin/env python3
import argparse
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np
from PIL import Image
import moondream as md

# -----------------------------
# Utility: convert normalized box -> pixel box
# -----------------------------
def denorm_box(obj, w, h):
    x_min = int(obj["x_min"] * w)
    y_min = int(obj["y_min"] * h)
    x_max = int(obj["x_max"] * w)
    y_max = int(obj["y_max"] * h)
    return (x_min, y_min, x_max - x_min, y_max - y_min)  # (x, y, w, h)

def iou_xywh(a, b):
    # a,b: (x, y, w, h)
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-6)

# -----------------------------
# Detector wrapper (MoonDream)
# -----------------------------
class MoonDreamDetector:
    def __init__(self, api_key: str, target_class: str):
        self.model = md.vl(api_key=api_key)
        self.target_class = target_class

    def detect_boxes(self, frame_bgr) -> List[Tuple[int,int,int,int]]:
        # Convert BGR->RGB for PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        res = self.model.detect(pil_img, self.target_class)
        objs = res.get("objects", []) or []
        H, W = frame_bgr.shape[:2]
        boxes = [denorm_box(o, W, H) for o in objs]
        return boxes

# -----------------------------
# Tracker wrapper (OpenCV CSRT/KCF/etc.)
# -----------------------------
def create_tracker(name="CSRT"):
    name = name.upper()
    if name == "CSRT":
        return cv2.TrackerCSRT_create()
    if name == "KCF":
        return cv2.TrackerKCF_create()
    if name == "MOSSE":
        return cv2.TrackerMOSSE_create()
    # Fallback
    return cv2.TrackerCSRT_create()

# -----------------------------
# Main loop
# -----------------------------
def run(
    source=0,
    api_key="YOUR_API_KEY",
    target_class="person",
    tracker_type="CSRT",
    redetect_every=30,     # force re-detect every N frames
    score_by="largest",    # "largest" or "closest" (to previous tracker box)
):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    det = MoonDreamDetector(api_key, target_class)

    tracker = None
    track_box = None  # (x,y,w,h)
    frame_count = 0
    last_detect_ts = 0.0

    clicked_point = None
    window_name = "Tracking (press q to quit)"

    def on_mouse(event, x, y, flags, param):
        nonlocal clicked_point
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        need_redetect = (
            tracker is None
            or track_box is None
            or frame_count % redetect_every == 0
        )

        # Try tracker first if initialized & not forced to redetect this frame
        tracked_ok = False
        if tracker is not None and track_box is not None and not need_redetect:
            tracked_ok, new_box = tracker.update(frame)
            if tracked_ok:
                track_box = tuple(map(int, new_box))
            else:
                # tracker lost -> force redetect now
                need_redetect = True

        # Redetect if needed
        if need_redetect:
            boxes = det.detect_boxes(frame)  # List[(x,y,w,h)]
            chosen = None

            if boxes:
                if clicked_point:
                    # pick detection containing the clicked point, else nearest center
                    cx, cy = clicked_point
                    candidates = []
                    for b in boxes:
                        x,y,w,h = b
                        inside = (x <= cx <= x+w) and (y <= cy <= y+h)
                        center = (x + w/2.0, y + h/2.0)
                        dist = (center[0]-cx)**2 + (center[1]-cy)**2
                        candidates.append((0 if inside else 1, dist, b))
                    candidates.sort(key=lambda t: (t[0], t[1]))
                    chosen = candidates[0][2]
                    # lock to that; no need to keep clicked_point after lock
                    clicked_point = None
                elif score_by == "largest":
                    # choose largest area
                    areas = [(b[2]*b[3], b) for b in boxes]
                    chosen = sorted(areas, key=lambda t: t[0], reverse=True)[0][1]
                elif score_by == "closest" and track_box is not None:
                    # choose max IoU with previous track_box
                    scored = [(iou_xywh(track_box, b), b) for b in boxes]
                    chosen = sorted(scored, key=lambda t: t[0], reverse=True)[0][1]
                else:
                    chosen = boxes[0]

            if chosen is not None:
                tracker = create_tracker(tracker_type)
                tracker.init(frame, tuple(chosen))
                track_box = chosen
                last_detect_ts = time.time()
                tracked_ok = True
            else:
                tracker = None
                track_box = None
                tracked_ok = False

        # Draw
        vis = frame.copy()
        if track_box is not None:
            x,y,w,h = map(int, track_box)
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(vis, f"{target_class}", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            cv2.putText(vis, f"Searching for {target_class}...", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(vis, "Click the object once to lock on", (20, vis.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1)

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MOONDREAM_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiJiYjM0MmJiYi1kMGNkLTQ3ZTgtYmM4Yy1hM2RkZTljODhiZWIiLCJvcmdfaWQiOiJZMHh5dUF2bmhtMFExMDZnemhvYzlSYjA0TFpkOTdTbiIsImlhdCI6MTc1MTc1MDExNSwidmVyIjoxfQ.uJ_XyUinKLJK6zcndJmuJdVqdjxwwxCtKv3wziqigkM"
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="Video source (0, 1, path/to/video.mp4, rtsp://...)")
    ap.add_argument("--api_key", default=MOONDREAM_API_KEY,
                    help="Your Moondream API key")
    ap.add_argument("--target", default="watch", help="Class to detect")
    ap.add_argument("--tracker", default="CSRT", choices=["CSRT","KCF","MOSSE"])
    ap.add_argument("--redetect_every", type=int, default=30)
    ap.add_argument("--score_by", default="largest", choices=["largest","closest"])
    args = ap.parse_args()

    # If numeric source string, treat as webcam index
    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    run(
        source=src,
        api_key=args.api_key,
        target_class=args.target,
        tracker_type=args.tracker,
        redetect_every=args.redetect_every,
        score_by=args.score_by,
    )

