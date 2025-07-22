import cv2
import numpy as np
import math
from collections import deque

def extract_line_waypoints(frame, spacing_px=20):
    """
    Given:
      - frame: BGR image containing a single continuous line
      - spacing_px: desired pixel spacing between waypoints
    Expects:
      - 'robot_pos.txt' with one line 'x,y,theta_robot'
    Returns:
      - List of (x, y, theta) triples in radians
    """
    # Read robot position (we only use x,y)
    rx, ry, _ = map(float, open("robot_pos.txt").read().strip().split(","))

    # Preprocess & skeletonize
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5,5), 0)
    _, bin_ = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    skel   = cv2.ximgproc.thinning(bin_)  # needs opencv-contrib

    # Build 8-connected graph of skeleton pixels
    pts     = np.transpose(np.nonzero(skel))
    pts_set = {tuple(p) for p in pts}
    G       = {p: [] for p in pts_set}
    for (r,c) in pts_set:
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                q = (r+dr, c+dc)
                if q in pts_set:
                    G[(r,c)].append(q)

    # Find endpoints
    endpoints = [p for p,nbrs in G.items() if len(nbrs)==1]
    if len(endpoints) != 2:
        return []

    # Pick start/goal based on robot pos
    start = min(endpoints, key=lambda p: math.hypot(p[1]-rx, p[0]-ry))
    goal  = endpoints[0] if endpoints[1]==start else endpoints[1]

    # BFS to get path
    queue, parent = deque([start]), {start: None}
    while queue:
        u = queue.popleft()
        if u == goal: break
        for v in G[u]:
            if v not in parent:
                parent[v] = u
                queue.append(v)
    if goal not in parent:
        return []
    path, cur = [], goal
    while cur:
        path.append(cur)
        cur = parent[cur]
    path.reverse()

    # Sample every spacing_px
    sampled = [path[0]]
    acc = 0.0
    for (r0,c0),(r1,c1) in zip(path, path[1:]):
        acc += math.hypot(r1-r0, c1-c0)
        if acc >= spacing_px:
            sampled.append((r1,c1))
            acc = 0.0
    if sampled[-1] != path[-1]:
        sampled.append(path[-1])

    # Compute headings
    waypoints = []
    for i,(r,c) in enumerate(sampled):
        if i < len(sampled)-1:
            dr, dc = sampled[i+1][0]-r, sampled[i+1][1]-c
        else:
            dr, dc = r-sampled[i-1][0], c-sampled[i-1][1]
        theta = math.atan2(dr, dc)
        waypoints.append((c, r, theta))  # (x=c, y=r)

    return waypoints
if __name__ == "__main__":
    # --- Main ---
    # 1) Read one frame (from image or camera)
    # If you prefer live camera:

    # cap = cv2.VideoCapture(0)
    # ret, frame = cap.read()
    # cap.release()
    # if not ret: raise RuntimeError("Camera failed")

    # Or load from disk:
    frame = cv2.imread("Arena3.png")
    if frame is None:
        raise RuntimeError("Failed to load image")

    # 2) Extract waypoints
    wps = extract_line_waypoints(frame, spacing_px=30)

    # 3) Save to file
    with open('line_following.txt', 'w') as f:
        for x, y, theta in wps:
            f.write(f"{x}, {y}, {theta:.5f}\n")

    # 4) Draw & preview
    for x, y, theta in wps:
        cv2.circle(frame, (x, y), 5, (0,255,0), -1)
        x2 = int(x + 20*math.cos(theta))
        y2 = int(y + 20*math.sin(theta))
        cv2.line(frame, (x, y), (x2, y2), (0,255,0), 2)

    cv2.imshow("Waypoints Preview", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
