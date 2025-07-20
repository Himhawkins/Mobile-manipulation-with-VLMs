#!/usr/bin/env python3
"""
Modular Robot Controller Library

Provides functions for arena-based pattern generation and plotting,
using plain-text CSV files instead of JSON for all I/O.
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # <-- no GUI, purely file-output
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

from arena_transform import ArenaTransformer
from pattern_dispatcher import dispatch

# Load environment variables for API keys
load_dotenv()

def overlay_points_on_image(
    img: np.ndarray,
    points: List[Dict[str, float]],
    scale: float = 20.0,
    dot_color: Tuple[int, int, int] = (255, 0, 255),
    arrow_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Draw each (x, y, theta) as a dot on an image.
    """
    out = img.copy()
    for p in points:
        x, y, theta = p['x'], p['y'], p['theta']
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(out, (ix, iy), 2, dot_color, -1)
    return out

def overlay_points_n_lines_on_image(
    img: np.ndarray,
    points: List[Dict[str, float]],
    scale: float = 20.0,
    dot_color: Tuple[int, int, int] = (255, 0, 255),
    arrow_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Draw each (x, y, theta) as a dot on an image, with lines connecting successive points.
    """
    out = img.copy()

    # Draw lines between consecutive points
    if len(points) > 1:
        for i in range(len(points) - 1):
            x1, y1 = points[i]['x'],     points[i]['y']
            x2, y2 = points[i+1]['x'],   points[i+1]['y']
            pt1 = (int(round(x1)), int(round(y1)))
            pt2 = (int(round(x2)), int(round(y2)))
            cv2.line(out, pt1, pt2, arrow_color, thickness=2)

    # Draw the dots on top
    for p in points:
        x, y, theta = p['x'], p['y'], p['theta']
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(out, (ix, iy), 3, dot_color, -1)

    return out


def generate_and_save_warped_plot(
    warped_txt_path: str,
    output_plot_path: str,
    warp_dims: Tuple[int, int],
):
    """
    Reads plain-text CSV of warped points (x,y,theta per line),
    then generates & saves a scatter+line plot with origin at top-left.
    """
    if not os.path.exists(warped_txt_path):
        print(f"Warped points file not found at {warped_txt_path}, skipping plot.")
        return

    # parse CSV
    pts: List[Tuple[float, float]] = []
    with open(warped_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            pts.append((x, y))

    if not pts:
        print(f"No valid points in {warped_txt_path}, skipping plot.")
        return

    x_coords = [p[0] for p in pts]
    y_coords = [p[1] for p in pts]
    img_width, img_height = warp_dims

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_coords, y_coords, s=50, zorder=5)
    ax.plot(x_coords, y_coords, linestyle='-', linewidth=2, zorder=1)

    ax.set_title('Warped Points Graph', color='white')
    ax.set_xlabel('X Coordinate', color='white')
    ax.set_ylabel('Y Coordinate', color='white')
    ax.set_facecolor('#333333')
    fig.patch.set_facecolor('black')
    ax.tick_params(colors='white')

    # <-- key change: origin top-left
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)

    ax.set_aspect('auto')
    ax.grid(True, linestyle=':', alpha=0.7)

    os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
    plt.savefig(
        output_plot_path,
        bbox_inches='tight', pad_inches=0.1,
        facecolor=fig.get_facecolor(),
        dpi=300
    )
    plt.close(fig)
    print(f"Warped points plot saved to {output_plot_path}")

def process_arena_pattern(
    image: np.ndarray,
    md_api_key: str,
    marker_prompt: str,
    gm_api_key: str,
    pattern_prompt: str,
    warp_dims: Tuple[int, int],
    raw_txt_path: str,
    unwarp_txt_path: str,
    warped_plot_path: str,
    target_plot_path: str
) -> None:
    """
    Processes an arena image to detect markers, warp it, generate a pattern,
    and save all outputs to plain-text CSV and image files.
    """
    original_frame = image.copy()

    # 1. Detect markers & warp
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transformer = ArenaTransformer(api_key=md_api_key, prompt=marker_prompt)
    corners = transformer.detect_markers(pil_img)
    M, out_size = transformer.compute_transform(corners, warp_dims)
    warped = cv2.warpPerspective(image, M, out_size)

    # 2. Generate pattern points
    genai.configure(api_key=gm_api_key)
    if 'model' in dispatch.__globals__:
        dispatch.__globals__['model'] = genai.GenerativeModel(
            dispatch.__globals__['model'].name
        )
    points = dispatch(pattern_prompt, warp_dims, gm_api_key)

    # 3. Save raw warped points as CSV
    os.makedirs(os.path.dirname(raw_txt_path), exist_ok=True)
    with open(raw_txt_path, 'w') as f:
        for x, y, theta in points:
            f.write(f"{int(round(x))},{int(round(y))},{theta}\n")
    print(f"Saved raw warped points to {raw_txt_path}")

    # 4. Generate & save the warped plot
    generate_and_save_warped_plot(raw_txt_path, warped_plot_path, warp_dims)

    # 5. Unwarp points back to original frame
    M_inv = transformer.invert_matrix(M)
    warped_xy = [(p[0], p[1]) for p in points]
    unwarped_xy = transformer.unwarp_points(warped_xy, M_inv)

    # prepare list of dicts for overlay
    unwarp_data: List[Dict[str, float]] = []
    for (xo, yo), (_, _, theta) in zip(unwarped_xy, points):
        unwarp_data.append({"x": xo, "y": yo, "theta": theta})

    # 6. Save unwarped points as CSV, with a header "(0/total_points)"
    os.makedirs(os.path.dirname(unwarp_txt_path), exist_ok=True)
    total = len(unwarp_data)
    with open(unwarp_txt_path, 'w') as f:
        # write header with the total number of points
        f.write(f"({total}/{total})\n")
        # then write each point
        for p in unwarp_data:
            x = int(round(p['x']))
            y = int(round(p['y']))
            theta = p['theta']
            f.write(f"{x},{y},{theta}\n")
    print(f"Saved unwarped points ({total}) to {unwarp_txt_path}")

    # 7. Overlay unwarped points on original and save
    try:
        overlay_img = overlay_points_n_lines_on_image(original_frame, unwarp_data)
        os.makedirs(os.path.dirname(target_plot_path), exist_ok=True)
        cv2.imwrite(target_plot_path, overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Target overlay saved to {target_plot_path}")
    except Exception as e:
        print(f"Error saving target overlay: {e}")

if __name__ == "__main__":
    # grab a single frame
    cap = cv2.VideoCapture(2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture frame from camera")

    process_arena_pattern(
        image=frame,
        md_api_key=os.getenv("MOONDREAM_API_KEY"),
        marker_prompt="blue circles",
        gm_api_key=os.getenv("GEMINI_API_KEY"),
        pattern_prompt="circle of radius 300",
        warp_dims=(800, 800),
        raw_txt_path="targets/pattern_llm/warped.txt",
        unwarp_txt_path="targets/pattern_llm/targets.txt",
        warped_plot_path="targets/pattern_llm/warped_plot.jpg",
        target_plot_path="targets/pattern_llm/targets_plot.jpg"
    )
