import math
from typing import List, Tuple


def read_grid_size(filepath: str = "coordinate.txt") -> Tuple[int, int]:
    """
    Reads grid size from a text file with lines:
      width=<int>
      height=<int>
    Defaults to "coordinate.txt" if no filepath is provided.
    Returns (width, height).
    """
    width = height = None
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip().lower()
            if line.startswith("width"):
                try:
                    width = int(line.split("=")[1])
                except ValueError:
                    raise ValueError(f"Invalid width value in '{filepath}'")
            elif line.startswith("height"):
                try:
                    height = int(line.split("=")[1])
                except ValueError:
                    raise ValueError(f"Invalid height value in '{filepath}'")
    if width is None or height is None:
        raise ValueError(f"Grid size not found in file '{filepath}'")
    return width, height


def generate_circle_pattern(
    radius: float,
    num_points: int = 50,
    center: Tuple[float, float] = None,
    grid_size: Tuple[int, int] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate points for a circle of given radius and number of points.
    Returns list of (x, y, theta) with theta as heading.
    If grid_size is not provided, uses read_grid_size().
    """
    if grid_size is None:
        width, height = read_grid_size()
    else:
        width, height = grid_size
    if center is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = center
    points: List[Tuple[float, float, float]] = []
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        heading = theta + math.pi / 2
        points.append((x, y, heading))
    return points


def generate_rectangle_pattern(
    width_rect: float,
    height_rect: float,
    num_points_per_side: int = 10,
    grid_size: Tuple[int, int] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate points along the perimeter of a rectangle.
    If grid_size is not provided, uses read_grid_size().
    """
    if grid_size is None:
        gw, gh = read_grid_size()
    else:
        gw, gh = grid_size
    cx, cy = gw / 2, gh / 2
    w2, h2 = width_rect / 2, height_rect / 2
    corners = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2), (-w2, -h2)]
    points: List[Tuple[float, float, float]] = []
    for i in range(len(corners) - 1):
        x0, y0 = corners[i]
        x1, y1 = corners[i + 1]
        for j in range(num_points_per_side + 1):
            t = j / num_points_per_side
            x_rel = x0 + (x1 - x0) * t
            y_rel = y0 + (y1 - y0) * t
            x = cx + x_rel
            y = cy + y_rel
            heading = math.atan2(y1 - y0, x1 - x0)
            points.append((x, y, heading))
    return points


def generate_trapezium_pattern(
    top_width: float,
    bottom_width: float,
    height: float,
    num_points_per_side: int = 10,
    grid_size: Tuple[int, int] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate points around a symmetric trapezium.
    If grid_size is not provided, uses read_grid_size().
    """
    if grid_size is None:
        gw, gh = read_grid_size()
    else:
        gw, gh = grid_size
    cx, cy = gw / 2, gh / 2
    tw2 = top_width / 2
    bw2 = bottom_width / 2
    h2 = height / 2
    corners = [(-bw2, -h2), (bw2, -h2), (tw2, h2), (-tw2, h2), (-bw2, -h2)]
    points: List[Tuple[float, float, float]] = []
    for i in range(len(corners) - 1):
        x0, y0 = corners[i]
        x1, y1 = corners[i + 1]
        for j in range(num_points_per_side + 1):
            t = j / num_points_per_side
            x_rel = x0 + (x1 - x0) * t
            y_rel = y0 + (y1 - y0) * t
            points.append((cx + x_rel, cy + y_rel, math.atan2(y1 - y0, x1 - x0)))
    return points


def generate_parallelogram_pattern(
    width_para: float,
    height_para: float,
    angle: float = math.pi / 6,
    num_points_per_side: int = 10,
    grid_size: Tuple[int, int] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate points around a parallelogram.
    If grid_size is not provided, uses read_grid_size().
    """
    if grid_size is None:
        gw, gh = read_grid_size()
    else:
        gw, gh = grid_size
    cx, cy = gw / 2, gh / 2
    offset = height_para * math.tan(angle)
    w2, h2 = width_para / 2, height_para / 2
    corners = [(-w2 - offset / 2, -h2), (w2 - offset / 2, -h2), (w2 + offset / 2, h2), (-w2 + offset / 2, h2), (-w2 - offset / 2, -h2)]
    points: List[Tuple[float, float, float]] = []
    for i in range(len(corners) - 1):
        x0, y0 = corners[i]
        x1, y1 = corners[i + 1]
        for j in range(num_points_per_side + 1):
            t = j / num_points_per_side
            x_rel = x0 + (x1 - x0) * t
            y_rel = y0 + (y1 - y0) * t
            points.append((cx + x_rel, cy + y_rel, math.atan2(y1 - y0, x1 - x0)))
    return points


def generate_diamond_pattern(
    width: float,
    height: float,
    num_points_per_side: int = 10,
    grid_size: Tuple[int, int] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate points around a diamond (rotated square).
    If grid_size is not provided, uses read_grid_size().
    """
    if grid_size is None:
        gw, gh = read_grid_size()
    else:
        gw, gh = grid_size
    cx, cy = gw / 2, gh / 2
    w2, h2 = width / 2, height / 2
    corners = [(0, h2), (w2, 0), (0, -h2), (-w2, 0), (0, h2)]
    points: List[Tuple[float, float, float]] = []
    for i in range(len(corners) - 1):
        x0, y0 = corners[i]
        x1, y1 = corners[i + 1]
        for j in range(num_points_per_side + 1):
            t = j / num_points_per_side
            x_rel = x0 + (x1 - x0) * t
            y_rel = y0 + (y1 - y0) * t
            points.append((cx + x_rel, cy + y_rel, math.atan2(y1 - y0, x1 - x0)))
    return points


def generate_lawnmower_pattern(
    width_area: float,
    height_area: float,
    spacing: float = 10.0,
    grid_size: Tuple[int, int] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate a back-and-forth lawnmower coverage pattern.
    If grid_size is not provided, uses read_grid_size().
    """
    if grid_size is None:
        gw, gh = read_grid_size()
    else:
        gw, gh = grid_size
    cx, cy = gw / 2, gh / 2
    w2, h2 = width_area / 2, height_area / 2
    y_lines: List[float] = []
    y = -h2
    while y <= h2:
        y_lines.append(y)
        y += spacing
    points: List[Tuple[float, float, float]] = []
    for idx, y_val in enumerate(y_lines):
        x_start = -w2 if idx % 2 == 0 else w2
        x_end = w2 if idx % 2 == 0 else -w2
        heading = 0 if idx % 2 == 0 else math.pi
        points.append((cx + x_start, cy + y_val, heading))
        points.append((cx + x_end, cy + y_val, heading))
    return points


def generate_zigzag_pattern(
    width_area: float,
    height_area: float,
    num_zigs: int = 10,
    grid_size: Tuple[int, int] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate a zigzag line across a bounding box.
    If grid_size is not provided, uses read_grid_size().
    """
    if grid_size is None:
        gw, gh = read_grid_size()
    else:
        gw, gh = grid_size
    cx, cy = gw / 2, gh / 2
    w2, h2 = width_area / 2, height_area / 2
    points: List[Tuple[float, float, float]] = []
    for i in range(num_zigs + 1):
        t = i / num_zigs
        x = -w2 + 2 * w2 * t
        y = h2 if i % 2 == 0 else -h2
        if i < num_zigs:
            t2 = (i + 1) / num_zigs
            x2 = -w2 + 2 * w2 * t2
            y2 = h2 if (i + 1) % 2 == 0 else -h2
            heading = math.atan2(y2 - y, x2 - x)
        else:
            heading = 0
        points.append((cx + x, cy + y, heading))
    return points


def generate_spiral_pattern(
    turns: float = 3.0,
    max_radius: float = 100.0,
    num_points: int = 200,
    grid_size: Tuple[int, int] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate an Archimedean spiral.
    If grid_size is not provided, uses read_grid_size().
    """
    if grid_size is None:
        gw, gh = read_grid_size()
    else:
        gw, gh = grid_size
    cx, cy = gw / 2, gh / 2
    points: List[Tuple[float, float, float]] = []
    for i in range(num_points):
        t = i / (num_points - 1)
        angle = turns * 2 * math.pi * t
        radius = max_radius * t
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        heading = angle + math.pi / 2
        points.append((x, y, heading))
    return points


def generate_ellipse_pattern(
    major_axis: float,
    minor_axis: float,
    num_points: int = 50,
    grid_size: Tuple[int, int] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate points for an ellipse.
    If grid_size is not provided, uses read_grid_size().
    """
    if grid_size is None:
        gw, gh = read_grid_size()
    else:
        gw, gh = grid_size
    cx, cy = gw / 2, gh / 2
    points: List[Tuple[float, float, float]] = []
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x = cx + major_axis * math.cos(theta)
        y = cy + minor_axis * math.sin(theta)
        dx = -major_axis * math.sin(theta)
        dy = minor_axis * math.cos(theta)
        heading = math.atan2(dy, dx)
        points.append((x, y, heading))
    return points
