import math
import json
from typing import List, Tuple
from Functions.Library.warping import unwarp_points
from Functions.Library.planning import trace_targets

def get_arena_dimensions(settings_path="Settings/settings.json"):
    DEFAULT_WIDTH = 1200
    DEFAULT_HEIGHT = 900
    try:
        with open(settings_path, "r") as f:
            settings = json.load(f)
    except FileNotFoundError:
        print("Warning: settings.json not found. Using default arena size.")
        settings = {}
    except json.JSONDecodeError as e:
        print(f"Warning: could not parse settings.json ({e}). Using default arena size.")
        settings = {}
    arena_width = int(settings.get("arena_width", str(DEFAULT_WIDTH)))
    arena_height = int(settings.get("arena_height", str(DEFAULT_HEIGHT)))

    return (arena_width, arena_height)

def generate_circle_pattern(
    radius: float,
    num_points: float = 20,
    center: Tuple[float, float] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate points for a circle of given radius and number of points.
    Returns list of (x, y, theta) with theta as heading.
    If grid_size is not provided, uses read_grid_size().
    """
    width, height = get_arena_dimensions()
    if center is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = center
    points: List[Tuple[float, float, float]] = []
    num_points = int(num_points)
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        points.append((x, y))
    uw_points = unwarp_points(points)
    return trace_targets(input_target_list=uw_points, output_target_path="Targets/path.txt")
    # return points


def generate_rectangle_pattern(
    width_rect: float,
    height_rect: float,
    num_points: float = 10,
    center: Tuple[float, float] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate points along the perimeter of a rectangle.
    If grid_size is not provided, uses read_grid_size().
    """
    width, height = get_arena_dimensions()
    if center is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = center
    w2, h2 = width_rect / 2, height_rect / 2
    corners = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2), (-w2, -h2)]
    points: List[Tuple[float, float, float]] = []
    for i in range(len(corners) - 1):
        x0, y0 = corners[i]
        x1, y1 = corners[i + 1]
        num_points = int(num_points)
        for j in range(num_points + 1):
            t = j / num_points
            x_rel = x0 + (x1 - x0) * t
            y_rel = y0 + (y1 - y0) * t
            x = cx + x_rel
            y = cy + y_rel
            points.append((x, y))
    uw_points = unwarp_points(points)
    return trace_targets(input_target_list=uw_points, output_target_path="Targets/path.txt")
    # return points


def generate_trapezium_pattern(
    top_width: float,
    bottom_width: float,
    height: float,
    num_points: float = 10,
    center: Tuple[float, float] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate points around a symmetric trapezium.
    If grid_size is not provided, uses read_grid_size().
    """
    width, height = get_arena_dimensions()
    if center is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = center
    tw2 = top_width / 2
    bw2 = bottom_width / 2
    h2 = height / 2
    corners = [(-bw2, -h2), (bw2, -h2), (tw2, h2), (-tw2, h2), (-bw2, -h2)]
    points: List[Tuple[float, float, float]] = []
    for i in range(len(corners) - 1):
        x0, y0 = corners[i]
        x1, y1 = corners[i + 1]
        num_points = int(num_points)
        for j in range(num_points + 1):
            t = j / num_points
            x_rel = x0 + (x1 - x0) * t
            y_rel = y0 + (y1 - y0) * t
            points.append((cx + x_rel, cy + y_rel))
    uw_points = unwarp_points(points)
    return trace_targets(input_target_list=uw_points, output_target_path="Targets/path.txt")
    # return points


def generate_parallelogram_pattern(
    width_para: float,
    height_para: float,
    angle: float = math.pi / 6,
    num_points: float = 10,
    center: Tuple[float, float] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate points around a parallelogram.
    If grid_size is not provided, uses read_grid_size().
    """
    width, height = get_arena_dimensions()
    if center is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = center
    offset = height_para * math.tan(angle)
    w2, h2 = width_para / 2, height_para / 2
    corners = [(-w2 - offset / 2, -h2), (w2 - offset / 2, -h2), (w2 + offset / 2, h2), (-w2 + offset / 2, h2), (-w2 - offset / 2, -h2)]
    points: List[Tuple[float, float, float]] = []
    for i in range(len(corners) - 1):
        x0, y0 = corners[i]
        x1, y1 = corners[i + 1]
        num_points = int(num_points)
        for j in range(num_points + 1):
            t = j / num_points
            x_rel = x0 + (x1 - x0) * t
            y_rel = y0 + (y1 - y0) * t
            points.append((cx + x_rel, cy + y_rel))
    uw_points = unwarp_points(points)
    return trace_targets(input_target_list=uw_points, output_target_path="Targets/path.txt")
    # return points


def generate_diamond_pattern(
    width: float,
    height: float,
    num_points: float = 10,
    center: Tuple[float, float] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate points around a diamond (rotated square).
    If grid_size is not provided, uses read_grid_size().
    """
    width, height = get_arena_dimensions()
    if center is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = center
    w2, h2 = width / 2, height / 2
    corners = [(0, h2), (w2, 0), (0, -h2), (-w2, 0), (0, h2)]
    points: List[Tuple[float, float, float]] = []
    for i in range(len(corners) - 1):
        x0, y0 = corners[i]
        x1, y1 = corners[i + 1]
        num_points = int(num_points)
        for j in range(num_points + 1):
            t = j / num_points
            x_rel = x0 + (x1 - x0) * t
            y_rel = y0 + (y1 - y0) * t
            points.append((cx + x_rel, cy + y_rel))
    uw_points = unwarp_points(points)
    return trace_targets(input_target_list=uw_points, output_target_path="Targets/path.txt")
    # return points


def generate_lawnmower_pattern(
    width_area: float,
    height_area: float,
    spacing: float = 10.0,
    center: Tuple[float, float] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate a back-and-forth lawnmower coverage pattern.
    If grid_size is not provided, uses read_grid_size().
    """
    width, height = get_arena_dimensions()
    if center is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = center
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
        points.append((cx + x_start, cy + y_val))
        points.append((cx + x_end, cy + y_val))
    uw_points = unwarp_points(points)
    return trace_targets(input_target_list=uw_points, output_target_path="Targets/path.txt")
    # return points


def generate_zigzag_pattern(
    width_area: float,
    height_area: float,
    num_zigs: float = 10,
    center: Tuple[float, float] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate a zigzag line across a bounding box.
    If grid_size is not provided, uses read_grid_size().
    """
    width, height = get_arena_dimensions()
    if center is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = center
    w2, h2 = width_area / 2, height_area / 2
    points: List[Tuple[float, float, float]] = []
    num_zigs = int(num_zigs)
    for i in range(num_zigs + 1):
        t = i / num_zigs
        x = -w2 + 2 * w2 * t
        y = h2 if i % 2 == 0 else -h2
        points.append((cx + x, cy + y))
    uw_points = unwarp_points(points)
    return trace_targets(input_target_list=uw_points, output_target_path="Targets/path.txt")
    # return points


def generate_spiral_pattern(
    turns: float = 3.0,
    max_radius: float = 100.0,
    num_points: float = 50,
    center: Tuple[float, float] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate an Archimedean spiral.
    If grid_size is not provided, uses read_grid_size().
    """
    width, height = get_arena_dimensions()
    if center is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = center
    points: List[Tuple[float, float, float]] = []
    num_points = int(num_points)
    for i in range(num_points):
        t = i / (num_points - 1)
        angle = turns * 2 * math.pi * t
        radius = max_radius * t
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    uw_points = unwarp_points(points)
    return trace_targets(input_target_list=uw_points, output_target_path="Targets/path.txt")
    # return points


def generate_ellipse_pattern(
    major_axis: float,
    minor_axis: float,
    num_points: float = 50,
    center: Tuple[float, float] = None
) -> List[Tuple[float, float, float]]:
    """
    Generate points for an ellipse.
    If grid_size is not provided, uses read_grid_size().
    """
    width, height = get_arena_dimensions()
    if center is None:
        cx, cy = width / 2, height / 2
    else:
        cx, cy = center
    points: List[Tuple[float, float, float]] = []
    num_points = int(num_points)
    for i in range(num_points):
        theta = 2 * math.pi * i / num_points
        x = cx + major_axis * math.cos(theta)
        y = cy + minor_axis * math.sin(theta)
        dx = -major_axis * math.sin(theta)
        dy = minor_axis * math.cos(theta)
        points.append((x, y))
    uw_points = unwarp_points(points)
    return trace_targets(input_target_list=uw_points, output_target_path="Targets/path.txt")
    # return points


# c={'center': [321.0, 211.0], 'num_points': 60.0, 'radius': 2.0}
# print(generate_circle_pattern(**c))