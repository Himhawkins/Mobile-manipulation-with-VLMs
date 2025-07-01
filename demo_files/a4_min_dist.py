import heapq
import math
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Configuration Constants ---
GRID_WIDTH = 640
GRID_HEIGHT = 480
# NOTE: A lower resolution increases computation time significantly. 
# 5-10 is a good balance of accuracy and performance. 
RESOLUTION = 5
SAFETY_DISTANCE = 15# The minimum distance in pixels to keep from obstacles.
MAX_PATH_SEGMENT_LENGTH = 2# The maximum distance between any two points in the final path.
NEIGHBOR_COST = RESOLUTION
NEIGHBOR_COST_DIAG = math.sqrt(2) * RESOLUTION # Use float for better accuracy
pause_time=0.001
# File names
OBSTACLES_FILE = 'obstacles.txt'
ROBOT_POS_FILE = 'robot_pos.txt'
TARGET_FILE = 'target.txt'
STAR_TARGET_FILE = 'star_target.txt'
ERROR_FILE = 'error.txt'

class Node:
    """A node class for A* Pathfinding"""
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position  # Tuple (x, y) in grid coordinates

        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost from current node to end
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        # This is important for the priority queue (heapq)
        return self.f < other.f

    def __hash__(self):
        # Required for adding nodes to sets
        return hash(self.position)

def read_obstacles(filename, grid, safety_distance):
    """
    Reads obstacle data, inflates them by a safety distance, and marks them on the grid.
    Returns the grid and a list of the original obstacle dimensions for visualization.
    """
    original_obstacles = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                parts = [int(p.strip()) for p in line.split(',')]
                xmin, ymin, xmax, ymax = parts
                original_obstacles.append((xmin, ymin, xmax - xmin, ymax - ymin))

                # Inflate the obstacle by the safety distance
                inflated_xmin = xmin - safety_distance
                inflated_ymin = ymin - safety_distance
                inflated_xmax = xmax + safety_distance
                inflated_ymax = ymax + safety_distance
                
                # Convert inflated pixel coordinates to grid coordinates
                grid_xmin = inflated_xmin // RESOLUTION
                grid_ymin = inflated_ymin // RESOLUTION
                grid_xmax = inflated_xmax // RESOLUTION
                grid_ymax = inflated_ymax // RESOLUTION

                for x in range(grid_xmin, grid_xmax + 1):
                    for y in range(grid_ymin, grid_ymax + 1):
                        if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                            grid[x][y] = 1 # Mark as obstacle
    except FileNotFoundError:
        print(f"Warning: Obstacle file '{filename}' not found. Continuing without obstacles.")
    except Exception as e:
        print(f"Error reading obstacles: {e}")
    return grid, original_obstacles

def read_pos(filename):
    """Reads a position (x,y) or (x,y,theta) from a file."""
    try:
        with open(filename, 'r') as f:
            line = f.readline()
            parts = [float(p.strip()) for p in line.split(',')]
            return (int(parts[0]), int(parts[1]))
    except FileNotFoundError:
        print(f"Error: Position file '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error reading position from {filename}: {e}")
        return None

def astar(grid, start_pos, end_pos):
    """
    Returns a list of tuples as a path from the given start to the given end in grid coordinates.
    This version is optimized to handle large grids more efficiently.
    """
    start_node_pos = (start_pos[0] // RESOLUTION, start_pos[1] // RESOLUTION)
    end_node_pos = (end_pos[0] // RESOLUTION, end_pos[1] // RESOLUTION)

    start_node = Node(None, start_node_pos)
    end_node = Node(None, end_node_pos)

    open_list = []
    closed_set = set()

    heapq.heappush(open_list, start_node)

    while len(open_list) > 0:
        current_node = heapq.heappop(open_list)

        if current_node.position in closed_set:
            continue
        closed_set.add(current_node.position)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                pixel_x = current.position[0] * RESOLUTION + RESOLUTION // 2
                pixel_y = current.position[1] * RESOLUTION + RESOLUTION // 2
                path.append((pixel_x, pixel_y))
                current = current.parent
            return path[::-1]

        # Generate children (8 directions)
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if not (0 <= node_position[0] < len(grid) and 0 <= node_position[1] < len(grid[0])):
                continue
            if grid[node_position[0]][node_position[1]] != 0:
                continue
            if node_position in closed_set:
                continue

            is_diagonal = new_position[0] != 0 and new_position[1] != 0
            cost = NEIGHBOR_COST_DIAG if is_diagonal else NEIGHBOR_COST
            
            new_node = Node(current_node, node_position)
            new_node.g = current_node.g + cost
            # Heuristic: Euclidean distance
            dx = abs(new_node.position[0] - end_node.position[0])
            dy = abs(new_node.position[1] - end_node.position[1])
            new_node.h = math.sqrt(dx**2 + dy**2) * RESOLUTION
            new_node.f = new_node.g + new_node.h
            
            heapq.heappush(open_list, new_node)

    return None

def line_of_sight(grid, p1, p2):
    """
    Checks if there is a clear straight line between two points.
    Uses Bresenham's line algorithm on the grid.
    p1 and p2 are in pixel coordinates.
    """
    x0, y0 = p1[0] // RESOLUTION, p1[1] // RESOLUTION
    x1, y1 = p2[0] // RESOLUTION, p2[1] // RESOLUTION

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if not (0 <= x0 < len(grid) and 0 <= y0 < len(grid[0])) or grid[x0][y0] == 1:
            return False
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return True

def simplify_path(grid, path):
    """
    Simplifies a path by removing unnecessary intermediate points using line-of-sight checks.
    """
    if not path or len(path) < 3:
        return path

    simplified_path = [path[0]]
    current_index = 0

    while current_index < len(path) - 1:
        farthest_visible_index = current_index + 1
        for i in range(current_index + 2, len(path)):
            if line_of_sight(grid, path[current_index], path[i]):
                farthest_visible_index = i
            else:
                break
        simplified_path.append(path[farthest_visible_index])
        current_index = farthest_visible_index
        
    return simplified_path

def densify_path(path, max_distance):
    """
    Adds intermediate points to a path to ensure the distance between consecutive points
    is no more than max_distance.
    """
    if not path or len(path) < 2:
        return path
    
    densified_path = [path[0]]
    
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > max_distance:
            # Calculate how many segments we need to break this into
            num_segments = math.ceil(distance / max_distance)
            
            # Interpolate points
            for j in range(1, int(num_segments)):
                inter_x = p1[0] + (dx / num_segments) * j
                inter_y = p1[1] + (dy / num_segments) * j
                densified_path.append((int(inter_x), int(inter_y)))
        
        densified_path.append(p2)
        
    return densified_path

def display_path(grid, path, start_pos, end_pos, original_obstacles, simplified_waypoints=None):
    """Visualizes the grid, obstacles, and path using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, GRID_WIDTH)
    ax.set_ylim(0, GRID_HEIGHT)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("A* Pathfinding (Optimized with Safety Distance)")

    # Draw inflated obstacle grid cells
    grid_cols = len(grid[0])
    grid_rows = len(grid)
    for r in range(grid_rows):
        for c in range(grid_cols):
            if grid[r][c] == 1:
                rect = patches.Rectangle(
                    (r * RESOLUTION, c * RESOLUTION), RESOLUTION, RESOLUTION,
                    linewidth=0, facecolor='black', alpha=0.2, label='Safety Margin'
                )
                ax.add_patch(rect)
    
    # Draw original physical obstacles
    for obs in original_obstacles:
        rect = patches.Rectangle(
            (obs[0], obs[1]), obs[2], obs[3],
            linewidth=1, edgecolor='black', facecolor='#555555', label='Physical Obstacle'
        )
        ax.add_patch(rect)


    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    ax.plot(end_pos[0], end_pos[1], 'ro', markersize=10, label='Target')

    if simplified_waypoints:
        way_x, way_y = zip(*simplified_waypoints)
        ax.plot(way_x, way_y, 'o:', color='gray', linewidth=1, markersize=5, label='Simplified Waypoints')

    if path:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, '-', color='blue', linewidth=2, label='Final Path')
    
    # Create a unique legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.show()

def exec_path(path):
    """
    Executes the path by writing points to a file and waiting for an error signal.
    """
    if not path:
        print("Execution failed: No path to execute.")
        return

    print(f"\n--- Starting Path Execution ({len(path)} points) ---")
    for i, point in enumerate(path):
        x, y = point
        print(f"Moving to point {i+1}/{len(path)}: ({x}, {y})")

        try:
            with open(STAR_TARGET_FILE, 'w') as f:
                f.write(f"{x},{y},0\n")
        except Exception as e:
            print(f"Error writing to {STAR_TARGET_FILE}: {e}")
            return
        
        # Pause for 2 seconds after sending the point
        print("  Pausing for 2 seconds before checking error...")
        time.sleep(pause_time)

        while True:
            try:
                with open(ERROR_FILE, 'r') as f:
                    line = f.readline()
                    if not line:
                        time.sleep(0.2)
                        continue
                    dist_error, angle_error = [float(e.strip()) for e in line.split(',')]
                
                print(f"  Current error: Distance={dist_error:.2f}, Angle={angle_error:.2f}", end='\r')

                if dist_error < 15.0:
                    print(f"\n  Target point {i+1} reached.")
                    break
                
                time.sleep(0.5)

            except FileNotFoundError:
                print(f"  Waiting for '{ERROR_FILE}' to be created...")
                time.sleep(1)
            except (IOError, ValueError) as e:
                print(f"\n  Error reading {ERROR_FILE}: {e}. Retrying...")
                time.sleep(1)
    
    print("--- Path Execution Complete ---")


def main():
    """Main function to run the pathfinder."""
    grid_map_width = GRID_WIDTH // RESOLUTION
    grid_map_height = GRID_HEIGHT // RESOLUTION
    grid = [[0 for _ in range(grid_map_height)] for _ in range(grid_map_width)]
    
    # Create dummy files for testing
    if not os.path.exists(OBSTACLES_FILE):
        with open(OBSTACLES_FILE, 'w') as f: f.write("100,100,200,150\n300,200,350,400\n500,50,550,250\n")
    if not os.path.exists(ROBOT_POS_FILE):
        with open(ROBOT_POS_FILE, 'w') as f: f.write("50,50,0\n")
    if not os.path.exists(TARGET_FILE):
        with open(TARGET_FILE, 'w') as f: f.write("600,400\n")
    if not os.path.exists(ERROR_FILE):
        with open(ERROR_FILE, 'w') as f: f.write("100.0, 1.0\n")

    grid, original_obstacles = read_obstacles(OBSTACLES_FILE, grid, SAFETY_DISTANCE)
    start_pos = read_pos(ROBOT_POS_FILE)
    target_pos = read_pos(TARGET_FILE)

    if not start_pos or not target_pos:
        print("Could not read start or target position. Exiting.")
        return

    print(f"Start: {start_pos}, Target: {target_pos}, Resolution: {RESOLUTION}px, Safety Distance: {SAFETY_DISTANCE}px")
    print("Calculating path...")
    
    start_time = time.time()
    raw_path = astar(grid, start_pos, target_pos)
    end_time = time.time()
    print(f"A* calculation took {end_time - start_time:.4f} seconds.")

    if raw_path:
        print(f"Raw A* path found with {len(raw_path)} points.")
        
        print("Simplifying path...")
        simplified_path = simplify_path(grid, raw_path)
        print(f"Simplified path has {len(simplified_path)} waypoints.")
        
        print(f"Densifying path to max segment length of {MAX_PATH_SEGMENT_LENGTH}px...")
        final_path = densify_path(simplified_path, MAX_PATH_SEGMENT_LENGTH)
        print(f"Final executable path has {len(final_path)} points.")
        
        display_path(grid, final_path, start_pos, target_pos, original_obstacles, simplified_waypoints=simplified_path)
        exec_path(final_path)
    else:
        print("No path found.")
        display_path(grid, None, start_pos, target_pos, original_obstacles)

if __name__ == '__main__':
    main()

