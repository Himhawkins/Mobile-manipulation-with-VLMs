import os
import sys
import numpy as np

def _ensure_path(path: str):
    """Create parent folders and an empty file if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as _:
            pass  # create empty file

def read_data(data_folder="Data"):
    """
    Reads robotics-related data from text files within a specified folder.

    If files/folders do not exist, they are created empty and this function
    returns safe defaults:
      - arena_corners: empty array of shape (0, 2), dtype=int
      - command:       empty float array (0,)
      - error:         empty float array (0,)
      - obstacles:     empty array of shape (0, 4, 2), dtype=int
      - robot_pos:     float array of shape (N, 4): [id, x, y, theta].
                       If file is empty => shape (0, 4).

    Files and expected formats:
      - arena_corners.txt: one "x,y" per line (4 lines typical)
      - command.txt:       one line of CSV floats
      - error.txt:         one line of CSV floats (may have trailing commas)
      - obstacles.txt:     one obstacle per line as "(x1,y1),(x2,y2),(x3,y3),(x4,y4)"
      - robot_pos.txt:     multi-line, each: "id,x,y,theta"
                           (backward-tolerant: "x,y,theta" -> id assumed 0)
    """
    # Ensure folder exists
    os.makedirs(data_folder, exist_ok=True)

    # --- File Paths ---
    arena_corners_file = os.path.join(data_folder, "arena_corners.txt")
    command_file       = os.path.join(data_folder, "command.txt")
    error_file         = os.path.join(data_folder, "error.txt")
    obstacles_file     = os.path.join(data_folder, "obstacles.txt")
    robot_pos_file     = os.path.join(data_folder, "robot_pos.txt")

    # Create files if missing (empty)
    for p in [arena_corners_file, command_file, error_file, obstacles_file, robot_pos_file]:
        _ensure_path(p)  # assumes your existing helper

    data = {}

    try:
        # --- Read Arena Corners ---
        with open(arena_corners_file, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        if lines:
            data['arena_corners'] = np.array([list(map(int, ln.split(','))) for ln in lines], dtype=int)
        else:
            data['arena_corners'] = np.zeros((0, 2), dtype=int)

        # --- Read Command ---
        with open(command_file, 'r') as f:
            line = f.readline().strip()
        if line:
            data['command'] = np.array(list(map(float, line.split(','))), dtype=float)
        else:
            data['command'] = np.zeros((0,), dtype=float)

        # --- Read Error ---
        with open(error_file, 'r') as f:
            line = f.readline().strip()
        if line:
            parts = [p for p in line.split(',') if p]  # drop empty entries from trailing commas
            data['error'] = np.array(list(map(float, parts)), dtype=float)
        else:
            data['error'] = np.zeros((0,), dtype=float)

        # --- Read Obstacles ---
        with open(obstacles_file, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        parsed_obstacles = []
        for line in lines:
            s = line.replace("(", "").replace(")", "")
            parts = s.split(",")
            # Expect 8 integers for 4 corners
            if len(parts) != 8:
                continue  # Skip malformed lines
            try:
                nums = list(map(int, parts))
            except ValueError:
                continue
            corners = [(nums[i], nums[i+1]) for i in range(0, 8, 2)]  # [(x1,y1),...,(x4,y4)]
            parsed_obstacles.append(corners)
        if parsed_obstacles:
            data['obstacles'] = np.array(parsed_obstacles, dtype=int)  # (N, 4, 2)
        else:
            data['obstacles'] = np.zeros((0, 4, 2), dtype=int)

        # --- Read Robot Positions (multi-robot) ---
        robots = []
        with open(robot_pos_file, 'r') as f:
            for raw in f:
                ln = raw.strip()
                if not ln or ln.startswith("#"):
                    continue
                # Normalize separators (allow spaces)
                parts = [p for p in ln.replace(" ", ",").split(",") if p != ""]
                # Accept either 4 fields (id,x,y,theta) or 3 fields (x,y,theta -> id=0)
                if len(parts) == 4:
                    try:
                        rid   = int(float(parts[0]))  # tolerate "7.0"
                        x     = float(parts[1])
                        y     = float(parts[2])
                        theta = float(parts[3])
                        robots.append([rid, x, y, theta])
                    except ValueError:
                        continue
                elif len(parts) == 3:
                    # Backward compatibility
                    try:
                        x     = float(parts[0])
                        y     = float(parts[1])
                        theta = float(parts[2])
                        robots.append([0, x, y, theta])  # id=0
                    except ValueError:
                        continue
                else:
                    continue

        if robots:
            data['robot_pos'] = np.array(robots, dtype=float)  # shape (N,4): [id,x,y,theta]
        else:
            data['robot_pos'] = np.zeros((0, 4), dtype=float)

    except Exception as e:
        print(f"Error while reading data files: {e}")
        return None

    return data

if __name__ == "__main__":
    data = read_data("Data")
    if data is None:
        sys.exit(1)

    for key, arr in data.items():
        print(f"{key}:")
        print(arr)
        print()
