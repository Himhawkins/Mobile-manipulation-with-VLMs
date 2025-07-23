def read_data(data_folder="Data"):
    """
    Reads robotics-related data from text files within a specified folder.

    This function reads data for arena corners, commands, errors, obstacles,
    and robot position from their respective .txt files. It parses the
    comma-separated values and returns them as NumPy arrays.

    Args:
        data_folder (str): The path to the folder containing the data files.
                           Defaults to "Data".

    Returns:
        dict: A dictionary containing the data from each file. The keys are:
              'arena_corners', 'command', 'error', 'obstacles', 'robot_pos'.
              The values are NumPy arrays containing the parsed data.
              Returns None if the data folder is not found.
    """
    if not os.path.isdir(data_folder):
        print(f"Error: Data folder '{data_folder}' not found.")
        return None

    # --- File Paths ---
    arena_corners_file = os.path.join(data_folder, "arena_corners.txt")
    command_file = os.path.join(data_folder, "command.txt")
    error_file = os.path.join(data_folder, "error.txt")
    obstacles_file = os.path.join(data_folder, "obstacles.txt")
    robot_pos_file = os.path.join(data_folder, "robot_pos.txt")

    # --- Data Storage ---
    data = {}

    try:
        # --- Read Arena Corners ---
        # Reads each line as an [x, y] coordinate
        with open(arena_corners_file, 'r') as f:
            lines = f.readlines()
            # Strips whitespace and splits by comma, then converts to int
            data['arena_corners'] = np.array([list(map(int, line.strip().split(','))) for line in lines])

        # --- Read Command ---
        # Reads the single line of command values
        with open(command_file, 'r') as f:
            line = f.readline().strip()
            data['command'] = np.array(list(map(float, line.split(','))))

        # --- Read Error ---
        # Reads the single line of error values
        with open(error_file, 'r') as f:
            line = f.readline().strip()
            # Filters out potential empty strings from trailing commas
            parts = [part for part in line.split(',') if part]
            data['error'] = np.array(list(map(float, parts)))

        # --- Read Obstacles ---
        # Reads each line as an [x, y, width, height] rectangle
        with open(obstacles_file, 'r') as f:
            lines = f.readlines()
            data['obstacles'] = np.array([list(map(int, line.strip().split(','))) for line in lines])

        # --- Read Robot Position ---
        # Reads the single line for [x, y, theta]
        with open(robot_pos_file, 'r') as f:
            line = f.readline().strip()
            data['robot_pos'] = np.array(list(map(float, line.split(','))))

    except FileNotFoundError as e:
        print(f"Error: A required data file was not found. {e}")
        return None
    except ValueError as e:
        print(f"Error: Could not convert data to a number. Check file formatting. {e}")
        return None

    return data