# vision_dashboard/config.py

# Directory for saving run-time data like captures and detected object coordinates
DATA_DIR = "Data"

# Path to the reference image used for background subtraction
REF_IMG_PATH = f"{DATA_DIR}/frame_img.png"

# Paths for detected object data
OBSTACLES_VLM_PATH = f"{DATA_DIR}/obstacles.txt"
ROBOT_POS_PATH = f"{DATA_DIR}/robot_pos.txt"
ARENA_CORNERS_PATH = f"{DATA_DIR}/arena_corners.txt"