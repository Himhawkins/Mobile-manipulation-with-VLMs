# read_data.py

import os
import json
from pathlib import Path
from Utilities.rpc_system import RPCClient

rpc=RPCClient()

def _ensure_txt(path, default_lines):
    """Creates a file with default content if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write("\n".join(default_lines) + "\n")

def initialize_files(data_folder="Data"):
    """Ensure all necessary data files exist."""
    _ensure_txt(os.path.join(data_folder, "obstacles.txt"), [])
    _ensure_txt(os.path.join(data_folder, "realtime_obstacles.txt"), [])
    _ensure_txt(os.path.join(data_folder, "arena_corners.txt"), [])
    _ensure_txt(os.path.join(data_folder, "robot_pos.txt"), ["0,0,0.0"])
    _ensure_txt(os.path.join(data_folder, "command.txt"), ["90,90"])
    _ensure_txt(os.path.join(data_folder, "error.txt"), ["0.0,0.0"])

def get_robot_path(robot_id):
    """
    Returns a list of (x, y, theta_rad, action) for a specific robot.
    Action can be an int (delay_ms) or a string ('open'/'close').
    # """
    # if not os.path.exists(json_path):
    #     return None
    # with open(json_path, "r") as f:
    #     data = json.load(f)

    # for entry in data.get("robots", []):
    #     if int(entry.get("id", -1)) != int(robot_id):
    #         continue
        
    #     path_data = entry.get("path", [])
    #     if not path_data:
    #         return None # Path exists but is empty
            
    #     # Simplified logic: assume path is a list of lists/tuples
    #     out = []
    #     for t in path_data:
    #         x, y = int(t[0]), int(t[1])
    #         theta = float(t[2]) if len(t) > 2 and isinstance(t[2], (int, float)) else 0.0
    #         action = t[3] if len(t) > 3 else (t[2] if isinstance(t[2], str) else 0)
    #         out.append((x, y, theta, action))
    #     return out
    # return None
    return rpc.Data.Trajectory(robot_id)


def read_robot_pose(robot_id):
    """Reads the last known pose (x, y, theta_rad) for a specific robot."""
    # try:
    #     with open(pose_file, 'r') as f:
    #         lines = [line.strip() for line in f if line.strip()]
    #     for line in reversed(lines):
    #         parts = [p for p in line.replace(" ", ",").split(",") if p]
    #         if len(parts) >= 4 and int(float(parts[0])) == robot_id:
    #             return float(parts[1]), float(parts[2]), float(parts[3])
    # except (IOError, ValueError, IndexError):
    #     pass
    # return 0.0, 0.0, 0.0
    return rpc.Robot.get_pose(robot_id)

def read_obstacle_polygons(name):
    # """Reads obstacle corner data from a file."""
    # polys = []
    # try:
    #     with open(obstacle_file, "r") as f:
    #         for line in f:
    #             try:
    #                 nums = list(map(int, line.strip().replace("(", "").replace(")", "").split(",")))
    #                 if len(nums) == 8:
    #                     polys.append([(nums[i], nums[i+1]) for i in range(0, 8, 2)])
    #             except ValueError:
    #                 continue
    # except FileNotFoundError:
    #     pass
    # return polys
    return rpc.Data.arena['obstacles'][name]


def read_arena_corners(arena_file):
    """Reads arena corner points from a file."""
    rpc.Data.arena['corners']

def read_all_robot_positions(robot_pos_file):
    """Reads all robot positions from the pose file."""
    # robots = []
    # try:
    #     with open(robot_pos_file, "r") as f:
    #         for line in f:
    #             parts = [p for p in line.strip().replace(" ", ",").split(",") if p]
    #             if len(parts) >= 4:
    #                 try:
    #                     robots.append((int(float(parts[0])), float(parts[1]), float(parts[2])))
    #                 except ValueError:
    #                     continue
    # except FileNotFoundError:
    #     pass
    pos=[]
    for i in rpc.Robots.id_list:
        pos.append(rpc.Robot.get_pos(i))
    return pos

def write_command(robot_id, left_speed, right_speed,gripper):
    rpc.Robot.set_command(id,[left_speed,right_speed,gripper])