#!/usr/bin/env python3
# modules/file_utils.py

import os
import json
import threading
import datetime
from pathlib import Path

# Global lock to ensure thread-safe writes to the shared command.json file
_COMMAND_JSON_LOCK = threading.Lock()

def _dbg(msg: str, tick=None, every=5):
    """A shared debug logging function."""
    if tick is not None and tick % every != 0:
        return
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[DBG {ts}] {msg}")

def _ensure_txt(path: str, default_lines: list):
    """Create a text file with default content if it doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w') as f:
            for line in default_lines:
                f.write(f"{line}\n")

def _clear_json_file(path: str, default_content={"robots": []}):
    """Clear a JSON file by writing default content to it."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(default_content, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not clear JSON file '{path}': {e}")

def _robot_id_exists(pose_file: str, robot_id: int) -> bool:
    """Check if a specific robot ID exists in the pose file."""
    try:
        with open(pose_file, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p for p in line.replace(" ", ",").split(",") if p]
                if len(parts) >= 4:
                    try:
                        rid = int(float(parts[0]))
                        if rid == int(robot_id):
                            return True
                    except (ValueError, IndexError):
                        continue
        return False
    except FileNotFoundError:
        return False

def _load_json_robust(json_path: str, default_content={"robots": []}) -> dict:
    """Load a JSON file, returning a default structure if it's missing or corrupt."""
    if not os.path.exists(json_path):
        return default_content
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            # Basic validation
            if isinstance(data, dict) and "robots" in data and isinstance(data["robots"], list):
                return data
    except (json.JSONDecodeError, TypeError):
        pass  # File is corrupt or not in the expected format
    return default_content

def _trace_append_point(json_path: str, robot_id: int, x: float, y: float):
    """Append a coordinate to a robot's trace path in a JSON file."""
    data = _load_json_robust(json_path)
    rid = int(robot_id)
    entry = next((r for r in data.get("robots", []) if r.get("id") == rid), None)

    if entry is None:
        entry = {"id": rid, "points": []}
        data["robots"].append(entry)

    # Avoid adding duplicate consecutive points
    points = entry.setdefault("points", [])
    if points and int(round(points[-1][0])) == int(round(x)) and int(round(points[-1][1])) == int(round(y)):
        return

    points.append([int(round(x)), int(round(y))])
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[WARN] Trace file write failed: {e}")

def _astar_dump_clear(json_path: str, robot_id: int):
    """Remove all A* path segments for a specific robot from the dump file."""
    data = _load_json_robust(json_path)
    rid = int(robot_id)
    data["robots"] = [r for r in data.get("robots", []) if r.get("id") != rid]
    _clear_json_file(json_path, default_content=data)

def _astar_dump_append(json_path: str, robot_id: int, goal_xy, path_pts, *, seg_type="normal"):
    """Append a planned A* segment to the dump file for visualization."""
    data = _load_json_robust(json_path)
    rid = int(robot_id)
    entry = next((r for r in data.get("robots", []) if r.get("id") == rid), None)

    if entry is None:
        entry = {"id": rid, "segments": []}
        data["robots"].append(entry)

    new_segment = {
        "type": seg_type,
        "goal": [int(goal_xy[0]), int(goal_xy[1])],
        "path": [[int(p[0]), int(p[1])] for p in path_pts],
    }
    entry.setdefault("segments", []).append(new_segment)
    try:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[WARN] A* dump append failed: {e}")

def _get_robot_path_from_json(json_path: str, robot_id: int) -> list | None:
    """Extract a specific robot's path from the main paths JSON file."""
    data = _load_json_robust(json_path)
    rid = int(robot_id)
    
    for entry in data.get("robots", []):
        if entry.get("id") == rid:
            path_data = entry.get("path", [])
            parsed_path = []
            for item in path_data:
                if not isinstance(item, (list, tuple)) or len(item) < 2:
                    continue
                x, y = item[0], item[1]
                theta = item[2] if len(item) > 2 and isinstance(item[2], (int, float)) else 0.0
                action = item[3] if len(item) > 3 else (item[2] if isinstance(item[2], str) else 0)
                parsed_path.append((x, y, theta, action))
            return parsed_path if parsed_path else None
    return None

def _path_for_robot_exists(json_path: str, robot_id: int) -> bool:
    """Check if a path exists for a given robot ID."""
    return _get_robot_path_from_json(json_path, robot_id) is not None


class FileInterface:
    """An abstraction layer for reading from and writing to all necessary files."""

    def __init__(self, target_file: str, pose_file: str, command_file: str, error_file: str, robot_id: int = None):
        self.target_file = target_file
        self.pose_file = pose_file
        self.command_file = command_file
        self.error_file = error_file
        self.robot_id = int(robot_id) if robot_id is not None else None
        self._initialize_files()

    def _initialize_files(self):
        """Ensure all required files exist with default content."""
        if not str(self.target_file).lower().endswith(".json"):
            _ensure_txt(self.target_file, ["0,0,0"])
        _ensure_txt(self.pose_file, ["0,0,0.0"])
        _ensure_txt(self.error_file, ["0.0,0.0"])
        # Ensure command.json exists and is valid
        if not os.path.exists(self.command_file):
            _clear_json_file(self.command_file)

    def _load_lines(self, path: str) -> list[str]:
        """Safely load all non-empty lines from a text file."""
        try:
            with open(path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"[WARN] Error reading {path}: {e}")
            return []

    def read_targets(self) -> list:
        """Read a robot's target waypoints from a JSON or legacy TXT file."""
        if str(self.target_file).lower().endswith(".json"):
            if self.robot_id is None:
                print("[WARN] Target file is JSON but no robot_id was provided.")
                return []
            return _get_robot_path_from_json(self.target_file, self.robot_id) or []
        
        # Legacy TXT file support
        lines = self._load_lines(self.target_file)
        out = []
        for line in lines:
            try:
                parts = [p.strip() for p in line.split(',') if p.strip()]
                if len(parts) < 2: continue
                x = float(parts[0])
                y = float(parts[1])
                delay_ms = int(float(parts[2])) if len(parts) >= 3 else 0
                out.append((x, y, 0.0, delay_ms))
            except (ValueError, IndexError) as e:
                print(f"[WARN] Skipping malformed target line '{line}': {e}")
        return out

    def read_pos(self) -> tuple[float, float, float] | None:
        """Read the robot's current position (x, y, theta_rad) from the pose file."""
        lines = self._load_lines(self.pose_file)
        if not lines:
            return None

        if self.robot_id is not None:
            for line in reversed(lines):
                parts = [p for p in line.replace(" ", ",").split(",") if p]
                if len(parts) == 4:
                    try:
                        rid = int(float(parts[0]))
                        if rid == self.robot_id:
                            return float(parts[1]), float(parts[2]), float(parts[3])
                    except (ValueError, IndexError):
                        continue
            return None  # ID not found
        
        # Legacy single-robot support
        try:
            x_str, y_str, th_str = lines[-1].split(',')
            return float(x_str), float(y_str), float(th_str)
        except (ValueError, IndexError) as e:
            print(f"[WARN] Malformed legacy pose '{lines[-1]}': {e}")
            return None

    def _update_command_json(self, updates: dict):
        """Thread-safely update the command.json file for this robot."""
        if self.robot_id is None:
            print("[ERR] Cannot write command without a robot_id.")
            return

        with _COMMAND_JSON_LOCK:
            data = _load_json_robust(self.command_file)
            robot_entry = next((r for r in data["robots"] if r.get("id") == self.robot_id), None)
            
            if robot_entry is None:
                robot_entry = {"id": self.robot_id}
                data["robots"].append(robot_entry)
            
            robot_entry.update(updates)
            
            try:
                with open(self.command_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"[ERR] Saving command to {self.command_file} failed: {e}")

    def write_wheel_command(self, left_speed: float, right_speed: float):
        """Write motor speeds to the command file."""
        updates = {"left": int(round(left_speed)), "right": int(round(right_speed))}
        self._update_command_json(updates)

    def write_gripper_command(self, state: str):
        """Write a gripper state ('open' or 'close') to the command file."""
        if state.lower() in ["open", "close"]:
            self._update_command_json({"gripper": state.lower()})

    def log_error(self, dist_err: float, angle_err: float):
        """Log PID errors to a file for debugging."""
        try:
            with open(self.error_file, 'w') as f:
                f.write(f"{dist_err},{angle_err}\n")
        except IOError as e:
            print(f"[ERR] Saving error log failed: {e}")