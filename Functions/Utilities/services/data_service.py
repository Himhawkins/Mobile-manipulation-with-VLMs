# vision_dashboard/services/data_service.py

import numpy as np
import threading

class Data:
    """A thread-safe class to hold all shared application state in memory."""
    def __init__(self):
        self._lock = threading.Lock()
        self.stitched_frame: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)
        self.robot_poses: dict = {}
        self.static_obstacles: list = []
        self.dynamic_obstacles: list = [] # Added this attribute

    def update_state(self, frame: np.ndarray, poses: dict, dynamic_obstacles: list):
        """Atomically update multiple state variables."""
        with self._lock:
            self.stitched_frame = frame
            self.robot_poses = poses
            self.dynamic_obstacles = dynamic_obstacles # Added this line

    def set_static_obstacles(self, obstacles: list):
        with self._lock:
            self.static_obstacles = obstacles
            
    def get_full_state(self):
        """Returns a copy of all data for the UI to render in one call."""
        with self._lock:
            return {
                "stitched_frame": self.stitched_frame.copy(),
                "robot_poses": self.robot_poses.copy(),
                "static_obstacles": self.static_obstacles.copy(),
                "dynamic_obstacles": self.dynamic_obstacles.copy(), # Added this line
            }