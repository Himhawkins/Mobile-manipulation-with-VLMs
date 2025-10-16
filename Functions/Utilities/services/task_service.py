# vision_dashboard/services/task_service.py

import cv2
from vision.detectors import VLDetector

class Tasks:
    """Service for long-running, complex tasks."""
    def __init__(self):
        self.server = None # Injected by RPCServer
        self.vlm_detector = VLDetector()

    def _set_server_reference(self, server_instance):
        self.server = server_instance

    def calibrate(self, obstacle_prompt: str):
        """Performs the calibration sequence and updates the Data service."""
        print(f"[Task Service] Starting calibration with prompt: '{obstacle_prompt}'")
        data_service = self.server.get_service("Data")
        
        # Get the latest frame from the data service
        frame = data_service.vars.stitched_frame
        if frame is None or frame.size == 0:
            raise RuntimeError("Cannot calibrate, no valid frame available.")
        
        # 1. Detect Arena (using frame boundaries as an example)
        h, w = frame.shape[:2]
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        data_service.set_arena_corners(corners)
        
        # 2. Detect Static Obstacles with VLM
        obstacles = self.vlm_detector.detect(frame, prompt=obstacle_prompt)
        bboxes = [obs['bbox'] for obs in obstacles]
        data_service.set_static_obstacles(bboxes)
        
        print("[Task Service] Calibration complete.")
        return True # Acknowledge success