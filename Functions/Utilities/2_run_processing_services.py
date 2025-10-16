# vision_dashboard/2_run_processing_service.py

import time
import cv2
import os
from rpc_system import RPCClient
from vision.camera import CameraManager
from vision.arena import ArenaProcessor
from vision.detectors import ArucoDetector, BGSubtractor
from utils.settings import load_arena_settings
from config import REF_IMG_PATH

def main():
    """Runs the main vision processing loop, including background subtraction."""
    print("[Processing Service] Starting...")
    client = RPCClient()
    settings = load_arena_settings()

    # Initialize all components
    cam_manager = CameraManager(settings)
    arena_processor = ArenaProcessor(settings)
    aruco_detector = ArucoDetector()
    bg_subtractor = BGSubtractor()  # Initialize the background subtractor
    
    print("[Processing Service] Initialized. Starting main loop.")
    while True:
        try:
            caps = cam_manager.get_frame
            if not caps:
                time.sleep(1)
                continue
            # 1. Stitch the arena view from all cameras
            stitched_frame, _, _ = arena_processor.stitch_arena(caps)
            if stitched_frame is None:
                continue

            # 2. Detect ArUco markers to know where the robots are
            poses = aruco_detector.detect_pose(stitched_frame)
            
            # 3. Detect dynamic obstacles using background subtraction
            dynamic_obstacles = []
            if os.path.exists(REF_IMG_PATH):
                ref_frame = cv2.imread(REF_IMG_PATH)
                if ref_frame is not None:
                    # Detect all moving blobs
                    boxes, _ = bg_subtractor.detect(stitched_frame, ref_frame)
                    
                    # Filter out blobs that are actually the known robots
                    for x, y, w, h in boxes:
                        is_robot = any(
                            x < pose['x'] < x + w and y < pose['y'] < y + h
                            for pose in poses.values()
                        )
                        if not is_robot:
                            dynamic_obstacles.append((x, y, w, h))
            
            # 4. Push all updated data to the server in a single call
            client.Data.update_state(stitched_frame, poses, dynamic_obstacles)

            # time.sleep(0.03) # ~30 FPS loop

        except Exception as e:
            print(f"[Processing Service] Error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()