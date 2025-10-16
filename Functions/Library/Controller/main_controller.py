# main_controller.py

import math
import time
import threading
from pathlib import Path

import read_data
import obstacle_avoidance

# --- Constants ---
DATA_FOLDER = "Data"
TARGET_FILE = str(Path("Targets") / "paths.json")
POSE_FILE = str(Path(DATA_FOLDER) / "robot_pos.txt")
COMMAND_FILE = str(Path(DATA_FOLDER) / "command.txt")
FRAME_SIZE = (480, 640) # (Height, Width) - Adjust as needed

# --- PID and Motion Parameters ---
KP_DIST = 0.18
KP_ANG = 3.0
SPEED_NEUTRAL = 90
SPEED_MIN = 70
SPEED_MAX = 110
MAX_LINEAR_VEL = 30
DIST_TOLERANCE = 18.0
FINAL_DIST_TOLERANCE = 10.0
ANG_TOLERANCE_DEG = 18.0

class PIDController:
    """Manages robot motion to follow a path using PID control."""
    
    def __init__(self, obstacle_manager, robot_id):
        self.obs_manager = obstacle_manager
        self.robot_id = robot_id
        
        self.prev_ang_err = 0.0
        self.integral_ang = 0.0
        self.prev_time = time.time()

    def _normalize_angle(self, angle):
        """Normalize angle to be within [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def run(self, stop_event=None):
        """Main control loop to navigate through all targets."""
        print(f"Controller started for robot {self.robot_id}.")
        targets = read_data.get_robot_path(self.robot_id)
        if not targets:
            print(f"No targets found for robot {self.robot_id}. Exiting.")
            return

        for i, (goal_x, goal_y, goal_theta, action) in enumerate(targets):
            print(f"--- Target {i+1}/{len(targets)}: ({goal_x}, {goal_y}) ---")
            
            path_segment = None
            path_index = 0

            while not (stop_event and stop_event.is_set()):
                current_x, current_y, current_theta = read_data.read_robot_pose(self.robot_id)

                # Check if we are close enough to the final goal of this segment
                dist_to_final_goal = math.hypot(goal_x - current_x, goal_y - current_y)
                if dist_to_final_goal < FINAL_DIST_TOLERANCE:
                    print(f"Reached final target {i+1}.")
                    break
                
                # If path is finished or invalid, plan a new one
                if not path_segment or path_index >= len(path_segment):
                    print("Planning new path segment...")
                    path_segment = obstacle_avoidance.get_safe_path(self.obs_manager, (current_x, current_y), (goal_x, goal_y))
                    path_index = 0
                    if not path_segment:
                        print("Failed to plan. Waiting...")
                        read_data.write_command(self.robot_id, SPEED_NEUTRAL, SPEED_NEUTRAL,0)
                        time.sleep(1.0)
                        continue
                
                # Get current waypoint from the path
                waypoint_x, waypoint_y = path_segment[path_index]
                dist_to_waypoint = math.hypot(waypoint_x - current_x, waypoint_y - current_y)

                # If close to waypoint, advance to the next one
                if dist_to_waypoint < DIST_TOLERANCE:
                    path_index += 1
                    if path_index >= len(path_segment):
                        print("Finished path segment, but not at final goal. Re-planning.")
                        continue # Force re-plan on next loop
                    waypoint_x, waypoint_y = path_segment[path_index]

                # --- PID Calculation ---
                dt = time.time() - self.prev_time
                self.prev_time += dt

                # Angular error
                target_heading = math.atan2(waypoint_y - current_y, waypoint_x - current_x)
                ang_err = self._normalize_angle(target_heading - current_theta)

                # PID terms for angle
                self.integral_ang += ang_err * dt
                deriv_ang = (ang_err - self.prev_ang_err) / dt
                self.prev_ang_err = ang_err
                
                ang_control = (KP_ANG * ang_err) + (0.02 * self.integral_ang) + (0.9 * deriv_ang)

                # Proportional term for distance
                lin_control = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, KP_DIST * dist_to_waypoint))
                
                # Proximity slowdown
                clr = self.obs_manager.clearance_dt[int(current_y), int(current_x)]
                slowdown_factor = min(1.0, clr / self.obs_manager.spacing_px)
                lin_control *= slowdown_factor

                # Combine controls to get motor speeds
                forward_gain = max(0.0, math.cos(ang_err))**1.5
                left_speed = SPEED_NEUTRAL + (lin_control * forward_gain) - ang_control
                right_speed = SPEED_NEUTRAL + (lin_control * forward_gain) + ang_control
                
                left_speed = max(SPEED_MIN, min(SPEED_MAX, left_speed))
                right_speed = max(SPEED_MIN, min(SPEED_MAX, right_speed))
                
                read_data.write_command(self.robot_id, left_speed, right_speed,0)
                time.sleep(0.05)
            
            # Stop robot and handle action at target
            read_data.write_command(self.robot_id, SPEED_NEUTRAL, SPEED_NEUTRAL,action)
            print(f"Action at target: {action}")
            # if isinstance(action, (int, float)) and action > 0:
            #     time.sleep(action / 1000.0) 
            # (Gripper 'open'/'close' logic would go here)
            #THIS HAS BEEN MOVED TO MOTION.PY
            if stop_event and stop_event.is_set():
                break

        print(f"Controller finished for robot {self.robot_id}.")
        read_data.write_command(self.robot_id, SPEED_NEUTRAL, SPEED_NEUTRAL,0)


def run_robot_controller(robot_id, robot_padding=30, spacing_px=40, stop_event=None):
    """Initializes and runs the controller for a single robot."""
    read_data.initialize_files()
    
    obs_manager = obstacle_avoidance.ObstacleManager(
        data_folder=DATA_FOLDER,
        frame_size=FRAME_SIZE,
        spacing_px=spacing_px,
        robot_padding=robot_padding,
        robot_id=robot_id
    )

    controller = PIDController(obs_manager, robot_id)
    controller.run(stop_event=stop_event)


if __name__ == "__main__":
    # Example: Run controller for robot with ID 782
    ROBOT_TO_RUN = 782
    
    print(f"Starting controller for Robot ID: {ROBOT_TO_RUN}")
    
    # To run in a thread and stop it after some time:
    stop_event = threading.Event()
    controller_thread = threading.Thread(
        target=run_robot_controller,
        args=(ROBOT_TO_RUN,),
        kwargs={'stop_event': stop_event}
    )
    
    controller_thread.start()
    
    try:
        # Let it run for 5 minutes, for example
        controller_thread.join(timeout=300)
    except KeyboardInterrupt:
        print("\nStopping controller due to user interrupt.")
    finally:
        if controller_thread.is_alive():
            stop_event.set()
            controller_thread.join() # Wait for thread to finish cleanly
    
    print("Execution complete.")