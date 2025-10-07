# main.py

import threading
import time
from pathlib import Path

# Import from our new modules
from PathControl.file_utils import FileInterface, _robot_id_exists, _path_for_robot_exists, _clear_json_file, _astar_dump_clear
from PathControl.robot_controller import PIDController

# Global registry of running robot controller threads
_robot_threads = {}
_robot_threads_lock = threading.Lock()

def run_controller(**kwargs):
    """A wrapper to initialize and run the PID controller."""
    iface = FileInterface(
        kwargs.get('target_file'),
        kwargs.get('pose_file'),
        kwargs.get('command_file'),
        kwargs.get('error_file'),
        robot_id=kwargs.get('robot_id')
    )
    # Pass all kwargs to the controller
    controller = PIDController(iface, **kwargs)
    controller.run(stop_event=kwargs.get('stop_event'))

def exec_robot_create_thread(robot_id: int, robot_padding: int = 30):
    robot_id = int(robot_id)
    target_file  = str(Path("Targets") / "paths.json")
    pose_file    = str(Path("Data") / "robot_pos.txt")
    
    if not _robot_id_exists(pose_file, robot_id):
        return "Selected ID doesn't exist"
    if not _path_for_robot_exists(target_file, robot_id):
        return "Path for specific robot doesn't exist. Generate path first"

    with _robot_threads_lock:
        if _robot_threads.get(robot_id, {}).get("ctrl_thread", threading.Thread()).is_alive():
            return f"Thread already running for robot {robot_id}"

        stop_event = threading.Event()
        
        # Prepare kwargs for the runner
        controller_kwargs = {
            "target_file": target_file,
            "pose_file": pose_file,
            "command_file": str(Path("Data") / "command.json"),
            "error_file": str(Path("Data") / "error.txt"),
            "stop_event": stop_event,
            "robot_id": robot_id,
            "robot_padding": robot_padding,
            # Add any other PIDController params here if you want to configure them
        }
        
        def _runner():
            """This function is the target for the new thread."""
            try:
                # This call blocks until the robot finishes or is stopped
                run_controller(**controller_kwargs)
            finally:
                # This block runs after run_controller is complete
                print(f"Execution finished for robot {robot_id}. Cleaning up...")
                astar_dump_path = str(Path("Data") / "astar_segments.json")
                _astar_dump_clear(astar_dump_path, robot_id) # <-- ADD THIS LINE

                with _robot_threads_lock:
                    if robot_id in _robot_threads:
                        _robot_threads[robot_id]["ctrl_done"] = True

        ctrl_thread = threading.Thread(target=_runner, name=f"robot-{robot_id}", daemon=True)
        ctrl_thread.start()
        _robot_threads[robot_id] = {
            "ctrl_thread": ctrl_thread,
            "stop_event": stop_event,
            "started_at": time.time(),
            "ctrl_done": False,
        }
    return f"Started controller thread for robot {robot_id}"

def stop_robot_thread(robot_id: int, join_timeout: float = 5.0):
    with _robot_threads_lock:
        info = _robot_threads.get(int(robot_id))
        if not info:
            return f"No running thread for robot {int(robot_id)}"
        
        info["stop_event"].set()
        thread = info.get("ctrl_thread")

    if thread:
        thread.join(timeout=join_timeout)

    with _robot_threads_lock:
        if thread and not thread.is_alive():
            _robot_threads.pop(int(robot_id), None)
            _clear_json_file(str(Path("Data") / "trace_paths.json"))
            _astar_dump_clear(str(Path("Data") / "astar_segments.json"), int(robot_id))
            return f"Stopped thread for robot {int(robot_id)}"
        else:
            return f"Stop requested; robot {int(robot_id)} is still stopping"

if __name__ == "__main__":
    ROBOT_ID = 3
    PADDING = 30
    
    try:
        print(f"Starting controller for robot ID: {ROBOT_ID}")
        start_msg = exec_robot_create_thread(robot_id=ROBOT_ID, robot_padding=PADDING)
        print(start_msg)

        if "Started" in start_msg:
            # Keep the main thread alive to let the controller run
            # You can add other logic here, like a command-line interface
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterruption detected. Stopping robot thread...")
        stop_msg = stop_robot_thread(robot_id=ROBOT_ID)
        print(stop_msg)
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
        stop_robot_thread(robot_id=ROBOT_ID)