# vision_dashboard/services/robot_service.py

import threading
import time
from Robot import motion

class Robot:
    """Service for controlling robot hardware."""
    def __init__(self, id_list=[0, 1, 2, 782]):
        self.id_list = id_list
        self.commands = {robot_id: [90, 90, 'open'] for robot_id in id_list}
        self.stop_event = threading.Event()
        self.server = None # Will be injected by RPCServer

        self.move_thread = threading.Thread(target=self._move_robot_loop, daemon=True)
        self.move_thread.start()

    def _set_server_reference(self, server_instance):
        """Dependency injection to access other services, like Data."""
        self.server = server_instance

    def get_pose(self, robot_id: int) -> dict:
        """Retrieves a single robot's pose from the Data service."""
        data_service = self.server.get_service("Data")
        return data_service.vars.robot_poses.get(robot_id, {})

    def set_command(self, robot_id: int, command: list):
        """Sets the command for a specific robot."""
        if robot_id in self.commands:
            self.commands[robot_id] = command
            print(f"[Robot Service] Set command for ID {robot_id}: {command}")
        else:
            raise ValueError(f"Robot ID {robot_id} not found.")

    def stop_all(self):
        """Stops all robot motion."""
        self.commands = {robot_id: [90, 90, 'open'] for robot_id in self.id_list}
        print("[Robot Service] Stop command issued.")

    def _move_robot_loop(self):
        while not self.stop_event.is_set():
            # In a real scenario, you'd pass self.commands to the motion function.
            # For this example, we'll simulate it.
            # motion.send_commands(self.commands)
            time.sleep(0.05)