#!/usr/bin/env python3
"""
Modular Robot Controller Library

Provides classes and a helper function for file-based robot control using a PID controller.
Targets, pose, command, and error logs are stored in four separate plain-text files.
Running the module directly will initialize default text files and execute a test run.

NOTE: Target delay values are in *milliseconds* (ms).
"""
import math
import os
import time
from pathlib import Path


def _ensure_txt(path, default_lines):
    """
    Ensures a plain-text file exists at the given path with default lines if it doesn't.
    Creates parent directories if they don't exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        try:
            with open(path, 'w') as f:
                for line in default_lines:
                    f.write(f"{line}\n")
            print(f"Initialized new text file: {path}")
        except IOError as e:
            print(f"Error initializing text file {path}: {e}")


class FileInterface:
    """
    Handles text-based I/O for targets, pose, command, and error logs.
    """
    def __init__(self, target_file, pose_file, command_file, error_file):
        self.target_file = target_file
        self.pose_file = pose_file
        self.command_file = command_file
        self.error_file = error_file
        self._initialize_files()

    def _initialize_files(self):
        # default targets: each line "x,y,delay_ms" (delay in milliseconds)
        # NOTE: reader also accepts legacy "x,y" (delay defaults to 0)
        default_targets = [
            "5,0,0",
            "5,5,1500",   # pause 1500 ms (1.5 s)
            "0,5,0",
            "0,0,2000",   # pause 2000 ms (2.0 s)
        ]
        _ensure_txt(self.target_file, default_targets)

        # default pose: single "x,y,theta" line
        default_pose = ["0,0,0.0"]
        _ensure_txt(self.pose_file, default_pose)

        # default command: single "left_speed,right_speed" line
        default_cmd = ["90,90"]
        _ensure_txt(self.command_file, default_cmd)

        # default error: single "dist_err,ang_err" line
        default_error = ["0.0,0.0"]
        _ensure_txt(self.error_file, default_error)

    def _load_lines(self, path):
        """Loads non-empty, stripped lines from a text file."""
        try:
            with open(path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: File not found at {path}. It might not have been created correctly.")
            raise
        except Exception as e:
            print(f"Error reading {path}: {e}")
            # send neutral command if any read error occurs
            try:
                self.write_command(90, 90)
                print("Sent neutral command speeds 90,90 due to read error.")
            except Exception:
                pass
            # return fallbacks
            if path == self.target_file:
                return []
            if path == self.pose_file:
                return ["0.0,0.0,0.0"]
            return []

    def read_targets(self):
        """
        Reads waypoints from the targets text file.
        Accepts 'x,y' or 'x,y,delay_ms' (delay in milliseconds).
        Returns: list of (x: float, y: float, delay_ms: float)
        """
        lines = self._load_lines(self.target_file)
        targets = []
        for line in lines:
            try:
                parts = [p.strip() for p in line.split(',') if p.strip() != ""]
                if len(parts) < 2:
                    raise ValueError("Too few values")
                x = float(parts[0])
                y = float(parts[1])
                delay_ms = float(parts[2]) if len(parts) >= 3 else 0.0  # ms
                targets.append((x, y, delay_ms))
            except Exception as e:
                print(f"Warning: Skipping malformed target line '{line}': {e}")
        return targets

    def read_pos(self):
        """
        Reads the robot's current pose (x, y, theta) from the pose text file.
        Expects a single line "x,y,theta".
        """
        lines = self._load_lines(self.pose_file)
        if not lines:
            return 0.0, 0.0, 0.0
        try:
            x_str, y_str, th_str = lines[0].split(',')
            return float(x_str), float(y_str), float(th_str)
        except Exception as e:
            print(f"Warning: Malformed pose line '{lines[0]}': {e}")
            return 0.0, 0.0, 0.0

    def write_command(self, left_speed, right_speed):
        """
        Writes the motor commands to the command text file as "left_speed,right_speed".
        """
        try:
            with open(self.command_file, 'w') as f:
                f.write(f"{int(left_speed)},{int(right_speed)}\n")
        except IOError as e:
            print(f"Error saving command to {self.command_file}: {e}")

    def log_error(self, dist_err, angle_err):
        """
        Logs distance and angle errors to the error text file as "dist_err,ang_err".
        """
        try:
            with open(self.error_file, 'w') as f:
                f.write(f"{dist_err},{angle_err}\n")
        except IOError as e:
            print(f"Error saving error to {self.error_file}: {e}")


class PIDController:
    """
    PID-based controller for waypoint navigation.
    """
    def __init__(
        self, iface: FileInterface,
        Kp_dist=0.2, Kp_ang=4.0, Ki_ang=0.07, Kd_ang=0.7,
        dist_tolerance=0.05, ang_tolerance=0.1, final_distance_tol=0.05
    ):
        self.iface = iface
        self.Kp_dist = Kp_dist
        self.Kp_ang = Kp_ang
        self.Ki_ang = Ki_ang
        self.Kd_ang = Kd_ang
        self.dist_tolerance = dist_tolerance
        self.ang_tolerance = ang_tolerance
        self.integral_ang = 0.0
        self.prev_ang_err = 0.0
        self.prev_time = time.time()
        self.final_distance_tol = final_distance_tol

    @staticmethod
    def normalize(angle):
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def adjust_speed(speed):
        """Custom deadzone and non-linear adjustments around 90."""
        if 89.75 <= speed <= 90.25:
            return 90.0
        if 90.25 < speed <= 95:
            return speed + 1.0
        if 85 <= speed < 89.75:
            return speed - 1.0
        return speed

    def _write_targets_header(self, idx, total, targets):
        """
        Overwrites the targets file to include a header "(idx/total)" followed by all waypoints.
        Writes x,y only to keep header tidy; delays are preserved in memory.
        """
        try:
            with open(self.iface.target_file, 'w') as f:
                f.write(f"({idx}/{total})\n")
                for x, y, d_ms in targets:
                    f.write(f"{int(x)},{int(y)}\n")
        except Exception as e:
            print(f"Error updating targets file header: {e}")

    def _pause_at_checkpoint(self, delay_s, stop_event=None):
        """
        Pause for delay_s seconds, remaining responsive to stop_event.
        Sends neutral command once at the start of the pause.
        """
        if delay_s <= 0:
            return
        self.iface.write_command(90, 90)  # neutral command during hold
        remaining = float(delay_s)
        step = 0.05  # 50 ms granularity for responsiveness
        start = time.time()
        while remaining > 0:
            if stop_event and stop_event.is_set():
                break
            sleep_dur = min(step, remaining)
            time.sleep(sleep_dur)
            remaining = delay_s - (time.time() - start)

    def run(self, stop_event=None):
        """
        Executes the PID control loop to navigate through the targets.
        Pauses at each checkpoint for the target's delay (in milliseconds).
        """
        targets = self.iface.read_targets()   # list[(x, y, delay_ms)]
        if not targets:
            print("No targets to process. Exiting controller.")
            return
        total = len(targets)
        idx = 0
        base_speed = 90
        max_lin = 15

        # self._write_targets_header(idx, total, targets)

        while idx < total and not (stop_event and stop_event.is_set()):
            now = time.time()
            dt = now - self.prev_time
            if dt <= 0:
                time.sleep(0.01)
                continue
            self.prev_time = now

            x, y, theta = self.iface.read_pos()
            if (x, y, theta) == (0.0, 0.0, 0.0) and not self.iface._load_lines(self.iface.pose_file):
                # pose file is blank, send stop command
                self.iface.write_command(90, 90)
                print("robot_pos.txt is blank — stopping robot.")
                time.sleep(0.05)
                continue
            tx, ty, delay_ms = targets[idx]  # <-- includes delay (ms)
            dist_err = math.hypot(tx - x, ty - y)
            heading = math.atan2(ty - y, tx - x)

            # Reached current target?
            if dist_err < self.dist_tolerance:
                # If last target, enforce tighter final tolerance before finishing
                if idx + 1 == total:
                    if dist_err < self.final_distance_tol:
                        # Pause at final checkpoint (convert ms -> s)
                        self._pause_at_checkpoint(delay_ms / 1000.0, stop_event=stop_event)
                        print("REACHED END!")
                        idx += 1
                        continue
                else:
                    print(f"Reached target {idx+1}/{total} at ({x:.2f}, {y:.2f}).")
                    # Pause at intermediate checkpoint (convert ms -> s)
                    self._pause_at_checkpoint(delay_ms / 1000.0, stop_event=stop_event)
                    # Reset heading controller terms before next leg
                    self.integral_ang = 0.0
                    self.prev_ang_err = 0.0
                    idx += 1
                    continue

            # Angular control & gating for forward motion
            ang_err = self.normalize(heading - theta)
            dir_mult = 1
            move_flag = 1 if (abs(math.degrees(ang_err)) <= self.ang_tolerance and abs(ang_err) < math.pi/2) else 0

            self.integral_ang += ang_err * dt
            deriv_ang = (ang_err - self.prev_ang_err) / dt
            self.prev_ang_err = ang_err

            ang_ctrl = (
                self.Kp_ang * ang_err +
                self.Ki_ang * self.integral_ang +
                self.Kd_ang * deriv_ang
            )
            lin_ctrl = max(-max_lin, min(max_lin, self.Kp_dist * dist_err))

            left = base_speed + (move_flag * dir_mult * lin_ctrl) - ang_ctrl
            right = base_speed + (move_flag * dir_mult * lin_ctrl) + ang_ctrl

            left = max(70, min(110, self.adjust_speed(left)))
            right = max(70, min(110, self.adjust_speed(right)))

            self.iface.write_command(left, right)
            self.iface.log_error(dist_err, ang_err)

        self.iface.write_command(90, 90)
        print("Robot navigation completed.")


def run_controller(
    target_file, pose_file, command_file, error_file,
    Kp_dist=0.2, Kp_ang=4.0, Ki_ang=0.11, Kd_ang=-0.001,
    dist_tolerance=15, ang_tolerance=12, final_distance_tol=10, stop_event=None
):
    """
    Convenience function to run the PID controller using text file paths.
    """
    iface = FileInterface(target_file, pose_file, command_file, error_file)
    controller = PIDController(
        iface,
        Kp_dist=Kp_dist, Kp_ang=Kp_ang,
        Ki_ang=Ki_ang, Kd_ang=Kd_ang,
        dist_tolerance=dist_tolerance,
        ang_tolerance=ang_tolerance, final_distance_tol=final_distance_tol
    )
    controller.run(stop_event=stop_event)


def exec_bot():
    target_file  = str(Path("Targets") / "path.txt")
    pose_file    = str(Path("Data") / "robot_pos.txt")
    command_file = str(Path("Data") / "command.txt")
    error_file   = str(Path("Data") / "error.txt")

    print("Starting demo run of PID controller…")
    try:
        run_controller(
            target_file,
            pose_file,
            command_file,
            error_file
        )
    except Exception as e:
        print(f"An error occurred during the controller run: {e}")
    finally:
        print(f"Demo run completed. Files used:\n"
              f"  Targets:  {target_file}\n"
              f"  Pose:     {pose_file}\n"
              f"  Command:  {command_file}\n"
              f"  Error:    {error_file}")

def exec_bot_with_thread(stop_event):
    target_file  = str(Path("Targets") / "path.txt")
    pose_file    = str(Path("Data") / "robot_pos.txt")
    command_file = str(Path("Data") / "command.txt")
    error_file   = str(Path("Data") / "error.txt")

    print("Starting demo run of PID controller…")
    try:
        run_controller(
            target_file,
            pose_file,
            command_file,
            error_file,
            stop_event=stop_event
        )
    except Exception as e:
        print(f"An error occurred during the controller run: {e}")
    finally:
        print(f"Demo run completed. Files used:\n"
              f"  Targets:  {target_file}\n"
              f"  Pose:     {pose_file}\n"
              f"  Command:  {command_file}\n"
              f"  Error:    {error_file}")


if __name__ == "__main__":
    exec_bot()