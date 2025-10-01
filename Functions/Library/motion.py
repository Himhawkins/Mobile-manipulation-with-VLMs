#!/usr/bin/env python3
import sys
import serial
import time
import json
import threading
from typing import List, Tuple, Optional

# --------------------- Global Thread Management ---------------------
_motion_threads = {}  # { serial_port: {"thread": Thread, "stop_event": Event} }
_motion_threads_lock = threading.Lock()

# --------------------- Low-level send helpers ---------------------
def _send_command(
    ser: serial.Serial,
    robot_id: int,
    left_val: int,
    right_val: int,
    gripper_state: str | None = None,
):
    """
    Send one JSON line:
      {"id": <robot_id>, "left": <val>, "right": <val>[, "gripper": "open"|"close"]}
    """
    payload = {
        "id": int(robot_id),
        "left": int(left_val),
        "right": int(right_val),
    }

    if gripper_state is not None:
        s = str(gripper_state).strip().lower()
        if s not in ("open", "close"):
            raise ValueError(f"Invalid gripper state: {gripper_state!r}. Use 'open' or 'close'.")
        payload["gripper"] = s

    ser.write(json.dumps(payload).encode("utf-8") + b"\n")

# --------------------- Streaming function (now dual-use) ---------------------

def move_robots(
    serial_port: str,
    baud_rate: int,
    stop_event: Optional[threading.Event] = None,
    command_file: str = "Data/command.json",
    send_interval_s: float = 0.1,
):
    """
    Continuously sends robot commands from a JSON file.

    - If 'stop_event' is provided (when run in a thread), the loop will
      terminate gracefully when the event is set.
    - If 'stop_event' is None (when run directly), it will loop until
      interrupted (e.g., with Ctrl+C).
    """
    ser = None
    last_known_robot_ids: List[int] = []
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        print(f"[{serial_port}] Opened serial port at {baud_rate} baud.")

        while True:
            if stop_event and stop_event.is_set():
                break

            try:
                with open(command_file, 'r') as f:
                    data = json.load(f)
                robot_commands = data.get("robots", [])
                
                current_ids = [cmd['id'] for cmd in robot_commands if 'id' in cmd]
                if current_ids:
                    last_known_robot_ids = current_ids

            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"[{serial_port}] [ERROR] Could not read or parse '{command_file}': {e}", file=sys.stderr)
                time.sleep(send_interval_s)
                continue

            if not robot_commands:
                print(f"[{serial_port}] [WARNING] No 'robots' found in '{command_file}'.")

            for robot_cmd in robot_commands:
                try:
                    robot_id = robot_cmd["id"]
                    left_speed = robot_cmd["left"]
                    right_speed = robot_cmd["right"]
                    gripper_state = robot_cmd.get("gripper", "open")

                    left_out = 180 - left_speed
                    right_out = right_speed

                    _send_command(ser, robot_id, left_out, right_out, gripper_state)

                except (KeyError, TypeError) as e:
                    print(f"[{serial_port}] [ERROR] Invalid command format: {robot_cmd}. Missing key: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"[{serial_port}] [ERROR] Failed to send command for robot {robot_cmd.get('id', 'N/A')}: {e}", file=sys.stderr)

            if stop_event:
                stop_event.wait(send_interval_s)
            else:
                time.sleep(send_interval_s)

    except serial.SerialException as e:
        print(f"[{serial_port}] [FATAL] Serial port error: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        print(f"\n[{serial_port}] Program interrupted by user.")
    except Exception as e:
        print(f"[{serial_port}] [FATAL] An unexpected error occurred: {e}", file=sys.stderr)
    finally:
        print(f"[{serial_port}] Motion loop has ended.")
        if ser and ser.is_open:
            if last_known_robot_ids:
                print(f"[{serial_port}] Sending STOP command to all known robots...")
                for robot_id in last_known_robot_ids:
                    try:
                        _send_command(ser, robot_id, 90, 90, "open")
                    except Exception as e:
                        print(f"[{serial_port}] [ERROR] Failed to send stop command to robot {robot_id}: {e}", file=sys.stderr)
                time.sleep(0.1)
            
            ser.close()
            print(f"[{serial_port}] Serial port closed.")

# --------------------- Thread Management Functions (Refactored) ---------------------

def start_motion_thread(
    serial_port: str,
    baud_rate: int,
    command_file: str = "Data/command.json"
) -> str:
    """
    Creates and starts the motion control thread for a specific serial port.
    Manages the thread in a global registry to prevent duplicates.
    Returns a status string.
    """
    with _motion_threads_lock:
        if serial_port in _motion_threads and _motion_threads[serial_port]['thread'].is_alive():
            return f"Motion thread is already running for port {serial_port}"

        print(f"Starting motion thread for port {serial_port}...")
        stop_event = threading.Event()
        motion_thread = threading.Thread(
            target=move_robots,
            args=(serial_port, baud_rate, stop_event, command_file),
            daemon=True
        )
        motion_thread.start()
        
        _motion_threads[serial_port] = {
            "thread": motion_thread,
            "stop_event": stop_event
        }
    return f"Started motion thread for port {serial_port}"

def stop_motion_thread(serial_port: str, join_timeout: float = 5.0) -> str:
    """
    Signals the motion control thread for a specific serial port to stop and waits for it to terminate.
    Returns a status string.
    """
    with _motion_threads_lock:
        info = _motion_threads.get(serial_port)
        if not info or not info['thread'].is_alive():
            return f"No running motion thread for port {serial_port}"
        
        print(f"Signaling motion thread on port {serial_port} to stop...")
        stop_event = info["stop_event"]
        thread = info["thread"]
        stop_event.set()

    # Release the lock before joining to avoid deadlocks
    thread.join(timeout=join_timeout)

    if thread.is_alive():
        return f"Stop requested; motion thread for port {serial_port} is still stopping"
    else:
        with _motion_threads_lock:
            # Check again in case it was stopped by other means
            if serial_port in _motion_threads:
                _motion_threads.pop(serial_port)
        print(f"Motion thread for port {serial_port} has been stopped.")
        return f"Stopped motion thread for port {serial_port}"

# --------------------- Main Execution Example ---------------------
if __name__ == "__main__":
    # --- Configuration ---
    SERIAL_PORT = '/dev/ttyACM0' # <-- IMPORTANT: CHANGE THIS TO YOUR SERIAL PORT
    BAUD_RATE = 1152_00
    COMMAND_FILE = "Data/command.json"

    # The main block now uses the simplified start/stop functions
    print("--- Running motion manager ---")
    try:
        status = start_motion_thread(SERIAL_PORT, BAUD_RATE, COMMAND_FILE)
        print(status)
        
        # If the thread started successfully, the main program can do other things.
        if "Started" in status:
            print("Main program is running. The motion thread is active in the background.")
            print("This will run for 10 seconds or until Ctrl+C is pressed.")
            time.sleep(10)

    except KeyboardInterrupt:
        print("\nCtrl+C detected in main program. Shutting down.")
    except Exception as e:
        print(f"An error occurred in the main program: {e}")
    finally:
        # This block ensures the thread is always stopped cleanly
        print(stop_motion_thread(SERIAL_PORT))
        print("Main program has finished.")