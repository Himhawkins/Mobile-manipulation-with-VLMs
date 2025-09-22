#!/usr/bin/env python3
import serial
import time
import json
import threading
import queue
from typing import Optional

# --------------------- Low-level send helpers ---------------------

def _send_wheel_command(ser: serial.Serial, robot_id: int, right_val: int, left_val: int):
    """
    Send a single JSON command line for wheels:
      {"id": <robot_id>, "left": <val>, "right": <val>}
    """
    data_to_send = {"id": int(robot_id), "left": int(left_val), "right": int(right_val)}
    ser.write(json.dumps(data_to_send).encode('utf-8') + b'\n')

def _send_gripper_command(ser: serial.Serial, robot_id: int, state: str):
    """
    Send a single JSON command line for gripper:
      {"id": <robot_id>, "gripper": "open" | "close"}
    """
    state = state.lower().strip()
    if state not in ("open", "close"):
        raise ValueError(f"Invalid gripper state: {state!r}. Use 'open' or 'close'.")
    data_to_send = {"id": int(robot_id), "gripper": state}
    ser.write(json.dumps(data_to_send).encode('utf-8') + b'\n')

# --------------------- One-shot helpers ---------------------

def stop_robot(serial_port: str, baud_rate: int, robot_id: int):
    """
    Send a single 'stop' command (90,90) and close.
    """
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            _send_wheel_command(ser, robot_id, 90, 90)
    except Exception as e:
        print(f"[stop_robot] failed: {e}")

def send_gripper_once(serial_port: str, baud_rate: int, robot_id: int, state: str):
    """
    Open port briefly and send a single gripper command: 'open' or 'close'.
    """
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            _send_gripper_command(ser, robot_id, state)
    except Exception as e:
        print(f"[send_gripper_once] failed: {e}")

def gripper_open(serial_port: str, baud_rate: int, robot_id: int):
    send_gripper_once(serial_port, baud_rate, robot_id, "open")

def gripper_close(serial_port: str, baud_rate: int, robot_id: int):
    send_gripper_once(serial_port, baud_rate, robot_id, "close")

# --------------------- Streaming thread ---------------------

def move_robot(
    robot_id: int,
    serial_port: str,
    baud_rate: int,
    command_file: str,
    stop_event: threading.Event,
    send_interval_s: float = 0.1,
    gripper_queue: Optional["queue.Queue[str]"] = None,
):
    """
    Continuously:
      1) reads one line from command_file (x,y wheel command as 'left,right'),
      2) sends wheel command to the robot,
      3) checks if a gripper action is queued and sends it immediately,
    until stop_event is set (or an error occurs).

    - Wheel JSON:   {"id": <id>, "left": <val>, "right": <val>}
    - Gripper JSON: {"id": <id>, "gripper": "open"|"close"}

    NOTE: If you pass a gripper_queue, push strings "open" or "close" into it
          from elsewhere (e.g., controller) to actuate the gripper in real time.
    """
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    try:
        while not stop_event.is_set():
            # --- Wheel command from file ---
            try:
                with open(command_file, 'r') as f:
                    line = f.readline().strip()
            except Exception as e:
                print(f"[move_robot] read '{command_file}' failed: {e}")
                time.sleep(send_interval_s)
                continue

            if line:
                try:
                    left_str, right_str = line.split(',')
                    left_speed = int(left_str)
                    right_speed = int(right_str)

                    # Hardware-specific transform (example: invert LEFT only)
                    left_out = 180 - left_speed
                    right_out = right_speed

                    _send_wheel_command(ser, robot_id, left_out, right_out)
                except Exception as e:
                    print(f"[move_robot] bad line '{line}': {e}")

            # --- Gripper command (optional queue) ---
            if gripper_queue is not None:
                try:
                    # Drain all currently queued gripper actions to avoid lag
                    while True:
                        state = gripper_queue.get_nowait()
                        try:
                            _send_gripper_command(ser, robot_id, state)
                        except Exception as ge:
                            print(f"[move_robot] gripper '{state}' failed: {ge}")
                except queue.Empty:
                    pass

            time.sleep(send_interval_s)
    finally:
        # Always send a final stop
        try:
            _send_wheel_command(ser, robot_id, 90, 90)
        except Exception:
            pass
        ser.close()

def start_move_robot_thread(
    robot_id: int,
    serial_port: str,
    baud_rate: int,
    command_file: str,
    stop_event: threading.Event,
    send_interval_s: float = 0.1,
    gripper_queue: Optional["queue.Queue[str]"] = None,
):
    """
    Spawn a daemon thread that runs move_robot(...) with the shared stop_event.
    If you pass gripper_queue, push "open"/"close" into it from other code.
    Returns the Thread object.
    """
    t = threading.Thread(
        target=move_robot,
        name=f"serial-{robot_id}",
        daemon=True,
        kwargs=dict(
            robot_id=int(robot_id),
            serial_port=serial_port,
            baud_rate=int(baud_rate),
            command_file=command_file,
            stop_event=stop_event,
            send_interval_s=float(send_interval_s),
            gripper_queue=gripper_queue,
        ),
    )
    t.start()
    return t

# --------------------- Quick manual test ---------------------

if __name__ == "__main__":
    SERIAL_PORT = '/dev/ttyACM0'
    BAUD_RATE = 115200
    INPUT_FILE = 'Data/command.txt'
    SEND_INTERVAL_S = 0.1
    ROBOT_ID = 782

    # Example usage:
    # - The wheel stream runs in the foreground here.
    # - Press Ctrl+C to stop; it will send a final (90,90).
    # - While it runs, uncomment the gripper_open/close calls below to test.

    evt = threading.Event()
    try:
        # Example: send a quick open/close once (uncomment to try)
        # gripper_open(SERIAL_PORT, BAUD_RATE, ROBOT_ID)
        # time.sleep(0.5)
        # gripper_close(SERIAL_PORT, BAUD_RATE, ROBOT_ID)

        move_robot(ROBOT_ID, SERIAL_PORT, BAUD_RATE, INPUT_FILE, evt, SEND_INTERVAL_S)
    except KeyboardInterrupt:
        evt.set()
