#!/usr/bin/env python3
import serial
import time
import json
import threading
from typing import Optional

def _send_command(ser: serial.Serial, robot_id: int, right_val: int, left_val: int):
    """
    Send a single JSON command line: {"id": <robot_id>, "left": <val>, "right": <val>}
    """
    data_to_send = {"id": int(robot_id), "left": int(left_val), "right": int(right_val)}
    ser.write(json.dumps(data_to_send).encode('utf-8') + b'\n')

def stop_robot(serial_port: str, baud_rate: int, robot_id: int):
    """
    Send a single 'stop' command (90,90) and close.
    """
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            _send_command(ser, robot_id, 90, 90)
    except Exception as e:
        print(f"[stop_robot] failed: {e}")

def move_robot(
    robot_id: int,
    serial_port: str,
    baud_rate: int,
    command_file: str,
    stop_event: threading.Event,
    send_interval_s: float = 0.1
):
    """
    Continuously reads one line from command_file and sends it to the robot,
    until stop_event is set (or an error occurs).
    """
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    try:
        while not stop_event.is_set():
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

                    _send_command(ser, robot_id, left_out, right_out)
                except Exception as e:
                    print(f"[move_robot] bad line '{line}': {e}")

            time.sleep(send_interval_s)
    finally:
        # Always send a final stop
        try:
            _send_command(ser, robot_id, 90, 90)
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
):
    """
    Spawn a daemon thread that runs move_robot(...) with the shared stop_event.
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
        ),
    )
    t.start()
    return t

if __name__ == "__main__":
    # Quick manual test
    SERIAL_PORT = '/dev/ttyACM0'
    BAUD_RATE = 115200
    INPUT_FILE = 'Data/command.txt'
    SEND_INTERVAL_S = 0.1
    ROBOT_ID = 782

    evt = threading.Event()
    try:
        move_robot(ROBOT_ID, SERIAL_PORT, BAUD_RATE, INPUT_FILE, evt, SEND_INTERVAL_S)
    except KeyboardInterrupt:
        evt.set()
