#!/usr/bin/env python3
import serial
import time
import json
from typing import Optional
import threading

def _send_command(ser: serial.Serial, left_val: int, right_val: int):
    data_to_send = {"left": left_val, "right": right_val}
    ser.write(json.dumps(data_to_send).encode('utf-8') + b'\n')

def stop_robot(serial_port: str, baud_rate: int):
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            _send_command(ser, 90, 90)
    except Exception as e:
        print(f"[stop_robot] failed: {e}")

def move_robot(
    serial_port: str,
    baud_rate: int,
    command_file: str,
    stop_event: threading.Event,
    send_interval_s: float = 0.1
):
    """
    Continuously reads one line from command_file and sends it to the robot,
    until stop_event is set (or an error occurs).
    Any exception here will bubble out.
    """
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    try:
        while True:#ot stop_event.is_set():
            with open(command_file, 'r') as f:
                line = f.readline().strip()
            # print("EXECFUTING::: ",line)
            if line:
                left_str, right_str = line.split(',')
                left_speed = int(left_str)
                right_speed = int(right_str)
                # your transform
                left_speed = 180 - left_speed
                _send_command(ser, left_speed, right_speed)

            time.sleep(send_interval_s)
    finally:
        # always send a final stop
        try:
            _send_command(ser, 90, 90)
        except:
            pass
        ser.close()

def move_robot_with_thread(
    serial_port: str,
    baud_rate: int,
    command_file: str,
    stop_event: threading.Event,
    send_interval_s: float = 0.1
):
    """
    Continuously reads one line from command_file and sends it to the robot,
    until stop_event is set (or an error occurs).
    Any exception here will bubble out.
    """
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    try:
        while not stop_event.is_set():
            with open(command_file, 'r') as f:
                line = f.readline().strip()
            # print("EXECFUTING::: ",line)
            if line:
                left_str, right_str = line.split(',')
                left_speed = int(left_str)
                right_speed = int(right_str)
                # your transform
                left_speed = 180 - left_speed
                _send_command(ser, left_speed, right_speed)

            time.sleep(send_interval_s)
    finally:
        # always send a final stop
        try:
            _send_command(ser, 90, 90)
        except:
            pass
        ser.close()


if __name__ == "__main__":
    SERIAL_PORT = '/dev/ttyACM0'
    BAUD_RATE = 115200
    INPUT_FILE = 'Data/command.txt'
    SEND_INTERVAL_S = 0.1

    move_robot(SERIAL_PORT, BAUD_RATE, INPUT_FILE, SEND_INTERVAL_S)
