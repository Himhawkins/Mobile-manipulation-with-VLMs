import os
import json
import cv2
import serial.tools.list_ports

def refresh_cameras(max_index=15, current_index=None):
    """
    Returns a list of camera labels like ['Camera 0', 'Camera 2'],
    including the currently active camera even if it appears busy.
    """
    working = []

    for i in range(max_index):
        path = f"/dev/video{i}"
        if not os.path.exists(path):
            continue
        if i == current_index:
            # Current camera is already open and working
            working.append(f"Camera {i}")
            continue
        cap = cv2.VideoCapture(i)
        ret, _ = cap.read()
        cap.release()
        if ret:
            working.append(f"Camera {i}")

    return working

    return working

def refresh_serial_ports():
    """
    Returns a list of USB serial ports with 'ACM' in their device name.
    """
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports if 'ACM' in port.device]