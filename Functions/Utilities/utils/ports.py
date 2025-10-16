# vision_dashboard/utils/ports.py

import serial.tools.list_ports
import cv2

def refresh_serial_ports() -> list[str]:
    """Scans for and returns a list of available serial port names."""
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports] if ports else ["No Ports Found"]

def refresh_cameras(max_index=10) -> list[str]:
    """Returns a list of available camera indices as strings."""
    return [str(i) for i in range(max_index) if cv2.VideoCapture(i).isOpened()]