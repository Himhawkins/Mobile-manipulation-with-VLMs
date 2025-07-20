import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import cv2
import serial.tools.list_ports
import threading
import time
from PIL import Image, ImageTk
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq
import os
from dotenv import load_dotenv

# Local imports from your project
from pipeline import process_arena_pattern, generate_and_save_warped_plot, overlay_points_on_image, overlay_points_n_lines_on_image
from controller import run_controller
from motion import move_robot
from arena_transform import detect_and_warp, detect_and_list, detect_and_read_alphabets, detect_home
from astar import PathPlanner
from marker_tracking import generate_marker_track
from marker_sequence_parser import parse_marker_sequence

# Load environment variables (e.g., for API keys)
load_dotenv()

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

IMG_PATH = 'arena_img_test3.png'

class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, master, app_instance):
        super().__init__(master)
        self.title("Settings")
        self.geometry("600x300")
        self.app_instance = app_instance

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)

        ctk.CTkLabel(self, text="Pattern Generation Settings", font=("Arial", 18, "bold")).grid(row=0, column=0, columnspan=2, pady=15)

        def create_setting_row(row_num, label_text, initial_value):
            ctk.CTkLabel(self, text=label_text).grid(row=row_num, column=0, padx=15, pady=5, sticky="w")
            entry = ctk.CTkEntry(self)
            entry.grid(row=row_num, column=1, padx=15, pady=5, sticky="ew")
            entry.insert(0, initial_value)
            return entry

        self.warp_width_entry = create_setting_row(1, "Warp Width:", str(self.app_instance.warp_dims[0]))
        self.warp_height_entry = create_setting_row(2, "Warp Height:", str(self.app_instance.warp_dims[1]))
        self.marker_prompt_entry = create_setting_row(3, "Marker Prompt:", self.app_instance.marker_prompt)
        self.obstacles_prompt_entry = create_setting_row(4, "Obstacles Prompt:", self.app_instance.obstacles_prompt)

        save_button = ctk.CTkButton(self, text="Save Settings", command=self.save_settings, font=("Arial", 16, "bold"), height=40)
        save_button.grid(row=5, column=0, columnspan=2, pady=25)

    def save_settings(self):
        try:
            new_warp_width = int(self.warp_width_entry.get())
            new_warp_height = int(self.warp_height_entry.get())
            self.app_instance.warp_dims = (new_warp_width, new_warp_height)
        except ValueError:
            print("Invalid warp dimensions. Please enter integers.")
            messagebox.showerror("Input Error", "Warp dimensions must be integers.")
            return

        self.app_instance.marker_prompt = self.marker_prompt_entry.get()
        self.app_instance.obstacles_prompt = self.obstacles_prompt_entry.get()
        self.app_instance.save_settings_to_file()
        print("Settings saved successfully!")
        self.destroy()

class CameraApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Advanced Camera Tracking App")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.camera_index = 0
        self.serial_port = None
        self.baud_rate = 115200
        self.cap = None
        self.video_thread = None
        self.stop_video_thread = False

        self.camera_image_tk = None
        self.preview1_image_tk = None
        self.preview2_image_tk = None

        self.fig1 = None
        self.canvas1_widget = None

        self.warp_dims = (800, 800)
        self.raw_json_path = "targets/pattern_llm/warped.txt"
        self.unwarp_json_path = "targets/pattern_llm/targets.txt"
        self.warped_plot_path = "targets/pattern_llm/warped_plot.jpg"
        self.target_plot_path = "targets/pattern_llm/targets_plot.jpg"
        self.pose_json_path = "controls/pose.txt"

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.target_marker_id = 782

        # --- Arena Calibration Attributes ---
        self.arena_corners = None
        self.is_calibrated = False
        self.perspective_matrix = None

        self.last_robot_angle = 0.0 # Stores the last detected robot angle in radians

        self.camera_matrix = np.array([[1000, 0, 640],
                                       [0, 1000, 360],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

        self.marker_length = 0.09
        self.object_points = np.array([
            [-self.marker_length / 2, self.marker_length / 2, 0],
            [self.marker_length / 2, self.marker_length / 2, 0],
            [self.marker_length / 2, -self.marker_length / 2, 0],
            [-self.marker_length / 2, -self.marker_length / 2, 0]
        ], dtype=np.float32)

        self.marker_prompt = "Blue Circles"
        self.obstacles_prompt = "Black Rectangles"
        self.alphabet_centroids   = []


        self.settings_path = "settings/settings.json"
        self.load_settings()

        self.create_welcome_page()

        self.is_executing_pattern = False
        self.point_track_save_path = "targets/point_track/targets.txt"
        self.tracked_points = []
        # point-track preview paths
        self.point_targets_plot_path = "targets/point_track/targets_plot.jpg"
        self.point_warped_plot_path  = "targets/point_track/warped_plot.jpg"

        # spline-track preview paths (reuse same previews or change as needed)
        self.spline_targets_plot_path = "targets/spline_track/targets_plot.jpg"
        self.spline_warped_plot_path  = "targets/spline_track/warped_plot.jpg"
        self.is_tracking = False
        self.bread_crumb_path = ""

    def load_settings(self):
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r') as f:
                    settings = json.load(f)
                    if "warp_width" in settings and "warp_height" in settings:
                        self.warp_dims = (settings["warp_width"], settings["warp_height"])
                    if "marker_prompt" in settings:
                        self.marker_prompt = settings["marker_prompt"]
                    if "obstacles_prompt" in settings:
                        self.obstacles_prompt = settings["obstacles_prompt"]
                print("Settings loaded successfully.")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading settings from {self.settings_path}: {e}. Using default settings.")
        else:
            print(f"Settings file not found at {self.settings_path}. Using default settings.")
    
    def _read_targets_with_header(self, header_path):
        idx = 0
        targets = []
        if not os.path.exists(header_path):
            return idx, targets

        with open(header_path, 'r') as f:
            first = f.readline().strip()
            if first.startswith('(') and '/' in first:
                try:
                    idx_str, _ = first.strip('()').split('/')
                    idx = int(idx_str)
                except ValueError:
                    idx = 0
            else:
                f.seek(0)

            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                try:
                    x, y, th = map(float, parts)
                    targets.append((x, y, th))
                except ValueError:
                    continue

        return idx, targets

    def save_settings_to_file(self):
        settings_dir = os.path.dirname(self.settings_path)
        os.makedirs(settings_dir, exist_ok=True)

        settings_data = {
            "warp_width": self.warp_dims[0],
            "warp_height": self.warp_dims[1],
            "marker_prompt": self.marker_prompt,
            "obstacles_prompt": self.obstacles_prompt
        }
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(settings_data, f, indent=4)
            print(f"Settings saved to {self.settings_path}")
        except Exception as e:
            print(f"Error saving settings to {self.settings_path}: {e}")

    def create_welcome_page(self):
        for widget in self.winfo_children():
            widget.destroy()

        self.welcome_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#2b2b2b")
        self.welcome_frame.grid(row=0, column=0, sticky="nsew", padx=80, pady=80)
        
        self.welcome_frame.grid_columnconfigure(0, weight=1)
        self.welcome_frame.grid_columnconfigure(1, weight=0)
        self.welcome_frame.grid_columnconfigure(2, weight=0)
        self.welcome_frame.grid_columnconfigure(3, weight=0)
        self.welcome_frame.grid_columnconfigure(4, weight=1)
        
        for i in range(10):
            self.welcome_frame.grid_rowconfigure(i, weight=1)

        ctk.CTkLabel(self.welcome_frame, text="Advanced Camera Tracking App", font=("Arial", 36, "bold"), text_color="#00aaff").grid(row=1, column=1, columnspan=3, pady=(20, 40), sticky="n")

        camera_label = ctk.CTkLabel(self.welcome_frame, text="Select Camera:", font=("Arial", 18))
        camera_label.grid(row=3, column=1, pady=(10, 5), sticky="e", padx=(0, 10))
        
        self.camera_options = self.get_available_cameras()
        self.camera_var = ctk.StringVar(value=self.camera_options[0] if self.camera_options else "No Cameras Found")
        self.camera_dropdown = ctk.CTkOptionMenu(self.welcome_frame, variable=self.camera_var, values=self.camera_options, width=250, height=40, font=("Arial", 16))
        self.camera_dropdown.grid(row=3, column=2, pady=(10, 5), sticky="w")
        
        camera_refresh_button = ctk.CTkButton(self.welcome_frame, text="Refresh", command=self.refresh_cameras, width=100, height=40, font=("Arial", 14), corner_radius=8)
        camera_refresh_button.grid(row=3, column=3, pady=(10, 5), sticky="w", padx=(10, 0))

        port_label = ctk.CTkLabel(self.welcome_frame, text="Select ESP32 Port:", font=("Arial", 18))
        port_label.grid(row=4, column=1, pady=(10, 5), sticky="e", padx=(0, 10))
        
        self.port_options = self.get_available_ports()
        self.port_var = ctk.StringVar(value=self.port_options[0] if self.port_options else "No Ports Found")
        self.port_dropdown = ctk.CTkOptionMenu(self.welcome_frame, variable=self.port_var, values=self.port_options, width=250, height=40, font=("Arial", 16))
        self.port_dropdown.grid(row=4, column=2, pady=(10, 5), sticky="w")
        
        port_refresh_button = ctk.CTkButton(self.welcome_frame, text="Refresh", command=self.refresh_ports, width=100, height=40, font=("Arial", 14), corner_radius=8)
        port_refresh_button.grid(row=4, column=3, pady=(10, 5), sticky="w", padx=(10, 0))

        baud_label = ctk.CTkLabel(self.welcome_frame, text="Select Baud Rate:", font=("Arial", 18))
        baud_label.grid(row=5, column=1, pady=(10, 20), sticky="e", padx=(0, 10))

        self.baud_rate_options = ["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"]
        if str(self.baud_rate) not in self.baud_rate_options:
            self.baud_rate_options.append(str(self.baud_rate))
            self.baud_rate_options.sort(key=int)

        self.baud_rate_var = ctk.StringVar(value=str(self.baud_rate))
        self.baud_rate_dropdown = ctk.CTkOptionMenu(self.welcome_frame, variable=self.baud_rate_var, values=self.baud_rate_options, width=250, height=40, font=("Arial", 16))
        self.baud_rate_dropdown.grid(row=5, column=2, pady=(10, 20), sticky="w")

        start_button = ctk.CTkButton(self.welcome_frame, text="Start Dashboard", command=self.show_dashboard,
                                     font=("Arial", 22, "bold"), height=60, corner_radius=15, width=300, fg_color="#0088cc", hover_color="#0066aa")
        start_button.grid(row=7, column=1, columnspan=3, pady=30, sticky="n")

        settings_button = ctk.CTkButton(self.welcome_frame, text="Settings", command=self.open_settings_window,
                                        font=("Arial", 18), height=45, corner_radius=12, width=200, fg_color="#555555", hover_color="#777777")
        settings_button.grid(row=8, column=1, columnspan=3, pady=(10, 20), sticky="n")

    def get_available_cameras(self):
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(f"Camera {i}")
                cap.release()
        return available_cameras if available_cameras else ["No Cameras Found"]

    def refresh_cameras(self):
        self.camera_options = self.get_available_cameras()
        self.camera_dropdown.configure(values=self.camera_options)
        self.camera_var.set(self.camera_options[0] if self.camera_options else "No Cameras Found")
        print("Cameras refreshed.")

    def get_available_ports(self):
        ports = serial.tools.list_ports.comports()
        available_ports = []
        for port in ports:
            if '/dev/ttyACM' in port.device:
                display_name = port.device.replace('/dev/tty', '')
                available_ports.append(display_name)
        return available_ports if available_ports else ["No Ports Found"]

    def refresh_ports(self):
        self.port_options = self.get_available_ports()
        self.port_dropdown.configure(values=self.port_options)
        self.port_var.set(self.port_options[0] if self.port_options else "No Ports Found")
        print("Ports refreshed.")

    def open_settings_window(self):
        if hasattr(self, 'settings_window') and self.settings_window.winfo_exists():
            self.settings_window.lift()
        else:
            self.settings_window = SettingsWindow(self, self)
            self.settings_window.update_idletasks()
            self.settings_window.grab_set()

    def show_dashboard(self):
        selected_camera_str = self.camera_var.get()
        if "Camera" in selected_camera_str:
            self.camera_index = int(selected_camera_str.split(" ")[1])
        else:
            self.camera_index = -1

        self.serial_port = self.port_var.get()
        self.selected_baud_rate = int(self.baud_rate_var.get())

        self.welcome_frame.destroy()

        self.dashboard_frame = ctk.CTkFrame(self, corner_radius=15)
        self.dashboard_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.dashboard_frame.grid_rowconfigure(0, weight=8)
        self.dashboard_frame.grid_rowconfigure(1, weight=2)
        self.dashboard_frame.grid_columnconfigure(0, weight=7)
        self.dashboard_frame.grid_columnconfigure(1, weight=3)

        top_frame = ctk.CTkFrame(self.dashboard_frame, corner_radius=15)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        top_frame.grid_rowconfigure(0, weight=1)
        top_frame.grid_columnconfigure(0, weight=7)
        top_frame.grid_columnconfigure(1, weight=3)

        self.main_camera_frame = ctk.CTkFrame(top_frame, corner_radius=10, fg_color="#333333")
        self.main_camera_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.main_camera_frame.grid_rowconfigure(0, weight=1)
        self.main_camera_frame.grid_columnconfigure(0, weight=1)
        self.camera_label = ctk.CTkLabel(self.main_camera_frame, text="", bg_color="#333333")
        self.camera_label.grid(row=0, column=0, sticky="nsew")

        self.preview_frame = ctk.CTkFrame(top_frame, corner_radius=10, fg_color="#444444")
        self.preview_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_rowconfigure(1, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)

        self.preview_sub_window1 = ctk.CTkFrame(self.preview_frame, corner_radius=8, fg_color="#555555")
        self.preview_sub_window1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.preview_sub_window1.grid_rowconfigure(0, weight=1)
        self.preview_sub_window1.grid_columnconfigure(0, weight=1)
        self.preview_label1 = ctk.CTkLabel(self.preview_sub_window1, text="Preview 1", bg_color="#555555")
        self.preview_label1.grid(row=0, column=0, sticky="nsew")


        self.preview_sub_window2 = ctk.CTkFrame(self.preview_frame, corner_radius=8, fg_color="#555555")
        self.preview_sub_window2.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.preview_sub_window2.grid_rowconfigure(0, weight=1)
        self.preview_sub_window2.grid_columnconfigure(0, weight=1)
        self.preview_label2 = ctk.CTkLabel(self.preview_sub_window2, text="Preview 2", bg_color="#555555")
        self.preview_label2.grid(row=0, column=0, sticky="nsew")

        self.control_frame = ctk.CTkFrame(self.dashboard_frame, corner_radius=15, fg_color="#222222")
        self.control_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        self.control_frame.grid_rowconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(1, weight=3)

        mode_label = ctk.CTkLabel(self.control_frame, text="Select Mode:", font=("Arial", 16))
        mode_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.mode_options = ["Pattern Gen", "Point Track", "Spline Track", "Marker Track"]
        self.mode_var = ctk.StringVar(value=self.mode_options[0])
        self.mode_dropdown = ctk.CTkOptionMenu(self.control_frame, variable=self.mode_var,
                                               values=self.mode_options, command=self.update_control_panel)
        self.mode_dropdown.grid(row=0, column=0, padx=(120, 10), pady=10, sticky="w")

        self.dynamic_content_frame = ctk.CTkFrame(self.control_frame, corner_radius=10, fg_color="transparent")
        self.dynamic_content_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.dynamic_content_frame.grid_rowconfigure(0, weight=1)

        if self.camera_index != -1:
            self.start_camera_feed()
        else:
            self.camera_label.configure(text="No live camera feed (select camera on welcome page)", font=("Arial", 18), text_color="red")
            self.camera_label.configure(image=None)
            self.camera_label.image = None
            self.preview_label1.configure(image=None)
            self.preview_label1.image = None
            self.preview_label2.configure(image=None)
            self.preview_label2.image = None

        self.update_control_panel(self.mode_var.get())
    
    def detect_aruco_bot(self, frame_to_draw_on, aruco_id):
        if self.ids is not None:
            for i, marker_id in enumerate(self.ids.flatten()):
                if marker_id == aruco_id:
                    pts = self.corners[i][0]
                    cx = int(pts[:, 0].mean())
                    cy = int(pts[:, 1].mean())
                    top_left, top_right = pts[0], pts[1]
                    angle_rad = np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0]) + (np.pi / 2)
                    
                    self.last_robot_angle = angle_rad

                    try:
                        with open(self.pose_json_path, "w") as f:
                            f.write(f"{cx},{cy},{angle_rad}")
                    except IOError as e:
                        print(f"Error writing to file {self.pose_json_path}: {e}")

                    cv2.polylines(frame_to_draw_on, [pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.circle(frame_to_draw_on, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(frame_to_draw_on, f"ID:{marker_id} ({cx},{cy})", (cx + 10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame_to_draw_on, f"Angle: {angle_rad:.2f} rad", (cx + 10, cy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (238, 244, 44), 2)
                    
    def start_camera_feed(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.camera_label.configure(text="Failed to open camera!", font=("Arial", 18), text_color="red")
            return

        self.stop_video_thread = False
        self.video_thread = threading.Thread(target=self.update_video_feed, daemon=True)
        self.video_thread.start()

    def update_video_feed(self):
        while not self.stop_video_thread:
            # ret, frame = self.cap.read()
            # TESTING ONLY - REMOVE LATER
            ret = True
            frame = cv2.imread(IMG_PATH)
            # TESTING ONLY - REMOVE LATER
            if not ret:
                print("Failed to grab frame")
                time.sleep(1)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.camera_index)
                continue
            
            frame_copy = frame.copy()

            # --- draw calibrated arena overlay ---
            if self.is_calibrated and self.arena_corners is not None:
                darkened = cv2.addWeighted(frame, 0.4, np.zeros(frame.shape, frame.dtype), 0, 0)
                mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.fillPoly(mask, [self.arena_corners], 255)
                bright = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)
                inv = cv2.bitwise_not(mask)
                dark_out = cv2.bitwise_and(darkened, darkened, mask=inv)
                frame = cv2.add(bright, dark_out)
            
            # --- darken obstacle regions ---
            if hasattr(self, "obstacles") and self.obstacles:
                # 1) build a mask of obstacle bboxes
                obs_mask = np.zeros(frame.shape[:2], dtype="uint8")
                for obs in self.obstacles:
                    x, y, w, h = obs["bbox"]
                    # fill each rect
                    cv2.rectangle(obs_mask, (x, y), (x + w, y + h), 255, -1)

                # 2) darken only those pixels
                dark_obs = cv2.addWeighted(frame, 0.4, np.zeros(frame.shape, frame.dtype), 0, 0)
                dark_region = cv2.bitwise_and(dark_obs, dark_obs, mask=obs_mask)
                keep_region = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(obs_mask))
                frame = cv2.add(dark_region, keep_region)

            # --- draw pattern‐gen points if any ---
            idx, targets = self._read_targets_with_header(self.bread_crumb_path)
            remaining = targets[idx:]
            if remaining:
                try:
                    pts = [{"x": x, "y": y, "theta": th} for x, y, th in remaining]
                    frame = overlay_points_on_image(frame, pts)
                except Exception as e:
                    print(f"Pattern overlay error: {e}")

            # --- draw user‐tracked points/lines if tracking ---
            if self.is_tracking and self.tracked_points:
                try:
                    pts = [{"x": x, "y": y, "theta": t} for x, y, t in self.tracked_points]
                    frame = overlay_points_n_lines_on_image(frame, pts)
                except Exception as e:
                    print(f"Tracking overlay error: {e}")

            # --- convert for marker detection & draw ArUco bot ---
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.corners, self.ids, _ = self.detector.detectMarkers(frame_copy)
            self.detect_aruco_bot(frame_to_draw_on=frame_rgb, aruco_id=self.target_marker_id)

            # --- display camera feed ---
            self.after(0, self._resize_and_update_label,
                    self.camera_label, frame_rgb, "camera_image_tk")

            # --- update side previews per mode ---
            mode = self.mode_var.get()
            if mode == "Pattern Gen":
                self.after(0, self.update_pattern_gen_previews)
            elif mode == "Point Track":
                self.after(0, self.update_point_track_previews)
            elif mode == "Spline Track":
                self.after(0, self.update_spline_track_previews)
            elif mode == "Marker Track":
                self.after(0, self.update_marker_track_previews)
            else:
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                self.after(0, self._resize_and_update_label,
                        self.preview_label1, gray, "preview1_image_tk")
                edges = cv2.Canny(gray, 100, 200)
                self.after(0, self._resize_and_update_label,
                        self.preview_label2, edges, "preview2_image_tk")

            time.sleep(0.01)

    def _resize_and_update_label(self, label, img_data, img_tk_attr):
        panel_width = label.winfo_width()
        panel_height = label.winfo_height()

        if panel_width > 1 and panel_height > 1:
            img_height, img_width = img_data.shape[:2]
            aspect_ratio = img_width / img_height
            if (panel_width / panel_height) > aspect_ratio:
                new_height = panel_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = panel_width
                new_height = int(new_width / aspect_ratio)
            
            if label is self.camera_label:
                # img_data.shape is (h, w, …)
                orig_h, orig_w = img_data.shape[:2]
                self.camera_display_size = (new_width, new_height)
                self.camera_orig_size    = (orig_w, orig_h)
            
            resized_frame = cv2.resize(img_data, (new_width, new_height), interpolation=cv2.INTER_AREA)

            if len(resized_frame.shape) == 2:
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)

            img_tk = ctk.CTkImage(light_image=Image.fromarray(resized_frame),
                                  dark_image=Image.fromarray(resized_frame),
                                  size=(new_width, new_height))
            label.configure(image=img_tk)
            setattr(self, img_tk_attr, img_tk)

    def update_control_panel(self, selected_mode):
        # If switching into Pattern Gen, clear any existing calibration
        if selected_mode == "Pattern Gen":
            self.is_calibrated = False
            self.arena_corners = None

        # Clear out any old controls
        for widget in self.dynamic_content_frame.winfo_children():
            widget.destroy()
        # Reset column weights
        for col_idx in range(4):
            self.dynamic_content_frame.grid_columnconfigure(col_idx, weight=0)

        # Ensure both preview labels exist in their sub‐windows
        for attr, subwin, text in [
            ("preview_label1", self.preview_sub_window1, "Preview 1"),
            ("preview_label2", self.preview_sub_window2, "Preview 2"),
        ]:
            lbl = getattr(self, attr, None)
            if lbl is None or not lbl.winfo_exists():
                new_lbl = ctk.CTkLabel(subwin, text=text, bg_color="#555555")
                new_lbl.grid(row=0, column=0, sticky="nsew")
                setattr(self, attr, new_lbl)

        # Clean up any existing matplotlib canvas
        if self.canvas1_widget and self.canvas1_widget.winfo_exists():
            self.canvas1_widget.destroy()
            self.canvas1_widget = None
            self.fig1 = None

        # Now branch by mode
        if selected_mode == "Pattern Gen":
            self.create_pattern_gen_widgets()
            self.update_pattern_gen_previews()

        elif selected_mode == "Point Track":
            self.create_point_track_widgets()
            self.update_point_track_previews()

        elif selected_mode == "Spline Track":
            self.create_spline_track_widgets()
            self.update_spline_track_previews()
        
        elif selected_mode == "Marker Track":
            self.create_marker_track_widgets()
            self.update_marker_track_previews()

    def create_pattern_gen_widgets(self):
        self.dynamic_content_frame.grid_columnconfigure(0, weight=0)
        self.dynamic_content_frame.grid_columnconfigure(1, weight=3)
        self.dynamic_content_frame.grid_columnconfigure((2,3), weight=1)

        prompt_label = ctk.CTkLabel(self.dynamic_content_frame, text="Pattern Prompt:", font=("Arial", 14))
        prompt_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.pattern_prompt_entry = ctk.CTkEntry(self.dynamic_content_frame, placeholder_text="Enter pattern description")
        self.pattern_prompt_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.generate_pattern_gen_button = ctk.CTkButton(self.dynamic_content_frame, text="Generate", command=self._start_pattern_generation_thread, corner_radius=8)
        self.generate_pattern_gen_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.execute_pattern_gen_button = ctk.CTkButton(self.dynamic_content_frame, text="Execute", command=self._start_execution_thread, corner_radius=8)
        self.execute_pattern_gen_button.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        if self.is_executing_pattern:
            self.generate_pattern_gen_button.configure(state="disabled")
            self.execute_pattern_gen_button.configure(state="disabled", text="Executing")
        else:
            self.generate_pattern_gen_button.configure(state="normal")
            self.execute_pattern_gen_button.configure(state="normal", text="Execute")

    def _start_pattern_generation_thread(self):
        self.generate_pattern_gen_button.configure(state="disabled")
        threading.Thread(target=self.generate_pattern, daemon=True).start()

    def generate_pattern(self):
        prompt = self.pattern_prompt_entry.get()
        
        if not prompt.strip():
            self.after(0, lambda: messagebox.showwarning("Input Required", "The pattern prompt cannot be empty. Please enter a description."))
            self.after(0, lambda: self.generate_pattern_gen_button.configure(state="normal"))
            return

        print(f"Generating pattern based on: {prompt}")

        current_frame = None
        if self.cap and self.cap.isOpened():
            # ret, current_frame = self.cap.read()
            # TESTING ONLY - REMOVE LATER
            ret = True
            current_frame = cv2.imread(IMG_PATH)
            # TESTING ONLY - REMOVE LATER
            if not ret:
                print("Failed to capture frame for pattern generation.")
                self.after(0, lambda: messagebox.showerror("Camera Error", "Failed to capture frame for pattern generation."))
                self.after(0, lambda: self.generate_pattern_gen_button.configure(state="normal"))
                return
        else:
            print("Camera not active, cannot generate pattern.")
            self.after(0, lambda: messagebox.showerror("Camera Error", "Camera not active. Please ensure camera is selected and running."))
            self.after(0, lambda: self.generate_pattern_gen_button.configure(state="normal"))
            return

        md_api_key = os.getenv("MOONDREAM_API_KEY")
        gm_api_key = os.getenv("GEMINI_API_KEY")

        warp_dims = self.warp_dims
        raw_json_path = self.raw_json_path
        unwarp_json_path = self.unwarp_json_path
        warped_plot_path = self.warped_plot_path
        target_plot_path = self.target_plot_path

        os.makedirs(os.path.dirname(raw_json_path), exist_ok=True)
        os.makedirs(os.path.dirname(unwarp_json_path), exist_ok=True)

        try:
            process_arena_pattern(
                image=current_frame,
                md_api_key=md_api_key,
                marker_prompt=self.marker_prompt,
                gm_api_key=gm_api_key,
                pattern_prompt=prompt,
                warp_dims=warp_dims,
                raw_txt_path=raw_json_path,
                unwarp_txt_path=unwarp_json_path,
                warped_plot_path=warped_plot_path,
                target_plot_path=target_plot_path
            )
            print("Pattern generation complete. Files saved.")
            self.after(0, self.update_pattern_gen_previews)
            self.after(0, lambda: messagebox.showinfo("Success", "Pattern generated successfully!"))
        except Exception as e:
            error_message = f"An error occurred during pattern generation:\n\n{e}"
            print(error_message)
            self.after(0, lambda: messagebox.showerror("Pattern Generation Failed", error_message))
        finally:
            self.after(0, lambda: self.generate_pattern_gen_button.configure(state="normal"))

    def update_pattern_gen_previews(self):
        # This method is now only for Pattern Gen mode, but we keep the name for clarity
        blank_img = np.zeros((self.warp_dims[1], self.warp_dims[0], 3), dtype=np.uint8)

        # Update Preview 1 with warped plot
        if os.path.exists(self.warped_plot_path):
            try:
                warped_plot_img = cv2.imread(self.warped_plot_path)
                if warped_plot_img is None:
                    raise FileNotFoundError(f"Could not load image: {self.warped_plot_path}")
                warped_plot_img_rgb = cv2.cvtColor(warped_plot_img, cv2.COLOR_BGR2RGB)
                self.after(0, self._resize_and_update_label, self.preview_label1, warped_plot_img_rgb, "preview1_image_tk")
            except Exception as e:
                print(f"Error loading or displaying {self.warped_plot_path}: {e}")
                self.after(0, self._resize_and_update_label, self.preview_label1, blank_img, "preview1_image_tk")
        else:
             self.after(0, self._resize_and_update_label, self.preview_label1, blank_img, "preview1_image_tk")
        
        # Update Preview 2 with target plot
        if os.path.exists(self.target_plot_path):
            try:
                target_plot_img = cv2.imread(self.target_plot_path)
                if target_plot_img is None:
                    raise FileNotFoundError(f"Could not load image: {self.target_plot_path}")
                target_plot_img_rgb = cv2.cvtColor(target_plot_img, cv2.COLOR_BGR2RGB)
                self.after(0, self._resize_and_update_label, self.preview_label2, target_plot_img_rgb, "preview2_image_tk")
            except Exception as e:
                print(f"Error loading or displaying {self.target_plot_path}: {e}")
                self.after(0, self._resize_and_update_label, self.preview_label2, blank_img, "preview2_image_tk")
        else:
            self.after(0, self._resize_and_update_label, self.preview_label2, blank_img, "preview2_image_tk")

    def update_point_track_previews(self):
        blank = np.zeros((200, 200, 3), dtype=np.uint8)  # or use your warp_dims
        
        # Preview 1 = the raw “targets_plot.jpg”
        if os.path.exists(self.point_warped_plot_path):
            img = cv2.imread(self.point_warped_plot_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._resize_and_update_label(self.preview_label1, img, "preview1_image_tk")
        else:
            self._resize_and_update_label(self.preview_label1, blank, "preview1_image_tk")

        # Preview 2 = the warped arena image “warped_plot.jpg”
        if os.path.exists(self.point_targets_plot_path):
            img2 = cv2.imread(self.point_targets_plot_path)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            self._resize_and_update_label(self.preview_label2, img2, "preview2_image_tk")
        else:
            self._resize_and_update_label(self.preview_label2, blank, "preview2_image_tk")

    def update_spline_track_previews(self):
        # blank fallback (same size as warp_dims, or smaller)
        blank = np.zeros((self.warp_dims[1], self.warp_dims[0], 3), dtype=np.uint8)

        # paths must match what your stop_tracking actually wrote
        spline_overlay_path = "targets/spline_track/targets_plot.jpg"
        warped_plot_path   = "targets/spline_track/warped_plot.jpg"

        # ---- Preview 1: spline overlay ----
        if os.path.exists(warped_plot_path):
            img = cv2.imread(warped_plot_path)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._resize_and_update_label(self.preview_label1, rgb, "preview1_image_tk")
            else:
                print(f"Warning: failed to load {spline_overlay_path}")
                self._resize_and_update_label(self.preview_label1, blank, "preview1_image_tk")
        else:
            self._resize_and_update_label(self.preview_label1, blank, "preview1_image_tk")

        # ---- Preview 2: warped arena plot ----
        if os.path.exists(spline_overlay_path):
            img2 = cv2.imread(spline_overlay_path)
            if img2 is not None:
                rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                self._resize_and_update_label(self.preview_label2, rgb2, "preview2_image_tk")
            else:
                print(f"Warning: failed to load {warped_plot_path}")
                self._resize_and_update_label(self.preview_label2, blank, "preview2_image_tk")
        else:
            self._resize_and_update_label(self.preview_label2, blank, "preview2_image_tk")
    
    def update_marker_track_previews(self):
        # blank fallback (same size as warp_dims, or smaller)
        blank = np.zeros((self.warp_dims[1], self.warp_dims[0], 3), dtype=np.uint8)

        # paths must match what your stop_tracking actually wrote
        marker_overlay_path = "targets/marker_track/targets_plot.jpg"
        warped_plot_path   = "targets/marker_track/warped_plot.jpg"

        # ---- Preview 1: spline overlay ----
        if os.path.exists(warped_plot_path):
            img = cv2.imread(warped_plot_path)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._resize_and_update_label(self.preview_label1, rgb, "preview1_image_tk")
            else:
                print(f"Warning: failed to load {marker_overlay_path}")
                self._resize_and_update_label(self.preview_label1, blank, "preview1_image_tk")
        else:
            self._resize_and_update_label(self.preview_label1, blank, "preview1_image_tk")

        # ---- Preview 2: warped arena plot ----
        if os.path.exists(marker_overlay_path):
            img2 = cv2.imread(marker_overlay_path)
            if img2 is not None:
                rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                self._resize_and_update_label(self.preview_label2, rgb2, "preview2_image_tk")
            else:
                print(f"Warning: failed to load {warped_plot_path}")
                self._resize_and_update_label(self.preview_label2, blank, "preview2_image_tk")
        else:
            self._resize_and_update_label(self.preview_label2, blank, "preview2_image_tk")

    def _start_execution_thread(self):
        # don’t double‐start
        if self.is_executing_pattern:
            return

        mode = self.mode_var.get()
        targets = {
            "Pattern Gen": "targets/pattern_llm/targets.txt",
            "Point Track":  "targets/point_track/targets.txt",
            "Spline Track": "targets/spline_track/targets.txt",
            "Marker Track": "targets/marker_track/targets.txt",
        }
        target_file = targets.get(mode)
        self.bread_crumb_path = target_file
        if not target_file:
            messagebox.showerror("Error", f"No target file for mode '{mode}'")
            return

        # disable this mode’s controls
        self.is_executing_pattern = True
        if mode == "Pattern Gen":
            self.generate_pattern_gen_button.  configure(state="disabled")
            self.execute_pattern_gen_button.   configure(state="disabled", text="Executing…")
        elif mode == "Point Track":
            self.calibrate_track_point_button .configure(state="disabled")
            self.start_track_point_button     .configure(state="disabled")
            self.stop_track_point_button      .configure(state="disabled")
            self.execute_track_point_button   .configure(state="disabled", text="Executing…")
        elif mode == "Spline Track":
            self.calibrate_track_spline_button.configure(state="disabled")
            self.start_track_spline_button    .configure(state="disabled")
            self.stop_track_spline_button     .configure(state="disabled")
            self.execute_track_spline_button  .configure(state="disabled", text="Executing…")
        elif mode == "Marker Track":
            self.calibrate_track_marker_button.configure(state="disabled")
            self.generate_marker_gen_button   .configure(state="disabled")
            self.execute_marker_spline_button .configure(state="disabled", text="Executing…")

        port_full = f"/dev/tty{self.serial_port}"
        baud      = self.selected_baud_rate
        cmd_file  = "controls/command.txt"
        pose_file = "controls/pose.txt"
        err_file  = "controls/error.txt"

        stop_evt = threading.Event()

        # Worker wrapper for move_robot
        def _move_worker():
            try:
                # ← now passing stop_evt
                move_robot(port_full, baud, cmd_file, stop_evt)
            except Exception as e:
                print(f"[Move Error] {e}")
            finally:
                # make extra sure UI is re‐enabled
                self.after(0, self._execution_complete)

        def _controller_worker():
            try:
                run_controller(target_file, pose_file, cmd_file, err_file)
            except Exception as e:
                print(f"[Controller Error] {e}")
            finally:
                # signal move_robot to exit its loop
                stop_evt.set()

        # start both threads
        threading.Thread(target=_move_worker,      daemon=True).start()
        threading.Thread(target=_controller_worker, daemon=True).start()

    def _execution_complete(self):
        self.bread_crumb_path = ""
        mode = self.mode_var.get()
        if mode == "Pattern Gen":
            self.generate_pattern_gen_button.configure(state="normal")
            self.execute_pattern_gen_button.configure(state="normal", text="Execute")
        elif mode == "Point Track":
            self.calibrate_track_point_button.configure(state="normal")
            self.start_track_point_button.configure(state="normal")
            self.stop_track_point_button.configure(state="disabled")
            self.execute_track_point_button.configure(state="normal", text="Execute")
        elif mode == "Spline Track":
            self.calibrate_track_spline_button.configure(state="normal")
            self.start_track_spline_button.configure(state="normal")
            self.stop_track_spline_button.configure(state="disabled")
            self.execute_track_spline_button.configure(state="normal", text="Execute")
        elif mode == "Marker Track":
            self.calibrate_track_marker_button.configure(state="normal")
            self.generate_marker_gen_button.configure(state="normal")
            self.execute_marker_spline_button.configure(state="normal", text="Execute")

        self.is_executing_pattern = False
        print("Execution complete.")


    def create_point_track_widgets(self):
        self.dynamic_content_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.calibrate_track_point_button = ctk.CTkButton(self.dynamic_content_frame, text="Calibrate Arena", corner_radius=8)
        self.calibrate_track_point_button.configure(command=lambda: self._start_calibration_thread(self.calibrate_track_point_button))
        self.calibrate_track_point_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.start_track_point_button = ctk.CTkButton(self.dynamic_content_frame, text="Start Tracking", command=self.track_point, corner_radius=8)
        self.start_track_point_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.stop_track_point_button = ctk.CTkButton(self.dynamic_content_frame, text="Stop Tracking", command=self.stop_tracking, corner_radius=8, state="disabled")
        self.stop_track_point_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.execute_track_point_button = ctk.CTkButton(self.dynamic_content_frame, text="Execute", command=self._start_execution_thread, corner_radius=8)
        self.execute_track_point_button.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

    def create_spline_track_widgets(self):
        self.dynamic_content_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.calibrate_track_spline_button = ctk.CTkButton(self.dynamic_content_frame, text="Calibrate Arena", corner_radius=8)
        self.calibrate_track_spline_button.configure(command=lambda: self._start_calibration_thread(self.calibrate_track_spline_button))
        self.calibrate_track_spline_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.start_track_spline_button = ctk.CTkButton(self.dynamic_content_frame, text="Start Tracking", command=self.track_spline, corner_radius=8)
        self.start_track_spline_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.stop_track_spline_button = ctk.CTkButton(self.dynamic_content_frame, text="Stop Tracking", command=self.stop_tracking, corner_radius=8, state="disabled")
        self.stop_track_spline_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.execute_track_spline_button = ctk.CTkButton(self.dynamic_content_frame, text="Execute", command=self._start_execution_thread, corner_radius=8)
        self.execute_track_spline_button.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

    def create_marker_track_widgets(self):
        # same column setup as Pattern-Gen
        self.dynamic_content_frame.grid_columnconfigure(0, weight=0)
        self.dynamic_content_frame.grid_columnconfigure(1, weight=3)
        self.dynamic_content_frame.grid_columnconfigure((2, 3), weight=1)

        # Sequence entry
        ctk.CTkLabel(self.dynamic_content_frame, text="Sequence:", font=("Arial",14)).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.seq_entry_marker_button = ctk.CTkEntry(self.dynamic_content_frame, placeholder_text="e.g. Home,A,B,Home")
        self.seq_entry_marker_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Calibrate button
        self.calibrate_track_marker_button = ctk.CTkButton(self.dynamic_content_frame, text="Calibrate Arena", corner_radius=8)
        self.calibrate_track_marker_button.configure(command=lambda: self._start_calibration_thread(self.calibrate_track_marker_button))
        self.calibrate_track_marker_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # Generate / Execute
        self.generate_marker_gen_button = ctk.CTkButton(self.dynamic_content_frame, text="Generate", command=self._start_marker_generation_thread, corner_radius=8)
        self.generate_marker_gen_button.grid(row=0, column=3, padx=5, pady=5, sticky="ew")
        
        self.execute_marker_spline_button = ctk.CTkButton(self.dynamic_content_frame, text="Execute", command=self._start_execution_thread, corner_radius=8)
        self.execute_marker_spline_button.grid(row=0, column=4, padx=5, pady=5, sticky="ew")

    def _start_calibration_thread(self, button):
        button.configure(state="disabled", text="Calibrating...")
        
        calibration_thread = threading.Thread(
            target=self.calibrate_camera, 
            args=(button,),
            daemon=True
        )
        calibration_thread.start()

    def _start_marker_generation_thread(self):
        self.generate_marker_gen_button.configure(state="disabled")
        threading.Thread(target=self.generate_marker, daemon=True).start()

    def generate_marker(self):
        prompt = self.seq_entry_marker_button.get()
        
        if not prompt.strip():
            self.after(0, lambda: messagebox.showwarning("Input Required", "The pattern prompt cannot be empty. Please enter a description."))
            self.after(0, lambda: self.generate_marker_gen_button.configure(state="normal"))
            return

        print(f"Generating pattern based on: {prompt}")
        gm_api_key = os.getenv("GEMINI_API_KEY")

        try:
            seq = parse_marker_sequence(prompt, gm_api_key)
            self._start_marker_generate(seq)
        except Exception as e:
            error_message = f"An error occurred during marker generation:\n\n{e}"
            print(error_message)
            self.after(0, lambda: messagebox.showerror("Marker Generation Failed", error_message))
        finally:
            self.after(0, lambda: self.generate_marker_gen_button.configure(state="normal"))

    def _start_marker_generate(self, seq):
        if not self.is_calibrated or self.perspective_matrix is None:
            messagebox.showerror("Calibration Required", "Please calibrate the arena before starting to track points.")
            return

        # 1) Generate Path using markers
        marker_target_dir = os.path.join("targets", "marker_track")
        letter_map = {ltr:pos for ltr,pos in self.alphabet_info}
        marker_target_points = generate_marker_track(
            sequence=seq,
            home_pos=self.home_pos,
            alphabet_info=letter_map,
            obstacles=self.obstacles,
            arena_corners=self.arena_corners,
            camera_size=self.camera_orig_size[::-1],
            last_robot_angle=self.last_robot_angle,
            out_dir="targets/marker_track"
        )

        # 2) Warp them
        warped_txt  = os.path.join(marker_target_dir, "warped.txt")
        warped_plot = os.path.join(marker_target_dir, "warped_plot.jpg")
        if self.perspective_matrix is not None and marker_target_points:
            pts_arr = np.array([[x, y] for x, y, _ in marker_target_points],
                                dtype=np.float32).reshape(-1,1,2)
            warped = cv2.perspectiveTransform(pts_arr, self.perspective_matrix).reshape(-1,2)
            with open(warped_txt, "w") as f:
                for (xw,yw), (_,_,theta) in zip(warped, marker_target_points):
                    f.write(f"{int(xw)},{int(yw)},{theta}\n")
        else:
            # ensure file exists even if empty
            open(warped_txt, "w").close()

        # 3) Generate the warped‐arena plot
        generate_and_save_warped_plot(warped_txt, warped_plot, self.warp_dims)

        # 4) Draw the spline (polyline) on a frame
        #    (here using your test image—swap to live frame if desired)
        frame = cv2.imread(IMG_PATH)
        pts_dicts = [{"x":x,"y":y,"theta":theta}
                        for x,y,theta in marker_target_points]

        targets_plot = os.path.join(marker_target_dir, "targets_plot.jpg")
        if frame is not None and pts_dicts:
            overlay = overlay_points_n_lines_on_image(frame, pts_dicts)
            cv2.imwrite(targets_plot, overlay)
        else:
            # touch file or skip
            open(targets_plot, "w").close()


    def calibrate_camera(self, button):
        try:
            if not self.cap or not self.cap.isOpened():
                print("Error: Camera not running for calibration.")
                return
            
            # ret, frame = self.cap.read()
            # TESTING ONLY - REMOVE LATER
            frame = cv2.imread(IMG_PATH)
            ret = frame is not None
            # TESTING ONLY - REMOVE LATER

            if not ret or frame is None:
                print("Error: Could not read frame from camera for calibration.")
                return

            print("Starting calibration... Please wait.")

            md_api_key = os.getenv("MOONDREAM_API_KEY")
            if not md_api_key:
                print("Error: MOONDREAM_API_KEY not set.")
                return
            
            # 1) Detect the Arena
            warped, M, out_size, corners_int = detect_and_warp(
                frame,
                md_api_key,
                self.marker_prompt,
                self.warp_dims
            )
            self.perspective_matrix = M
            self.arena_corners = corners_int

            # 2) Detect the Obstacles
            raw_obstacles = detect_and_list(frame, md_api_key, self.obstacles_prompt)
            # inflate by 120px
            offset = 20
            h, w = frame.shape[:2]
            inflated = []
            for obs in raw_obstacles:
                x, y, bw, bh = obs["bbox"]
                x1 = max(0, x - offset)
                y1 = max(0, y - offset)
                x2 = min(w, x + bw + offset)
                y2 = min(h, y + bh + offset)
                inflated.append({
                    "bbox":     (x1, y1, x2 - x1, y2 - y1),
                    "centroid": obs["centroid"],
                    "area":     obs["area"],
                })
            # store the inflated obstacles for use later
            self.obstacles = inflated

            # 3) Detect the Alphabets
            letters = detect_and_read_alphabets(frame, md_api_key)
            self.alphabet_info = letters
            print(f"Detected letters: {self.alphabet_info}")

            # 4) Detect Homing Position
            home = detect_home(frame, md_api_key)
            self.home_pos = home
            print(f"Detected letters: {self.home_pos}")

            self.is_calibrated = True
            print("Calibration successful.")

        except Exception as e:
            print(f"An exception occurred during calibration: {e}")
            self.is_calibrated = False
            self.arena_corners = None
            self.perspective_matrix = None

        finally:
            # Re-enable the button on the UI thread
            self.after(0, lambda: button.configure(state="normal", text="Calibrate Arena"))

    def _build_obstacle_mask(self):
        """Create (or reuse) a binary mask where obstacles==255, free==0."""
        if hasattr(self, "_obs_mask") and self._obs_mask is not None:
            return self._obs_mask

        h, w = self.camera_orig_size[1], self.camera_orig_size[0]
        mask = np.zeros((h, w), dtype="uint8")
        for obs in getattr(self, "obstacles", []):
            x, y, bw, bh = obs["bbox"]
            cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
        self._obs_mask = mask
        return mask

    def _line_intersects_obstacle(self, p1, p2):
        """Return True if the line segment p1→p2 hits any non-zero pixel in obs mask."""
        mask = self._build_obstacle_mask()
        x1, y1 = p1; x2, y2 = p2
        dx = abs(x2 - x1); dy = abs(y2 - y1)
        x, y = x1, y1
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if mask[y, x]:
                    return True
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                if mask[y, x]:
                    return True
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        return mask[y2, x2] != 0

    def _astar_path(self, start, goal):
        """A* on the obstacle mask (8-connected). Returns list of (x,y) or None."""
        mask = self._build_obstacle_mask()
        h, w = mask.shape
        def hscore(a, b):
            return math.hypot(a[0]-b[0], a[1]-b[1])

        neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        open_set = []
        heapq.heappush(open_set, (hscore(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            f, g, current = heapq.heappop(open_set)
            if current == goal:
                # reconstruct
                path = []
                while current:
                    path.append(current)
                    current = came_from.get(current)
                return path[::-1]

            if g > g_score.get(current, float('inf')):
                continue

            for dx, dy in neighbors:
                nx, ny = current[0]+dx, current[1]+dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if mask[ny, nx]:
                    continue
                tentative_g = g + math.hypot(dx, dy)
                if tentative_g < g_score.get((nx, ny), float('inf')):
                    g_score[(nx, ny)] = tentative_g
                    came_from[(nx, ny)] = current
                    heapq.heappush(open_set, (tentative_g + hscore((nx,ny), goal), tentative_g, (nx, ny)))

        return None

    def on_video_click(self, event):
        mode = self.mode_var.get()
        if mode not in ("Point Track", "Spline Track"):
            return

        # 1) map widget‐coords → full‐res coords
        click_x, click_y = event.x, event.y
        disp_w, disp_h = self.camera_display_size
        orig_w, orig_h = self.camera_orig_size
        if not (0 <= click_x <= disp_w and 0 <= click_y <= disp_h):
            return
        orig_x = int(click_x * orig_w / disp_w)
        orig_y = int(click_y * orig_h / disp_h)
        
        # 2) arena bounds check
        if self.arena_corners is not None:
            inside = cv2.pointPolygonTest(
                self.arena_corners.astype(np.int32),
                (orig_x, orig_y),
                False
            ) >= 0
            if not inside:
                messagebox.showwarning("Outside Arena", "That point is outside the arena.")
                return

        # 3) obstacle‐aware routing
        theta = getattr(self, "last_robot_angle", 0.0)
        new_pt = (orig_x, orig_y)

        # if we have at least one previous point, try to route
        if self.tracked_points:
            prev = self.tracked_points[-1]
            prev_pt = (prev[0], prev[1])

            # build planner once per click
            planner = PathPlanner(
                self.obstacles,
                self.camera_orig_size[::-1],
                arena_corners=self.arena_corners
            )

            route = planner.find_obstacle_aware_path(prev_pt, new_pt, simplify_dist=10)

            if route is None:
                messagebox.showwarning(
                    "Blocked",
                    "No obstacle‐free path found to that point; try another."
                )
                return

            # if route is just [start, goal], record the click as before
            if len(route) == 2:
                self.tracked_points.append((orig_x, orig_y, theta))
                print(f"[{mode}] Tracked point: x={orig_x}, y={orig_y}, θ={theta:.2f}")
            else:
                # we got an A* path: append intermediate waypoints
                # drop the first element (same as prev_pt)
                for x, y in route[1:]:
                    self.tracked_points.append((x, y, theta))
                print(f"[{mode}] Routed around obstacle with {len(route)-1} segments.")
        else:
            # first point: just record it
            self.tracked_points.append((orig_x, orig_y, theta))
            print(f"[{mode}] Tracked point: x={orig_x}, y={orig_y}, θ={theta:.2f}")

    def track_point(self):
        if not self.is_calibrated or self.perspective_matrix is None:
            messagebox.showerror("Calibration Required", "Please calibrate the arena before starting to track points.")
            return

        # clear previous points
        self.tracked_points.clear()

        # bind left-click on the main video feed
        self.camera_label.bind("<Button-1>", self.on_video_click)
        self.is_tracking = True

        print("Point tracking mode activated.")
        self.calibrate_track_point_button.configure(state="disabled")
        self.start_track_point_button.configure(state="disabled")
        self.stop_track_point_button.configure(state="normal")
        self.execute_track_point_button.configure(state="disabled")


    def execute_point_track(self):
        print("Executing point track...")

    def track_spline(self):
        if not self.is_calibrated or self.perspective_matrix is None:
            messagebox.showerror("Calibration Required", "Please calibrate the arena before starting to track splines.")
            return

        # clear previous points
        self.tracked_points.clear()

        # bind both the button-1 press and motion-while-button-1
        self.camera_label.bind("<Button-1>",     self.on_video_click)
        self.camera_label.bind("<B1-Motion>",    self.on_video_click)
        self.is_tracking = True

        print("Spline tracking mode activated.")

        self.calibrate_track_spline_button.configure(state="disabled")
        self.start_track_spline_button.configure(state="disabled")
        self.stop_track_spline_button.configure(state="normal")
        self.execute_track_spline_button.configure(state="disabled")


    
    def stop_tracking(self):
        print("Tracking stopped.")
        current_mode = self.mode_var.get()

        if current_mode == "Point Track":
            # 1) Save the raw points to file:
            # unbind clicks
            self.camera_label.unbind("<Button-1>")
            self.is_tracking = False
            os.makedirs(os.path.dirname(self.point_track_save_path), exist_ok=True)
            total = len(self.tracked_points)
            with open(self.point_track_save_path, "w") as f:
                f.write(f"({total}/{total})\n")
                for x, y, theta in self.tracked_points:
                    f.write(f"{x},{y},{theta}\n")

            # 2) Warp the arena to self.warp_dims and transform each point
            warp_save_path = os.path.join(
                os.path.dirname(self.point_track_save_path),
                "warped.txt"
            )
            warp_plot_save_path = os.path.join(
                os.path.dirname(self.point_track_save_path),
                "warped_plot.jpg"
            )
            if self.perspective_matrix is not None and self.tracked_points:
                # build an Nx1x2 array of your (x,y) coords
                pts_arr = np.array(
                    [[x, y] for x, y, _ in self.tracked_points],
                    dtype=np.float32
                ).reshape(-1, 1, 2)

                # apply the homography
                warped = cv2.perspectiveTransform(pts_arr, self.perspective_matrix)
                warped = warped.reshape(-1, 2)  # back to Nx2

                # write warped x,y (and keep theta unchanged)
                with open(warp_save_path, "w") as f:
                    for (xw, yw), (_, _, theta) in zip(warped, self.tracked_points):
                        f.write(f"{int(xw)},{int(yw)},{theta}\n")
                warped_msg = f"Warped points saved to\n{warp_save_path}"
            else:
                warped_msg = "(no warped points generated)"
            
            # 3) Save the warped plot
            generate_and_save_warped_plot(warp_save_path, warp_plot_save_path, self.warp_dims)

            # 3) Grab a frame to draw on            
            # ret, frame = self.cap.read()
            # TESTING ONLY - REMOVE LATER
            ret = True
            frame = cv2.imread(IMG_PATH)
            # TESTING ONLY - REMOVE LATER

            # 4) Build the list of dicts for the overlay function
            pts_dicts = [
                {"x": float(x), "y": float(y), "theta": float(theta)}
                for x, y, theta in self.tracked_points
            ]

            # 5) Overlay and save
            if frame is not None and pts_dicts:
                overlay_img = overlay_points_n_lines_on_image(frame, pts_dicts)
                overlay_path = os.path.join(
                    os.path.dirname(self.point_track_save_path),
                    "targets_plot.jpg"
                )
                cv2.imwrite(overlay_path, overlay_img)
                overlay_msg = f"Overlay image saved to\n{overlay_path}"
            else:
                overlay_msg = "(no overlay image generated)"

            # final dialog
            messagebox.showinfo("Success", "Points tracked successfully!")

            # reset button states
            self.calibrate_track_point_button.configure(state="normal")
            self.start_track_point_button.configure(state="normal")
            self.stop_track_point_button.configure(state="disabled")
            self.execute_track_point_button.configure(state="normal")

        elif current_mode == "Spline Track":
            # unbind clicks
            self.camera_label.unbind("<Button-1>")
            self.camera_label.unbind("<B1-Motion>")
            self.is_tracking = False
            # base folder for spline outputs
            spline_dir = os.path.join("targets", "spline_track")
            os.makedirs(spline_dir, exist_ok=True)

            # 1) Save raw control points
            raw_path   = os.path.join(spline_dir, "targets.txt")
            total = len(self.tracked_points)
            with open(raw_path, "w") as f:
                f.write(f"({total}/{total})\n")
                for x, y, theta in self.tracked_points:
                    f.write(f"{x},{y},{theta}\n")

            # 2) Warp them
            warped_txt  = os.path.join(spline_dir, "warped.txt")
            warped_plot = os.path.join(spline_dir, "warped_plot.jpg")
            if self.perspective_matrix is not None and self.tracked_points:
                pts_arr = np.array([[x, y] for x, y, _ in self.tracked_points],
                                   dtype=np.float32).reshape(-1,1,2)
                warped = cv2.perspectiveTransform(pts_arr, self.perspective_matrix).reshape(-1,2)
                with open(warped_txt, "w") as f:
                    for (xw,yw), (_,_,theta) in zip(warped, self.tracked_points):
                        f.write(f"{int(xw)},{int(yw)},{theta}\n")
            else:
                # ensure file exists even if empty
                open(warped_txt, "w").close()

            # 3) Generate the warped‐arena plot
            generate_and_save_warped_plot(warped_txt, warped_plot, self.warp_dims)

            # 4) Draw the spline (polyline) on a frame
            #    (here using your test image—swap to live frame if desired)
            frame = cv2.imread(IMG_PATH)
            pts_dicts = [{"x":x,"y":y,"theta":theta}
                         for x,y,theta in self.tracked_points]

            targets_plot = os.path.join(spline_dir, "targets_plot.jpg")
            if frame is not None and pts_dicts:
                overlay = overlay_points_n_lines_on_image(frame, pts_dicts)
                cv2.imwrite(targets_plot, overlay)
            else:
                # touch file or skip
                open(targets_plot, "w").close()

            # 5) Inform the user
            messagebox.showinfo("Success", "Spline tracked successfully!")

            # reset button states
            self.calibrate_track_spline_button.configure(state="normal")
            self.start_track_spline_button.configure(state="normal")
            self.stop_track_spline_button.configure(state="disabled")
            self.execute_track_spline_button.configure(state="normal")
        
    def execute_spline_track(self):
        print("Executing spline track...")

    def on_closing(self):
        self.stop_video_thread = True
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
            if self.video_thread.is_alive():
                print("Warning: Video thread did not terminate cleanly.")
        if self.cap is not None:
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = CameraApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()