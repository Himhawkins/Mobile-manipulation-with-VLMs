import os
import re
import json
import cv2
from PIL import Image, ImageTk
import customtkinter as ctk
import threading

# Import from new helper modules
from ui_utils import CTkMessageBox, CheckGroup, get_app_settings, open_settings_popup
from camera_utils import start_camera, read_frame, display_frame, draw_robot_pose
from agent_utils import save_agent_to_disk, get_agent_folders, get_agent_functions, get_all_functions
from port_utils import refresh_cameras, refresh_serial_ports
from detection import detect_robot_pose
from thread_utils import run_in_thread, disable_button, enable_button, callibrate_task, run_task, toggle_execute
from ui_utils import overlay_arena_and_obstacles, get_overlay_frame, draw_path_on_frame
from motion import move_robot_with_thread
from controller import exec_bot

class DashboardApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Dashboard")
        self.geometry("1200x800")

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # Top Section
        top_frame = ctk.CTkFrame(self)
        top_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        top_frame.grid_columnconfigure((0, 1, 2), weight=1)

        cam_frame = ctk.CTkFrame(top_frame)
        cam_frame.grid(row=0, column=0, sticky="w", padx=5)
        cam_frame.grid_columnconfigure((0,1), weight=1)
        init_camera_list = refresh_cameras()
        self.camera_var = ctk.StringVar(value=init_camera_list[0] if init_camera_list else "")
        self.camera_menu = ctk.CTkOptionMenu(
            cam_frame,
            variable=self.camera_var,
            values=init_camera_list,
            command=self.on_camera_change
        )
        self.camera_menu.grid(row=0, column=0, sticky="w")
        ctk.CTkButton(cam_frame, text="Refresh Cameras", command=self.on_refresh_cameras)\
            .grid(row=0, column=1, padx=(5,0))

        port_frame = ctk.CTkFrame(top_frame)
        port_frame.grid(row=0, column=1, padx=(5,150))
        init_serial_list = refresh_serial_ports()
        self.serial_var = ctk.StringVar(value=init_serial_list[0] if init_serial_list else "")
        self.serial_menu = ctk.CTkOptionMenu(
            port_frame,
            variable=self.serial_var,
            values=init_serial_list,
            command=self.on_serial_change
        )
        self.serial_menu.grid(row=0, column=0)
        ctk.CTkButton(port_frame, text="Refresh Ports", command=self.on_refresh_ports)\
            .grid(row=0, column=1, padx=(5,0))

        ctk.CTkButton(top_frame, text="Settings", command=self.on_settings)\
            .grid(row=0, column=2, sticky="e", padx=5)

        # Middle Section
        middle_frame = ctk.CTkFrame(self)
        middle_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        middle_frame.grid_columnconfigure(0, weight=2)
        middle_frame.grid_columnconfigure(1, weight=1)
        middle_frame.grid_rowconfigure((0,1), weight=1)

        self.video_frame = ctk.CTkFrame(middle_frame, border_width=2, corner_radius=10)
        self.video_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(10,5), pady=10)
        self.video_label = ctk.CTkLabel(self.video_frame, text="", fg_color="black", corner_radius=10)
        self.video_label.pack(expand=True, fill="both")

        self.cap = None
        self.after(500, lambda: self.on_camera_change(self.camera_var.get()))

        self.preview1_txt = ctk.CTkTextbox(
            middle_frame,
            width=700,
            border_width=1,
            corner_radius=8,
            state="disabled"
        )
        self.preview1_txt.grid(row=0, column=1, sticky="nsew", pady=(10,5), padx=(5,10))
        self.preview1_txt.grid_propagate(False)

        self.preview2 = ctk.CTkFrame(middle_frame, width=700, border_width=1, corner_radius=8)
        self.preview2.grid(row=1, column=1, sticky="nsew", pady=(5,10), padx=(5,10))
        self.preview2.grid_propagate(False)

        self.preview2_img_label = ctk.CTkLabel(self.preview2, text="")
        self.preview2_img_label.pack(expand=True, fill="both")

        # Bottom Section
        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(5,10))
        bottom_frame.grid_columnconfigure(0, weight=0)
        bottom_frame.grid_columnconfigure(1, weight=0)
        bottom_frame.grid_columnconfigure(2, weight=1)

        agent_folders = get_agent_folders()
        agent_folders.append("Create New")
        self.mode_var = ctk.StringVar(value=agent_folders[0] if agent_folders else "")
        self.mode_menu = ctk.CTkOptionMenu(
            bottom_frame,
            width=200,
            variable=self.mode_var,
            values=agent_folders,
            command=self.on_mode_change
        )
        self.mode_menu.grid(row=0, column=0, sticky="w")
        ctk.CTkButton(bottom_frame, text="Edit", command=self.on_edit)\
            .grid(row=0, column=1, sticky="w", padx=(5,0))

        action_frame = ctk.CTkFrame(bottom_frame)
        action_frame.grid(row=0, column=2, sticky="ew", padx=(10,0))
        action_frame.grid_columnconfigure(0, weight=1)

        self.input_var = ctk.StringVar()
        self.input_entry = ctk.CTkEntry(
            action_frame,
            textvariable=self.input_var,
            placeholder_text="Enter value...",
            width=250
        )
        self.input_entry.grid(row=0, column=0, sticky="ew", padx=(0,5))

        self.calibrate_btn = ctk.CTkButton(
            action_frame,
            text="Calibrate",
            command=lambda: self.on_mode_action("Calibrate")
        )
        self.calibrate_btn.grid(row=0, column=1, padx=5)

        self.run_btn = ctk.CTkButton(
            action_frame,
            text="Run",
            command=lambda: self.on_mode_action("Run")
        )
        self.run_btn.grid(row=0, column=2, padx=5)

        self.is_executing = False
        self.execute_btn = ctk.CTkButton(
            action_frame,
            text="Execute",
            fg_color="#7228E9",
            command=lambda: toggle_execute(self, self.serial_var, self.execute_btn)
        )
        self.execute_btn.grid(row=0, column=3, padx=5)

        self.move_thread = None
        self.stop_event = threading.Event()

    def on_camera_change(self, value):
        m = re.search(r"\d+", value)
        if not m:
            print(f"Could not parse camera index from '{value}'")
            return
        cam_index = int(m.group())

        if self.cap:
            self.cap.release()

        self.cap = start_camera(cam_index)
        if not self.cap:
            print(f"Failed to open camera {cam_index}")
            return
        self.on_update_video()

    def on_update_video(self):
        if not self.cap or not self.cap.isOpened():
            return

        self.update_idletasks()
        cw = self.video_frame.winfo_width()
        ch = self.video_frame.winfo_height()
        if cw < 2 or ch < 2:
            return self.after(16, self.on_update_video)

        self.current_frame = read_frame(self.cap)
        self.settings = get_app_settings()
        aruco_id = int(self.settings.get("aruco_id", "782"))
        pose = detect_robot_pose(
            frame=self.current_frame,
            aruco_id=aruco_id,
            save_path="Data/robot_pos.txt"
        )

        if pose:
            cx, cy, theta, pts = pose
            d_frame = draw_robot_pose(self.current_frame, cx, cy, theta, pts)
        else:
            d_frame = self.current_frame
        
        overlay = overlay_arena_and_obstacles(
                frame=d_frame,
                arena_path="Data/arena_corners.txt",
                obstacles_path="Data/obstacles.txt"
            )
        draw_frame = draw_path_on_frame(overlay, path_file="Targets/path.txt")

        img = display_frame(frame=draw_frame, target_w=cw, target_h=ch)
        if img:
            self.video_label.configure(image=img)
            self.video_label.image = img

        # preview2: warped top-down arena
        width = int(self.settings.get("arena_width", "1200"))
        height = int(self.settings.get("arena_height", "900"))
        warped = get_overlay_frame(warp_size=(width, height))
        preview_img = display_frame(warped, width//2, height//2)
        if preview_img:
            self.preview2_img_label.configure(image=preview_img)
            self.preview2_img_label.image = preview_img

        self.after(16, self.on_update_video)

    def on_refresh_cameras(self):
        vals = refresh_cameras()
        self.camera_menu.configure(values=vals)
        if vals:
            self.camera_var.set(vals[0])

    def on_serial_change(self, value):
        pass

    def on_refresh_ports(self):
        vals = refresh_serial_ports()
        self.serial_menu.configure(values=vals)
        if vals:
            self.serial_var.set(vals[0])

    def on_settings(self):
        if getattr(self, "settings_popup", None) and self.settings_popup.winfo_exists():
            self.settings_popup.lift()
            self.settings_popup.focus_force()
            return
        open_settings_popup(self)

    def on_mode_change(self, value):
        if value == "Create New":
            self.on_edit()

    def on_mode_action(self, action):
        if action == "Calibrate":
            run_in_thread(
                callback=lambda: callibrate_task(self),
                on_start=lambda: disable_button(self.calibrate_btn),
                on_complete=lambda: enable_button(self.calibrate_btn)
            )
        elif action == "Run":
            run_in_thread(
                callback=lambda: run_task(self, self.preview1_txt, self.input_var.get(), self.mode_var.get()),
                on_start=lambda: disable_button(self.run_btn),
                on_complete=lambda: enable_button(self.run_btn)
            )

    def on_edit(self):
        if getattr(self, "edit_popup", None) and self.edit_popup.winfo_exists():
            self.edit_popup.lift()
            self.edit_popup.focus_force()
            return

        selected = self.mode_var.get()
        old_folder = "" if selected == "Create New" else selected
        existing = {} if selected == "Create New" else get_agent_functions(old_folder)

        self.edit_popup = ctk.CTkToplevel(self)
        self.edit_popup.title(f"Edit Agent: {old_folder}")
        self.edit_popup.geometry("500x900")
        self.edit_popup.protocol("WM_DELETE_WINDOW", lambda: self.edit_popup.destroy())

        ctk.CTkLabel(self.edit_popup, text="Name:").pack(anchor="w", padx=10, pady=(10,0))
        name_var = ctk.StringVar(value=old_folder)
        ctk.CTkEntry(self.edit_popup, textvariable=name_var).pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(self.edit_popup, text="Description:").pack(anchor="w", padx=10, pady=(10,0))
        desc_box = ctk.CTkTextbox(self.edit_popup, height=350)
        desc_box.pack(fill="both", padx=10, pady=5)
        desc_path = os.path.join("Agents", old_folder, "description.txt")
        try:
            with open(desc_path, "r") as f:
                desc_box.insert("0.0", f.read())
        except FileNotFoundError:
            pass

        ctk.CTkLabel(self.edit_popup, text="Functions:").pack(anchor="w", padx=10, pady=(10,0))
        scroll = ctk.CTkScrollableFrame(self.edit_popup, height=300)
        scroll.pack(fill="both", expand=True, padx=10, pady=5)

        self.check_groups = []
        for label, funcs in get_all_functions().items():
            cg = CheckGroup(scroll, label, funcs, selected=existing.get(label, []))
            cg.pack(fill="x", pady=2)
            self.check_groups.append(cg)

        btn_frame = ctk.CTkFrame(self.edit_popup)
        btn_frame.pack(pady=10)
        ctk.CTkButton(btn_frame, text="Save",
            command=lambda: self.on_save_agent(name_var, desc_box, old_folder)
        ).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Cancel",
            command=lambda: self.edit_popup.destroy()
        ).pack(side="left", padx=5)

    def on_save_agent(self, name_var, desc_box, old_folder):
        new_name = name_var.get().strip()
        if not new_name:
            CTkMessageBox(self.edit_popup, "Invalid Name", "Agent name cannot be empty!", "red")
            return
        desc_text = desc_box.get("0.0", "end-1c")
        save_agent_to_disk(self.edit_popup, old_folder, new_name, desc_text, self.check_groups)
        CTkMessageBox(self.edit_popup, "Success", "Saved agent successfully.", "white")
        updated = get_agent_folders() + ["Create New"]
        self.mode_menu.configure(values=updated)
        self.mode_var.set(new_name)
        self.edit_popup.destroy()

if __name__ == "__main__":
    app = DashboardApp()
    app.mainloop()
