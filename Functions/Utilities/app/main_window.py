# vision_dashboard/app/main_window.py

import customtkinter as ctk
from PIL import Image
import cv2
import os

# Local application imports
from rpc_system import RPCClient
from .widgets import ControlPanel, CTkMessageBox, CheckGroup # Import CheckGroup
from core import threading as thread_utils
from utils import settings, ports, agents

class DashboardApp(ctk.CTk):
    """The main application window, which acts as an RPC client."""
    def __init__(self):
        super().__init__()
        self.title("Vision Dashboard (RPC Client)")
        self.geometry("1400x850")
        
        # --- State & Backend Connection ---
        self.is_executing = False
        try:
            self.client = RPCClient()
        except Exception as e:
            print(f"FATAL: Could not connect to RPC Server. Is it running? Error: {e}")
            self.after(20, self.destroy)
            return

        # --- GUI Layout ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.control_panel = ControlPanel(self)
        self.control_panel.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
        
        video_panel = ctk.CTkFrame(self)
        video_panel.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
        video_panel.grid_rowconfigure(0, weight=1)
        video_panel.grid_columnconfigure(0, weight=1)
        
        self.video_label = ctk.CTkLabel(video_panel, text="Connecting to data server...", fg_color="black")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Start the display loop and set the close protocol
        self.update_display()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- Callbacks that trigger tasks ---
    def on_calibrate(self):
        prompt = self.control_panel.prompt_entry.get()
        if not prompt:
            CTkMessageBox(self, "Warning", "Calibration prompt cannot be empty.", "yellow")
            return
        thread_utils.run_in_thread(
            callback=lambda: self.client.Tasks.calibrate(prompt),
            on_start=lambda: self.control_panel.calibrate_button.configure(state="disabled"),
            on_complete=lambda: self.control_panel.calibrate_button.configure(state="normal")
        )

    def on_execute(self):
        self.is_executing = not self.is_executing
        try:
            if self.is_executing:
                self.client.Robot.start_execution()
                self.control_panel.execute_button.configure(text="Stop", fg_color="#FF3B30")
            else:
                self.client.Robot.stop_execution()
                self.control_panel.execute_button.configure(text="Execute", fg_color="#7228E9")
        except Exception as e:
            CTkMessageBox(self, "Error", f"Execution command failed:\n{e}", "red")
            self.is_executing = False # Reset state on error
            self.control_panel.execute_button.configure(text="Execute", fg_color="#7228E9")

    def on_agent_change(self, value):
        """Called when a new agent is selected from the dropdown."""
        if value == "Create New":
            # Clear selection and open edit window for a new agent
            current_agents = agents.get_agent_folders()
            self.control_panel.agent_var.set(current_agents[0] if current_agents else "Create New")
            self.on_edit_agent(is_new=True)

    def on_edit_agent(self, is_new=False):
        """Opens the popup window to create or edit an agent."""
        selected_agent = self.control_panel.agent_var.get()
        old_folder = "" if is_new or selected_agent == "Create New" else selected_agent
        
        popup = ctk.CTkToplevel(self)
        popup.title(f"Edit Agent: {old_folder or 'New Agent'}")
        popup.geometry("500x700")
        
        # --- Name and Description ---
        ctk.CTkLabel(popup, text="Agent Name:").pack(anchor="w", padx=10, pady=(10,0))
        name_var = ctk.StringVar(value=old_folder)
        ctk.CTkEntry(popup, textvariable=name_var).pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(popup, text="Description:").pack(anchor="w", padx=10, pady=(10,0))
        desc_box = ctk.CTkTextbox(popup, height=150)
        desc_box.pack(fill="x", padx=10, pady=5)
        
        # --- Functions Checklist ---
        ctk.CTkLabel(popup, text="Functions:", font=("", 14, "bold")).pack(anchor="w", padx=10, pady=(10,0))
        scroll_frame = ctk.CTkScrollableFrame(popup, height=300)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Load existing data
        if old_folder:
            try:
                with open(os.path.join("Agents", old_folder, "description.txt"), "r") as f:
                    desc_box.insert("0.0", f.read())
            except FileNotFoundError:
                pass

        all_funcs = agents.get_all_functions()
        agent_funcs = agents.get_agent_functions(old_folder)
        
        check_groups = []
        for category, funcs in all_funcs.items():
            # Create a CheckGroup for each category
            group = CheckGroup(scroll_frame, category, funcs, selected=agent_funcs.get(category, []))
            group.pack(fill="x", pady=5, padx=5)
            check_groups.append(group)

        # --- Save and Cancel Buttons ---
        def on_save():
            new_name = name_var.get().strip()
            description = desc_box.get("0.0", "end-1c")
            if not new_name:
                CTkMessageBox(popup, "Error", "Agent name cannot be empty.", "red")
                return
            
            # Pass the list of CheckGroup objects to the save function
            agents.save_agent_to_disk(popup, old_folder, new_name, description, check_groups)
            
            # Refresh agent list in the main window's dropdown
            updated_agents = agents.get_agent_folders() + ["Create New"]
            self.control_panel.agent_menu.configure(values=updated_agents)
            self.control_panel.agent_var.set(new_name)
            popup.destroy()

        btn_frame = ctk.CTkFrame(popup)
        btn_frame.pack(pady=10)
        ctk.CTkButton(btn_frame, text="Save", command=on_save).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", command=popup.destroy).pack(side="left", padx=5)

    # --- Main Display Loop ---
    def update_display(self):
        try:
            # Get the latest data packet from the server in one efficient call
            state = self.client.Data.get_full_state()
            frame = state['stitched_frame']
            
            # Draw robot poses from the shared data
            for mid, pose in state['robot_poses'].items():
                cv2.circle(frame, (pose['x'], pose['y']), 12, (0, 255, 0), 2)
                cv2.putText(frame, f"ID {mid}", (pose['x'], pose['y'] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw static obstacles detected by the VLM
            for x, y, w, h in state['static_obstacles']:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)  # Magenta

            # Display the final composed frame
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ctk_img = ctk.CTkImage(light_image=img, size=(img.width, img.height))
            self.video_label.configure(image=ctk_img, text="")
        except Exception as e:
            # If the server is down, show a disconnected message
            self.video_label.configure(image=None, text=f"Disconnected from server:\n{e}")

        # Schedule the next update
        self.after(30, self.update_display)

    def on_closing(self):
        """Cleanly closes the application."""
        print("Closing application...")
        self.destroy()