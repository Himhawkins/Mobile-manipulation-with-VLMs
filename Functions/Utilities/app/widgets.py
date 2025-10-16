# vision_dashboard/app/widgets.py

import customtkinter as ctk
from utils import ports, agents

class CTkMessageBox(ctk.CTkToplevel):
    def __init__(self, parent, title, message, color):
        super().__init__(parent)
        self.title(title)
        self.geometry("300x150")
        self.attributes("-topmost", True)
        label = ctk.CTkLabel(self, text=message, wraplength=280, text_color=color)
        label.pack(expand=True, padx=20, pady=20)
        self.after(3000, self.destroy)

class ControlPanel(ctk.CTkFrame):
    def __init__(self, parent_app):
        super().__init__(parent_app, width=350)
        self.app = parent_app
        self.pack_propagate(False)

        # --- AI Agent Selection ---
        ctk.CTkLabel(self, text="AI Agent", font=("", 16, "bold")).pack(pady=(10, 5), padx=10, anchor="w")
        agent_frame = ctk.CTkFrame(self)
        agent_frame.pack(pady=5, padx=10, fill="x")
        
        agent_folders = agents.get_agent_folders() + ["Create New"]
        self.agent_var = ctk.StringVar(value=agent_folders[0])
        self.agent_menu = ctk.CTkOptionMenu(agent_frame, variable=self.agent_var, values=agent_folders, command=self.app.on_agent_change)
        self.agent_menu.pack(side="left", expand=True, fill="x")
        ctk.CTkButton(agent_frame, text="Edit", width=50, command=self.app.on_edit_agent).pack(side="left", padx=5)

        # --- Prompts & Previews ---
        self.prompt_entry = ctk.CTkEntry(self, placeholder_text="Calibration prompt, e.g., 'black box'")
        self.prompt_entry.pack(pady=(10, 5), padx=10, fill="x")
        self.preview_box = ctk.CTkTextbox(self, state="disabled", height=200)
        self.preview_box.pack(pady=5, padx=10, expand=True, fill="both")

        # --- Action Buttons ---
        ctk.CTkLabel(self, text="Actions", font=("", 16, "bold")).pack(pady=(20, 5), padx=10, anchor="w")
        self.calibrate_button = ctk.CTkButton(self, text="Calibrate Arena", command=self.app.on_calibrate)
        self.calibrate_button.pack(pady=5, padx=10, fill="x")
        
        # --- Robot Execution Section ---
        port_frame = ctk.CTkFrame(self)
        port_frame.pack(pady=5, padx=10, fill="x")
        self.serial_var = ctk.StringVar(value="Select Port")
        self.serial_menu = ctk.CTkOptionMenu(port_frame, variable=self.serial_var, values=["Select Port"])
        self.serial_menu.pack(side="left", expand=True, fill="x")
        ctk.CTkButton(port_frame, text="⟳", width=30, command=self.refresh_ports).pack(side="left", padx=5)
        
        self.execute_button = ctk.CTkButton(self, text="Execute", fg_color="#7228E9", command=self.app.on_execute)
        self.execute_button.pack(pady=5, padx=10, fill="x", side="bottom")

        self.refresh_ports()

    def refresh_ports(self):
        vals = ports.refresh_serial_ports() or ["No Ports Found"]
        self.serial_menu.configure(values=vals)
        self.serial_var.set(vals[0])

class CheckGroup(ctk.CTkFrame):
    """A widget that displays a group of checkboxes with a title."""
    def __init__(self, parent, label, values, selected=None):
        super().__init__(parent)
        self.label = label
        self.vars = {}
        
        ctk.CTkLabel(self, text=label, font=("", 14, "bold")).pack(anchor="w", padx=10)
        
        for value in values:
            var = ctk.StringVar(value="on" if selected and value in selected else "off")
            cb = ctk.CTkCheckBox(self, text=value, variable=var, onvalue="on", offvalue="off")
            cb.pack(anchor="w", padx=20)
            self.vars[value] = var

    def get_selected(self) -> list:
        """Returns a list of the text from all checked boxes."""
        return [value for value, var in self.vars.items() if var.get() == "on"]
