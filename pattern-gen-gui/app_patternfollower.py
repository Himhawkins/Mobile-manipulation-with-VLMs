import cv2
import cv2.aruco as aruco
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
# Assuming pattern_gen_llm.py and execution.py/move_bot.py are in the same directory
from pattern_gen_llm import generate_path
import threading

# --- Import the functions to be threaded ---
from execution import main_controller
from move_bot import main_move_bot

MARKER_ID_TO_TRACK = 782
MARKER_DICT = aruco.DICT_ARUCO_ORIGINAL
CAMERA_INDEX = 0
OUTPUT_FILENAME = "robot_pos.txt"

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Virtual Camera Viewer")
        self.root.geometry("800x600")
        self.root.configure(bg='black')

        self.video_label = tk.Label(root, bg='black')
        self.video_label.pack(fill="both", expand=True)

        self.start_button = tk.Button(
            root, text="Start Camera", command=self.start_camera,
            font=("Helvetica", 16), padx=20, pady=10
        )
        self.start_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.input_label = tk.Label(root, text="Type what shape you want the bot to follow:",
                                    font=("Helvetica", 14), bg='black', fg='white')
        self.input_textbox = tk.Text(root, font=("Helvetica", 14), width=60, height=4, wrap=tk.WORD)
        
        # --- Create a frame to hold the action buttons ---
        self.button_frame = tk.Frame(root, bg='black')

        # --- Submit button for shape generation ---
        self.submit_button = tk.Button(self.button_frame, text="Submit Shape", font=("Helvetica", 12),
                                       command=self.submit_shape)
        self.submit_button.pack(side=tk.LEFT, padx=10)

        # --- New button to start the execution controller ---
        self.execution_button = tk.Button(self.button_frame, text="Start Execution", font=("Helvetica", 12),
                                          command=self.start_execution)
        self.execution_button.pack(side=tk.LEFT, padx=10)

        self.running = False
        self.cap = None
        self.frame = None # Initialize frame to None
        self.execution_thread = None # Store the thread object
        self.execution_in_progress = False # Flag to track if execution is currently running

    def start_camera(self):
        print("Starting background services...")
        # The 'daemon=True' argument ensures this thread will exit when the main app closes
        thread3 = threading.Thread(target=main_move_bot, daemon=True)
        thread3.start()
        print("Move bot service running.")

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        aruco_dict = aruco.getPredefinedDictionary(MARKER_DICT)
        parameters = aruco.DetectorParameters()

        self.detector = aruco.ArucoDetector(aruco_dict, parameters)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open /dev/video{CAMERA_INDEX}.")
            return

        self.start_button.destroy()
        self.running = True

        # Show the input widgets and the button frame
        self.input_label.pack(pady=(10, 0))
        self.input_textbox.pack(pady=(0, 10))
        self.button_frame.pack(pady=(0, 20))

        self.update_frame()
        
    def start_execution(self):
        """Starts the main_controller in a background thread, ensuring only one instance runs at a time."""
        if self.execution_in_progress:
            messagebox.showinfo("Info", "Execution controller is already running.")
            return

        print("Starting execution controller...")
        # Disable the button and update text
        self.execution_button.config(state=tk.DISABLED, text="Executing...")
        self.execution_in_progress = True # Set the flag to True

        # Create and start the thread
        self.execution_thread = threading.Thread(target=main_controller, daemon=True)
        self.execution_thread.start()
        print("Execution controller thread started.")
        
        # Start monitoring the thread's completion
        self.root.after(100, self.check_execution_thread_status)

    def check_execution_thread_status(self):
        """
        Periodically checks if the execution thread is still alive.
        Re-enables the button when the thread completes.
        """
        if self.execution_thread and self.execution_thread.is_alive():
            # If the thread is still running, check again after a delay
            self.root.after(100, self.check_execution_thread_status)
        else:
            # Thread has finished (or never started/already finished)
            print("Execution controller thread finished.")
            self.execution_button.config(state=tk.NORMAL, text="Start Execution")
            self.execution_in_progress = False
            self.execution_thread = None # Clear the reference to the finished thread

    def update_frame(self):
        if self.running and self.cap and self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if ret:
                frame_copy = self.frame.copy()
                self.corners, self.ids, _ = self.detector.detectMarkers(frame_copy)
                self.detect_aruco_bot(frame_copy)

                img = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img_pil)

                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.root.after(10, self.update_frame)

    def detect_aruco_bot(self, frame_to_draw_on):
        if self.ids is not None:
            for i, marker_id in enumerate(self.ids.flatten()):
                if marker_id == MARKER_ID_TO_TRACK:
                    pts = self.corners[i][0]
                    cx = int(pts[:, 0].mean())
                    cy = int(pts[:, 1].mean())
                    top_left, top_right = pts[0], pts[1]
                    angle_rad = np.arctan2(top_right[1] - top_left[1], top_right[0] - top_left[0]) + (np.pi / 2)

                    try:
                        with open(OUTPUT_FILENAME, "w") as f:
                            f.write(f"{cx},{cy},{angle_rad}")
                    except IOError as e:
                        print(f"Error writing to file {OUTPUT_FILENAME}: {e}")

                    cv2.polylines(frame_to_draw_on, [pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.circle(frame_to_draw_on, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame_to_draw_on, f"ID:{marker_id} ({cx},{cy})", (cx + 10, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame_to_draw_on, f"Angle: {angle_rad:.2f} rad", (cx + 10, cy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (238, 244, 44), 2)

    def submit_shape(self):
        shape = self.input_textbox.get("1.0", tk.END).strip().lower()
        if not shape:
            messagebox.showwarning("Input Required", "Please type a shape before submitting.")
            return

        self.submit_button.config(state=tk.DISABLED, text="Generating...")
        thread = threading.Thread(target=self.run_path_generation, args=(shape,), daemon=True)
        thread.start()

    def run_path_generation(self, shape):
        if not self.cap or not self.cap.isOpened():
            self.root.after(0, self.show_error, "Camera is not available.")
            return

        # Use the latest available frame for path generation
        if self.frame is None:
            self.root.after(0, self.show_error, "No frame captured from camera yet.")
            return
            
        pattern_frame = self.frame.copy()
        filepath = "temp_camera_frame.jpg"
        cv2.imwrite(filepath, pattern_frame)
        
        # IMPORTANT: Replace with your actual API key or use a secure method to load it
        api_key = "AIzaSyCrsKecGWsLorr0tEmBXUjmBhu9xLCkPs0" # Placeholder API key

        try:
            generate_path(filepath, shape, api_key)
            self.root.after(0, messagebox.showinfo, "Done", "Path generated and visualized.")
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Error", f"Something went wrong:\n{str(e)}")
        finally:
            # Use root.after to ensure this runs on the main GUI thread
            self.root.after(0, self.reset_submit_button)
    
    def reset_submit_button(self):
        self.submit_button.config(state=tk.NORMAL, text="Submit Shape")

    def show_error(self, message):
        messagebox.showerror("Error", message)
        self.reset_submit_button()

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
