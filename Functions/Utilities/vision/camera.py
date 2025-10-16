import cv2
import time
import threading

class CameraStream:
    """
    A threaded wrapper for a cv2.VideoCapture object to get the latest frame.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print(f"[ERROR] Could not open camera {src}.")
            return
            
        # Read the first frame and get its properties
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        
        # Start the thread to read frames from the video stream
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Thread will die with the main program
        self.thread.start()

    def update(self):
        """The target function for the reading thread."""
        while not self.stopped:
            if not self.stream.isOpened():
                self.stop()
                continue
            self.ret, self.frame = self.stream.read()

    def read(self):
        """Return the most recent frame."""
        return self.frame

    def stop(self):
        """Stop the thread and release camera resources."""
        self.stopped = True
        # Wait for thread to finish
        self.thread.join(timeout=1) 
        if self.stream.isOpened():
            self.stream.release()

class CameraManager:
    """Manages all camera hardware using threaded streams to prevent lag."""
    def __init__(self, settings: dict):
        self.streams = self._initialize_cameras(settings)
        if not self.streams:
            print("[WARN] CameraManager could not open any cameras.")

    def get_frame(self, cam_id: int):
        """Gets the latest frame from the specified camera stream."""
        stream = self.streams.get(cam_id)
        # The read() method in our class now returns the latest frame
        return stream.read() if stream else None

    def get_open_streams(self) -> dict:
        return self.streams

    def release(self):
        """Stops all camera threads and releases resources."""
        for stream in self.streams.values():
            stream.stop()
        cv2.destroyAllWindows()
        
    def __del__(self):
        self.release()

    def _initialize_cameras(self, settings, width=1280, height=720, fps=30):
        """Initializes cameras using the threaded CameraStream class."""
        needed_ids = sorted(list(set(
            int(cell["camera"]) for cell in settings.get("cells", {}).values() if "camera" in cell
        )))
        
        streams = {}
        for cam_id in needed_ids:
            # First, try to set properties on a temporary capture object
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened():
                print(f"[WARN] Camera {cam_id} failed to open.")
                continue
            
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.release() # Release the temporary object
            time.sleep(0.1) # Give time for settings to apply

            # Now create the threaded stream
            stream = CameraStream(cam_id)
            # Check if the stream's internal capture object is valid
            if hasattr(stream, 'stream') and stream.stream.isOpened():
                 streams[cam_id] = stream
                 print(f"[INFO] Threaded stream for camera {cam_id} started successfully.")
            else:
                print(f"[WARN] Failed to start threaded stream for camera {cam_id}.")

        return streams