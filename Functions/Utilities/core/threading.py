# vision_dashboard/core/threading.py

import threading

def run_in_thread(callback, on_start=None, on_complete=None):
    """Runs a function in a separate thread to keep the UI responsive."""
    def wrapper():
        if on_start:
            on_start()
        try:
            callback()
        finally:
            if on_complete:
                on_complete()
    threading.Thread(target=wrapper, daemon=True).start()