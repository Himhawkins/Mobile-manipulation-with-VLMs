import cv2
import numpy as np
import time

# --- Mock rpc_system for demonstration ---
# In a real scenario, you would have the 'rpc_system' library installed.
# This mock class is included so the example code can run successfully.
# If 'rpc_system' is found, this mock will be ignored.

from rpc_system import RPCClient


def display_stitched_frame():
    """
    Connects to the RPC server, fetches the stitched frame, and displays it
    in an OpenCV window.
    """
    try:
        print("Initializing RPC client...")
        client = RPCClient()
        print("Client initialized successfully.")

        print("Fetching the full state from the RPC server...")
        # Call the method to get the dictionary containing the system state
        while 1:
            full_state = client.Data.get_full_state()
            print("State received.")

            # Check if 'stitched_frame' key exists in the response
            if 'stitched_frame' in full_state and full_state['stitched_frame'] is not None:
                # Extract the image data from the dictionary
                stitched_frame = full_state['stitched_frame']

                # Verify that the frame is a valid NumPy array before displaying
                if isinstance(stitched_frame, np.ndarray):
                    print("Displaying the stitched frame. Press any key to close the window.")
                    # Display the image in a window titled 'Stitched Frame'
                    cv2.imshow('Stitched Frame Viewer', stitched_frame)

                    # Wait indefinitely for a keyboard event (a key press)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print(f"Error: The value for 'stitched_frame' is not a valid image array. Found type: {type(stitched_frame)}.")

            else:
                print("Error: 'stitched_frame' key was not found or its value was None in the RPC response.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Ensure all OpenCV windows are closed upon exiting
        cv2.destroyAllWindows()
        print("Windows closed. Program finished.")


if __name__ == '__main__':
    
    display_stitched_frame()
