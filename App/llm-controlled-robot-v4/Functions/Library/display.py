import cv2

def display_frame(window_name='OpenCV Frame',image_name='live.jpg',folder='Data'):
    """
    Displays an OpenCV frame in a window.

    This function will show the provided frame and wait for a keypress.
    Press any key to close the window.

    Args:
        frame (np.ndarray): The image or frame (as a NumPy array) to be displayed.
        window_name (str): The name of the window in which the frame will be shown.
    """
    frame = cv2.imread(folder+'/'+image_name)
    try:
        # Check if the frame is valid
        if frame is None or frame.size == 0:
            print("Error: The provided frame is empty or invalid.")
            return

        # Show the frame in a window
        cv2.imshow(window_name, frame)

        # Wait indefinitely for a key press.
        # The window will remain open until any key is pressed.
        print(f"Displaying frame in window: '{window_name}'. Press any key to close.")
        cv2.waitKey(0)

    except cv2.error as e:
        print(f"An OpenCV error occurred: {e}")
    finally:
        # Clean up and close all OpenCV windows
        cv2.destroyAllWindows()
        print("Window closed.")

# params = {'window_name': "Camera Feed", 'image_name': "live.jpg"}
# display_frame(**params)