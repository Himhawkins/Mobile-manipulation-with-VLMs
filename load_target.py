import cv2
import numpy as np

# --- Global Variables ---
# This variable will store the latest clicked point (x, y).
# We use a list so it's mutable and can be updated by the callback function.
clicked_point = []
window_name = "Webcam Feed - Click to select a point (Press 'q' to quit)"

# --- Mouse Callback Function ---
def get_mouse_coords(event, x, y, flags, param):
    """
    This function is called whenever a mouse event occurs in the window.
    It checks for a left mouse button click, updates the global clicked_point,
    and saves the coordinates to a file.
    """
    global clicked_point
    # Check if the event is a left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Update the global variable with the new coordinates
        clicked_point = [x, y]
        print(f"Point selected: (x={x}, y={y})")

        # Open the file 'target.txt' in write mode ('w').
        # This will overwrite the file each time with the new coordinates.
        try:
            with open("target.txt", "w") as f:
                # Write the coordinates in the specified "x,y,0" format
                f.write(f"{x},{y},0\n")
            print("Coordinates saved to target.txt")
        except IOError as e:
            print(f"Error writing to file: {e}")


# --- Main Program Logic ---
def main():
    """
    Initializes the webcam, creates a window, sets the mouse callback,
    and runs the main loop to display the video feed.
    """
    global clicked_point

    # Initialize video capture from the default webcam (index 0)
    cap = cv2.VideoCapture(4)

    # Check if the webcam opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create a window to display the webcam feed
    cv2.namedWindow(window_name)

    # Set the mouse callback function for the window
    cv2.setMouseCallback(window_name, get_mouse_coords)

    print("Webcam feed started. Click on the window to select a point.")
    print("Press 'q' to quit the program.")

    # Main loop to continuously get frames from the webcam
    while True:
        # Read a new frame from the webcam
        # ret is a boolean that is True if the frame was read successfully
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # If a point has been clicked, draw on the frame
        if clicked_point:
            # Get the coordinates from the global variable
            x, y = clicked_point[0], clicked_point[1]

            # Draw a green circle at the clicked point
            # - Center: (x, y)
            # - Radius: 5 pixels
            # - Color: (0, 255, 0) which is Green in BGR format
            # - Thickness: -1 (to fill the circle)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Prepare the text to display the coordinates
            coord_text = f"({x}, {y})"

            # Put the coordinate text on the frame near the point
            # - Font: cv2.FONT_HERSHEY_SIMPLEX
            # - Scale: 0.5
            # - Color: (0, 255, 0) Green
            # - Thickness: 1
            cv2.putText(frame, coord_text, (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the resulting frame in the window
        cv2.imshow(window_name, frame)

        # Wait for a key press for 1 millisecond.
        # If the pressed key is 'q' (ASCII value), break the loop.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    # When everything is done, release the capture object
    cap.release()
    # Destroy all the windows created by OpenCV
    cv2.destroyAllWindows()
    print("Program terminated.")

if __name__ == "__main__":
    main()

