import cv2

# The argument '1' specifies the second camera. 
# The first camera is typically '0'.
cap = cv2.VideoCapture(2)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera 2.")
    # Try opening the default camera if the second one fails
    print("Trying to open default camera (0)...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open any camera.")
        exit()

# Set a name for the window
window_name = 'Camera 2 Feed'
cv2.namedWindow(window_name)

while True:
    # Read a new frame from the camera
    # 'ret' will be True if the frame is read correctly
    ret, frame = cap.read()

    # If the frame was not captured correctly, break the loop
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    # Display the resulting frame in the window
    cv2.imshow(window_name, frame)

    # Wait for the 'q' key to be pressed to exit the loop
    # cv2.waitKey(1) waits for 1 millisecond
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture object
cap.release()
# Destroy all the windows OpenCV has created
cv
