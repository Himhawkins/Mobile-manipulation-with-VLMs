import cv2

# Open camera
cap = cv2.VideoCapture(6)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Add 20 px black border on all sides
    bordered = cv2.copyMakeBorder(frame, top=40, bottom=40, left=40, right=40, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Show result
    cv2.imshow("Bordered Frame", bordered)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
