import cv2
from cv2 import aruco
import os

# --- Define ChArUco Board Specifications ---
# These must match the settings in the calibration script
CHARUCO_SQUARES_X = 6       # How many squares wide is your board?
CHARUCO_SQUARES_Y = 4       # How many squares tall is your board?
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

# --- Define Image Generation Specifications ---
# Image size in pixels
IMG_WIDTH_PX = 2000
IMG_HEIGHT_PX = 1500
# Add a white margin around the board
MARGIN_PX = 50
FILENAME = "charuco_board.png"

def main():
    """
    Generates and saves a ChArUco board image based on the defined specifications.
    """
    print("Generating ChArUco board...")

    # Create the board object
    board = aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        squareLength=1.0,  # Proportional, actual size defined in calibration
        markerLength=0.8,  # Proportional, actual size defined in calibration
        dictionary=ARUCO_DICT
    )

    # Generate the board image
    board_image = board.generateImage(
        (IMG_WIDTH_PX, IMG_HEIGHT_PX),
        marginSize=MARGIN_PX,
        borderBits=1
    )

    # Save the image
    cv2.imwrite(FILENAME, board_image)

    print(f"Successfully generated '{FILENAME}'.")
    print("Please print this file at its actual size and attach it to a rigid, flat surface.")
    print(f"Current working directory is: {os.getcwd()}")

if __name__ == "__main__":
    main()
