import cv2
import numpy as np
import time
import os

class ImageStitcher:
    """
    A class to stitch multiple images together onto a fixed-size canvas.
    It uses ORB for feature detection and BFMatcher for feature matching.
    Homography is found using RANSAC.
    """
    def __init__(self, output_shape=(1920, 1080)):
        # Using ORB (Oriented FAST and Rotated BRIEF) as it is free to use.
        self.orb = cv2.ORB_create()
        # Using Brute-Force Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.output_width, self.output_height = output_shape

    def stitch(self, images, min_match_count=10):
        """
        Stitches a list of images together onto a pre-defined canvas.
        It calculates transformations between adjacent images and chains them together.
        
        Args:
            images (list): A list of undistorted images (numpy arrays) to be stitched.
            min_match_count (int): Minimum number of good matches to proceed with homography.

        Returns:
            numpy.ndarray: The final stitched panoramic image, or None if stitching fails.
        """
        if not images or len(images) < 2:
            print("Warning: Need at least two images to stitch.")
            # Return an empty canvas if no images are provided
            return np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

        # Calculate offset to center the first image on the canvas
        h_img0, w_img0 = images[0].shape[:2]
        x_offset = (self.output_width - w_img0) // 2
        y_offset = (self.output_height - h_img0) // 2
        
        # H_chain will hold the cumulative transformation from the *current* image to the canvas
        H_chain = np.array([[1, 0, x_offset], [0, 1, y_offset], [0, 0, 1]], dtype=np.float32)

        # Create the final canvas
        panorama = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

        # Warp the first image, which is our starting point
        cv2.warpPerspective(images[0], H_chain, (self.output_width, self.output_height), panorama, borderMode=cv2.BORDER_TRANSPARENT)

        # Process subsequent images
        for i in range(len(images) - 1):
            img_current = images[i]
            img_next = images[i+1]

            # Find keypoints and descriptors
            kp1, des1 = self.orb.detectAndCompute(img_current, None)
            kp2, des2 = self.orb.detectAndCompute(img_next, None)

            if des1 is None or des2 is None:
                print(f"Warning: Could not find descriptors for image pair {i}-{i+1}. Skipping.")
                continue

            # Match descriptors
            matches = self.bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = matches

            print(f"Found {len(good_matches)} good matches between image {i} and {i+1}.")

            if len(good_matches) > min_match_count:
                # Extract location of good matches
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find homography M that maps from img_next -> img_current
                M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    # Chain the homography: H_next = H_current @ M
                    # This gives the transform from the next image's coordinates to the canvas coordinates
                    H_chain = H_chain @ M

                    # Warp the next image onto the panorama canvas
                    cv2.warpPerspective(img_next, H_chain, (self.output_width, self.output_height), panorama, borderMode=cv2.BORDER_TRANSPARENT)
                else:
                    print(f"Warning: Homography could not be computed for image pair {i}-{i+1}.")
            else:
                print(f"Warning: Not enough matches found for image pair {i}-{i+1} - {len(good_matches)}/{min_match_count}")
        
        return panorama


def main():
    # --- Configuration ---
    CAMERA_IDS = [2, 4, 6]
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    # Define the fixed size for the output panorama
    PANORAMA_WIDTH = 1920
    PANORAMA_HEIGHT = 1080
    
    # --- Load Camera Calibration Data ---
    calibration_data = []
    for cam_id in CAMERA_IDS:
        calib_file = f"camera_calibration_data_{cam_id}.npz"
        if os.path.exists(calib_file):
            try:
                with np.load(calib_file) as data:
                    # Using the key names from your reference logic
                    mtx = data['camera_matrix']
                    dist = data['dist_coeffs']
                    calibration_data.append({'mtx': mtx, 'dist': dist})
                    print(f"Successfully loaded calibration data for camera {cam_id}.")
            except KeyError as e:
                print(f"Error: Calibration file '{calib_file}' is missing a required key: {e}.")
                # To help with debugging, check what keys are actually in the file
                with np.load(calib_file) as data:
                    print(f"       Available keys in the file are: {data.files}")
                calibration_data.append(None)
        else:
            calibration_data.append(None)
            print(f"Warning: Calibration file not found for camera {cam_id}: {calib_file}")

    # --- Initialization ---
    caps = []
    for cam_id in CAMERA_IDS:
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {cam_id}.")
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            caps.append(cap)

    if len(caps) < 2:
        print("Error: Need at least two cameras to perform stitching.")
        return

    stitcher = ImageStitcher(output_shape=(PANORAMA_WIDTH, PANORAMA_HEIGHT))
    
    print("\nStarting live stitching... Press 'q' to quit.")
    
    # --- Timing and State Variables ---
    last_stitch_time = 0
    stitch_interval = 5  # seconds
    stitched_frame = np.zeros((PANORAMA_HEIGHT, PANORAMA_WIDTH, 3), dtype=np.uint8)
    cv2.putText(stitched_frame, "Waiting for first stitch...", (50, PANORAMA_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    # --- Main Loop ---
    while True:
        frames = []
        # Read and undistort frames from all cameras
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if ret:
                # Undistort the frame if calibration data is available
                if calibration_data[i] is not None:
                    calib = calibration_data[i]
                    frame = cv2.undistort(frame, calib['mtx'], calib['dist'], None, calib['mtx'])
                frames.append(frame)
            else:
                print(f"Warning: Failed to grab frame from camera {CAMERA_IDS[i]}")
        
        # Display individual camera feeds
        for i, frame in enumerate(frames):
             cv2.imshow(f'Camera {CAMERA_IDS[i]}', frame)

        # --- Stitching Logic (every 5 seconds) ---
        if time.time() - last_stitch_time > stitch_interval and len(frames) >= 2:
            print(f"Attempting to stitch frames at {time.ctime()}...")
            last_stitch_time = time.time()

            new_stitched_frame = stitcher.stitch(frames)

            if new_stitched_frame is not None:
                stitched_frame = new_stitched_frame
                print("Stitching successful.")
            else:
                print("Stitching failed for the current set of frames.")

        # --- Display Logic (every loop) ---
        cv2.imshow('Stitched Panorama', stitched_frame)
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    print("Releasing resources...")
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


