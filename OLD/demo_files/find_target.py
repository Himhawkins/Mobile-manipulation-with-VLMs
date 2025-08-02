import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM
import torch

def save_target_center(detection, width, height, filename="target.txt"):
    """
    Calculates the center of a detected object and writes it to a file.

    The coordinates are saved in the format: center_x,center_y,0
    The file is overwritten on each call.

    Args:
        detection (dict): A single detection dictionary from the model.
        width (int): The width of the image frame for scaling coordinates.
        height (int): The height of the image frame for scaling coordinates.
        filename (str): The name of the output file.
    """
    try:
        # Scale normalized coordinates to pixel values
        x1 = int(detection['x_min'] * width)
        y1 = int(detection['y_min'] * height)
        x2 = int(detection['x_max'] * width)
        y2 = int(detection['y_max'] * height)

        # Calculate the center point of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        with open(filename, 'w') as f:
            # Write the formatted string to the file, followed by a newline
            f.write(f"{center_x},{center_y},0\n")
        print(f"Target center ({center_x},{center_y}) saved to {filename}")

    except IOError as e:
        print(f"Error writing to file {filename}: {e}")

def find_target(prompt, camera_index=4, output_filename="target.txt"):
    """
    Initializes a model, captures a single frame, finds an object based on a prompt,
    and saves the center coordinates of the first match.

    Args:
        prompt (str): The descriptive prompt for the object to detect.
        camera_index (int): The index of the camera to use for video capture.
        output_filename (str): The filename for saving the target coordinates.
    """
    # --- GPU and Model Configuration ---
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU with CUDA support.")
        print("Exiting...")
        return

    model_revision = "2025-06-21"
    device_map_config = {"": "cuda"}
    print(f"CUDA is available. Loading model to GPU with revision {model_revision}.")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision=model_revision,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device_map_config,
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Webcam Initialization ---
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return

    print("-" * 30)
    print(f"Attempting to find target: '{prompt}'.")
    
    try:
        # Capture a single frame
        ret, frame_cv = cap.read()
        if not ret:
            print("Failed to grab a frame from the camera. Exiting.")
            return

        print("Frame captured successfully. Starting detection...")
        height, width, _ = frame_cv.shape

        # Convert the OpenCV frame (BGR) to a Pillow image (RGB)
        image_rgb_cv = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb_cv)

        # Use the model to detect objects matching the prompt
        detections = model.detect(image_pil, prompt)["objects"]

        # If any objects are detected, process the first one as the target
        if detections:
            print(f"Target '{prompt}' FOUND.")
            target_object = detections[0] # Get the first detected object
            
            # Save the center point of the target
            save_target_center(target_object, width, height, output_filename)

            # Draw bounding box for visualization
            x1 = int(target_object['x_min'] * width)
            y1 = int(target_object['y_min'] * height)
            x2 = int(target_object['x_max'] * width)
            y2 = int(target_object['y_max'] * height)
            cv2.rectangle(frame_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_cv, prompt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            print(f"Target '{prompt}' NOT found in the frame.")

        # Display the resulting frame
        cv2.imshow('Target Finder', frame_cv)
        print("-" * 30)
        print("Detection complete. The program will now exit.")
        
        # Wait for 1 millisecond to allow the window to draw
        cv2.waitKey(1)

    except Exception as e:
        print(f"An error occurred during processing: {e}")

    finally:
        # When everything is done, release the capture and destroy windows
        print("Shutting down...")
        cap.release()
        cv2.destroyAllWindows()

def main():
    """
    Main function to configure and run the target finding process.
    """
    # --- Configuration ---
    # Define the object you want to find
    target_prompt = "water bottle" 
    camera_index = 4
    output_filename = "target.txt"
    # ---

    find_target(target_prompt, camera_index, output_filename)

if __name__ == "__main__":
    main()

