import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM
import torch

def dump_result(detections, width, height, filename="obstacles.txt"):
    """
    Writes the bounding box coordinates of detections to a text file.

    Each line in the file will be in the format: xmin,ymin,xmax,ymax
    The file is overwritten on each call with the latest detections.

    Args:
        detections (list): A list of detection dictionaries from the model.
        width (int): The width of the image frame for scaling coordinates.
        height (int): The height of the image frame for scaling coordinates.
        filename (str): The name of the output file.
    """
    try:
        with open(filename, 'w') as f:
            for obj in detections:
                # Scale normalized coordinates to pixel values
                x1 = int(obj['x_min'] * width)
                y1 = int(obj['y_min'] * height)
                x2 = int(obj['x_max'] * width)
                y2 = int(obj['y_max'] * height)
                # Write the formatted string to the file, followed by a newline
                f.write(f"{x1},{y1},{x2},{y2}\n")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")

def find_obstacles(object_prompt, camera_index, output_filename):
    """
    Initializes the model, captures a single frame, performs object detection,
    and exits automatically.

    Args:
        object_prompt (str): The descriptive prompt for the object to detect.
        camera_index (int): The index of the camera to use for video capture.
        output_filename (str): The filename for saving detection coordinates.
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
    print(f"Attempting to capture one frame to detect: '{object_prompt}'.")
    
    try:
        # Capture a single frame
        ret, frame_cv = cap.read()
        if not ret:
            print("Failed to grab a frame from the camera. Exiting.")
            cap.release() # Release camera before exiting
            return

        print("Frame captured successfully. Starting detection...")
        height, width, _ = frame_cv.shape

        # Convert the OpenCV frame (BGR) to a Pillow image (RGB)
        image_rgb_cv = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb_cv)

        # Use the model to detect objects
        detections = model.detect(image_pil, object_prompt)["objects"]

        if detections:
            print(f"Found {len(detections)} object(s). Writing results to '{output_filename}'.")
            dump_result(detections, width, height, output_filename)

            for obj in detections:
                x1 = int(obj['x_min'] * width)
                y1 = int(obj['y_min'] * height)
                x2 = int(obj['x_max'] * width)
                y2 = int(obj['y_max'] * height)

                cv2.rectangle(frame_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_cv, object_prompt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            print("No objects matching the prompt were detected in the frame.")

        # Display the resulting frame
        cv2.imshow('Object Detection Result', frame_cv)
        print("-" * 30)
        print("Detection complete. The program will now exit.")
        
        # Wait for 1 millisecond. This allows the window to be drawn
        # before the script continues to the finally block and closes it.
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
    Main function to configure and run the object detection.
    """
    # --- Configuration ---
    object_prompt = "obstacles black rectangles solid"
    camera_index = 4
    output_filename = "obstacles.txt"
    # ---

    find_obstacles(object_prompt, camera_index, output_filename)

if __name__ == "__main__":
    main()
