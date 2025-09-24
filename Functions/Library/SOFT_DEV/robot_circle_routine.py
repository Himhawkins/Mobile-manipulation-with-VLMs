
import sys
# Add the parent directory to the system path to allow importing sibling modules
sys.path.insert(0, '../') 

# Import functions from their respective libraries
from planning import trace_targets
from detection import detect_objects
from shapes import generate_circle_pattern

def robot_circle_routine():
    """
    This routine makes the robot go in a circle around a detected red ball.
    It uses object detection to find the red ball, generates a circular path,
    and then commands robot ID 1 to follow that path.
    """
    print("Starting robot circle routine...")

    # Step 1: Detect the red ball
    # Using a common default image path for the arena. This path might need adjustment.
    image_path = 'Data/arena_image.png'
    print(f"Detecting red ball in image: {image_path}...")
    try:
        detection_results = detect_objects(prompt_list=['red ball'], img_path=image_path)
    except Exception as e:
        print(f"Error during object detection: {e}")
        print("Please ensure the image path is correct and the detection library is set up.")
        return
    
    if not detection_results:
        print("No red ball detected. Cannot proceed with circular path.")
        return

    # Assuming the first detected object is the red ball
    red_ball_center_tuple = detection_results[0]
    red_ball_center_list = list(red_ball_center_tuple) # Convert tuple to list as required by generate_circle_pattern

    print(f"Red ball detected at coordinates: {red_ball_center_list}")

    # Step 2: Generate a circle pattern around the red ball
    circle_radius = 100 # Default radius in pixels for the circular path around the ball
    num_circle_points = 20 # Number of points to generate on the circle's circumference
    print(f"Generating a circular path with radius {circle_radius} around the red ball...")
    try:
        circle_points = generate_circle_pattern(radius=circle_radius, center=red_ball_center_list, num_points=num_circle_points)
    except Exception as e:
        print(f"Error generating circle pattern: {e}")
        return

    if not circle_points:
        print("Failed to generate circle points. The center or radius might be invalid.")
        return

    print(f"Generated {len(circle_points)} points for the circular path.")

    # Step 3: Make the robot trace the circular path
    robot_id = 1 # As specified in the request
    output_json_file = 'Data/robot_path_circle.json' # File to save the robot's path
    
    print(f"Commanding robot ID {robot_id} to trace the generated circular path...")
    try:
        trace_targets_results = trace_targets(
            input_target_list=circle_points,
            robot_id=robot_id,
            output_json_path=output_json_file,
            spacing=50,  # Dilation spacing around obstacles for safety
            delay=500,   # Delay in milliseconds applied to each target point
            offset=120,  # Offset distance to try around a blocked target
            verbose=True # Enable verbose logging during path planning
        )
        print("Path tracing process initiated. Result:")
        print(trace_targets_results)
        print(f"Robot path saved to: {output_json_file}")
    except Exception as e:
        print(f"Error during path tracing for robot {robot_id}: {e}")
        print("Please ensure 'Data/robot_pos.txt' exists and contains robot ID 1, and the planning library is configured correctly.")
        return

    print("Robot circle routine finished.")

if __name__ == '__main__':
    robot_circle_routine()
