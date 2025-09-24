
import default_api

def run_robot_circular_motion_around_red_ball():
    print("Starting routine: Robot moving in a circle around a red ball.")

    # Step 1: Detect the red ball
    # Assuming 'Data/current_frame.png' is the image containing the arena and objects.
    # The default_api.detect_objects function returns a list of (x, y) coordinates.
    print("Detecting red ball in 'Data/current_frame.png'...")
    try:
        detected_objects = default_api.detect_objects(prompt_list=['red ball'], img_path='Data/current_frame.png')
    except Exception as e:
        print(f"Error during object detection: {e}")
        return

    if not detected_objects:
        print("No red ball detected. Aborting routine.")
        return

    # Assuming the first detected object is the red ball we want to orbit.
    red_ball_center = detected_objects[0]
    print(f"Red ball detected at coordinates: {red_ball_center}")

    # Step 2: Generate a circular path around the detected red ball
    # Choose a radius for the circular path.
    # num_points defines the granularity of the circle.
    radius = 100.0  # Pixels
    num_points = 36 # One point every 10 degrees for a smoother circle
    print(f"Generating a circular path with radius {radius} around {red_ball_center} with {num_points} points.")
    try:
        # Convert tuple to list for the 'center' parameter, and ensure float type
        circle_waypoints = default_api.generate_circle_pattern(radius=radius, center=[float(red_ball_center[0]), float(red_ball_center[1])], num_points=num_points)
    except Exception as e:
        print(f"Error during circle pattern generation: {e}")
        return

    if not circle_waypoints:
        print("Failed to generate circular waypoints. Aborting routine.")
        return

    print(f"Generated {len(circle_waypoints)} circular waypoints.")

    # Step 3: Plan the robot's path to follow the circular waypoints
    robot_id = 1
    output_path_file = 'Data/robot_1_red_ball_circle_path.json'
    print(f"Planning robot path for robot ID {robot_id} to follow the circular path.")
    print(f"Saving path to {output_path_file}")
    try:
        path_planning_result = default_api.trace_targets(
            input_target_list=circle_waypoints,
            robot_id=robot_id,
            output_json_path=output_path_file,
            delay=200,    # Delay in milliseconds between reaching each point
            spacing=60,   # Dilation spacing around obstacles for planning safety
            offset=100,   # Offset distance to try if a target is blocked
            verbose=True  # Print debug logs during planning
        )
        print(f"Path planning status: {path_planning_result}")
    except Exception as e:
        print(f"Error during robot path tracing: {e}")
        return

    print("Routine completed.")

# Execute the routine if the script is run directly
if __name__ == "__main__":
    run_robot_circular_motion_around_red_ball()
