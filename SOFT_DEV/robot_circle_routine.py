import json

def robot_circle_around_red_ball(robot_id: int = 1, circle_radius: float = 100.0, num_circle_points: int = 20, img_path: str = 'Data/frame.png', output_path: str = 'Data/circular_path_robot_1.json'):
    print(f"Robot {robot_id}: Starting routine to circle around a red ball.")

    # Step 1: Detect the red ball
    print("Detecting red ball...")
    detected_objects_response = default_api.detect_objects(prompt_list=['red ball'], img_path=img_path)

    if not detected_objects_response or not detected_objects_response: # The tool returns a list of tuples, so check if the list is empty
        print("No red ball detected. Exiting routine.")
        return "Failure: No red ball detected."

    # Assuming the first detected object is the red ball and it returns a list of tuples
    red_ball_center = detected_objects_response[0]
    print(f"Red ball detected at center: {red_ball_center}")

    # Step 2: Generate a circular pattern around the red ball
    print(f"Generating circular path with radius {circle_radius} around {red_ball_center}...")
    circle_points_response = default_api.generate_circle_pattern(
        radius=circle_radius,
        center=list(red_ball_center), # Convert tuple to list for the function
        num_points=num_circle_points
    )

    if not circle_points_response:
        print("Failed to generate circle pattern. Exiting routine.")
        return "Failure: Failed to generate circle pattern."

    # The generate_circle_pattern function returns a dictionary, and the points are under the 'points' key.
    circular_path_targets = circle_points_response['points']
    print(f"Generated {len(circular_path_targets)} points for the circular path.")

    # Step 3: Trace the generated circular path for the robot
    print(f"Tracing path for robot {robot_id}...")
    trace_result = default_api.trace_targets(
        input_target_list=circular_path_targets,
        robot_id=robot_id,
        output_json_path=output_path,
        spacing=50, # Using default
        delay=1000, # Using default
        offset=100 # Using default
    )

    print(f"Path tracing result: {trace_result}")
    return "Success: Robot path generated for circling the red ball."

if __name__ == "__main__":
    # Example of how to run the routine
    # Make sure 'Data/frame.png' exists or adjust img_path
    # Make sure 'Data/robot_pos.txt' contains robot_id 1
    result = robot_circle_around_red_ball(robot_id=1, circle_radius=150.0)
    print(result)
