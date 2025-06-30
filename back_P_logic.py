import math
import time
import os

# --- File-Based Robot Interface ---
# These functions interact with text files to get robot state and send commands.

def initialize_files():
    """
    Creates initial state and target files if they don't exist.
    This allows the controller to run out-of-the-box.
    """
    # Create a target file if it doesn't exist
    if not os.path.exists('target.txt'):
        print("Creating default target.txt file.")
        with open('target.txt', 'w') as f:
            # Default Target: x=5, y=5, theta=90 degrees (pi/2 radians)
            f.write(f"5.0,5.0,{math.pi / 2}")

    # Create a robot position file if it doesn't exist
    if not os.path.exists('robot_pos.txt'):
        print("Creating default robot_pos.txt file.")
        with open('robot_pos.txt', 'w') as f:
            # Default Initial Position: x=0, y=0, theta=0
            f.write("0.0,0.0,0.0")

    # Create a command file if it doesn't exist
    if not os.path.exists('command.txt'):
        print("Creating default command.txt file.")
        with open('command.txt', 'w') as f:
            # Default command: stop
            f.write("90,90")

def move(left_wheel_velocity, right_wheel_velocity):
    """
    Writes the calculated wheel velocities (0-180) to command.txt.
    """
    try:
        with open('command.txt', 'w') as f:
            f.write(f"{int(left_wheel_velocity)},{int(right_wheel_velocity)}")
        # This print statement is now more focused, as current pose is printed in the loop
        # print(f"Writing to command.txt: Left Vel: {left_wheel_velocity:.2f}, Right Vel: {right_wheel_velocity:.2f}")
    except IOError as e:
        print(f"Error: Could not write to command.txt: {e}")

def read_pos():
    """Reads the current robot position (x, y, theta) from robot_pos.txt."""
    try:
        with open('robot_pos.txt', 'r') as f:
            line = f.readline().strip()
            parts = line.split(',')
            x = float(parts[0])
            y = float(parts[1])
            theta = float(parts[2])
            return x, y, theta
    except (IOError, IndexError, ValueError) as e:
        print(f"Error reading robot_pos.txt: {e}. Using (0,0,0).")
        return 0.0, 0.0, 0.0

def read_target():
    """Reads the target destination (x, y, theta) from target.txt."""
    try:
        with open('target.txt', 'r') as f:
            line = f.readline().strip()
            parts = line.split(',')
            x = float(parts[0])
            y = float(parts[1])
            theta = float(parts[2])
            return x, y, theta
    except (IOError, IndexError, ValueError) as e:
        print(f"Error reading target.txt: {e}. Using (0,0,0).")
        return 0.0, 0.0, 0.0

# --- Closed-Loop Controller ---

def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))

def main_controller():
    """
    Main closed-loop control function to drive the robot to the target.
    """
    # Create necessary files with default values on first run
    initialize_files()

    # --- Controller Gains ---
    # These are crucial for performance. You will need to tune them.
    Kp_distance = 0.08#0.5 # Proportional gain for distance error
    Kp_angle =4  #20   # Proportional gain for angle error

    # --- Tolerances ---
    # How close the robot needs to be to the target to stop.
    distance_tolerance = 3.0  # meters
    angle_tolerance = 0.1     # radians (about 5.7 degrees)

    # Initial read of the target
    target_x, target_y, target_theta = read_target()
    print(f"Initial Target: (x={target_x:.2f}, y={target_y:.2f}, th={math.degrees(target_theta):.2f})")

    # Main loop for approaching the target location
    while True:
        # 1. Read current state from sensors (via file)
        current_x, current_y, current_theta = read_pos()
        # Read target dynamically inside the loop
        target_x, target_y, target_theta = read_target()

        # Print current robot state
        print(f"Current Pose: (x={current_x:.2f}, y={current_y:.2f}, th={math.degrees(current_theta):.2f})", end=" | ")

        # 2. Calculate errors
        distance_error = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        print(f"Dist Err: {distance_error:.2f}", end=" | ")

        if distance_error < distance_tolerance:
            print("\nDistance tolerance reached. Focusing on final angle.")
            break # Move to final angle alignment

        angle_to_target = math.atan2(target_y - current_y, target_x - current_x)
        angle_error = normalize_angle(angle_to_target - current_theta)

        # --- FORWARD/BACKWARD LOGIC ---
        direction = 1.0
        # If the target is behind the robot (more than 90 degrees away from the front)
        if abs(angle_error) > (math.pi / 2):
            # It's more efficient to move backward
            direction = -1.0
            # Adjust angle_error to point the BACK of the robot to the target
            angle_error = normalize_angle(angle_to_target - math.pi - current_theta)
        
        print(f"Angle Err: {math.degrees(angle_error):.2f}", end=" -> ")

        # 3. Calculate control signals (Proportional Control)
        linear_control = Kp_distance * distance_error
        angular_control = Kp_angle * angle_error

        # 4. Convert control signals to wheel velocities
        base_speed = 90  # Stop speed
        # Apply direction to linear control
        left_wheel_speed = base_speed + (direction * linear_control) - angular_control
        right_wheel_speed = base_speed + (direction * linear_control) + angular_control

        # 5. Clamp wheel velocities to the allowed range [0, 180] to allow full motion
        left_wheel_speed = max(70, min(110, left_wheel_speed))
        right_wheel_speed = max(70, min(110, right_wheel_speed))

        # 6. Send commands to the robot (via file)
        move(left_wheel_speed, right_wheel_speed)
        print(f"Cmd: (L:{left_wheel_speed:.2f}, R:{right_wheel_speed:.2f})")


        # Loop delay
        time.sleep(0.1)

    # Final Angle Alignment loop
    print("Aligning to final target orientation.")
    while True:
        current_x, current_y, current_theta = read_pos()
        # We need the target theta from the file for this loop as well
        _, _, target_theta = read_target()
        
        final_angle_error = normalize_angle(target_theta - current_theta)
        
        print(f"Final Align: Current th: {math.degrees(current_theta):.2f}, Target th: {math.degrees(target_theta):.2f}, Error: {math.degrees(final_angle_error):.2f}")

        if abs(final_angle_error) < angle_tolerance:
            print("Target reached!")
            move(90, 90) # Send stop command
            break

        angular_control = Kp_angle * final_angle_error

        left_wheel_speed = 90 - angular_control
        right_wheel_speed = 90 + angular_control

        # Clamp wheel velocities to the full range [0, 180] for turning in place
        left_wheel_speed = max(0, min(180, left_wheel_speed))
        right_wheel_speed = max(0, min(180, right_wheel_speed))

        move(left_wheel_speed, right_wheel_speed)
        time.sleep(0.1)


if __name__ == '__main__':
    main_controller()

