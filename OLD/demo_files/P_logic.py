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
        with open('star_target.txt', 'w') as f:
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
    except IOError as e:
        print(f"Error: Could not write to command.txt: {e}")
def log_error(dist_error, angle_error):
    """Writes the current distance and angle errors to error.txt."""
    try:
        with open('error.txt', 'w') as f:
            f.write(f"{dist_error},{angle_error}")
    except IOError as e:
        print(f"Error: Could not write to error.txt: {e}")
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
        with open('star_target.txt', 'r') as f:
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
    Main closed-loop control function to drive the robot to the target using PID.
    """
    # Create necessary files with default values on first run
    initialize_files()

    # --- PID Controller Gains (Tuned for gentler motion) ---
    # Proportional gains
    Kp_distance = 0.2
    Kp_angle = 4.0
    # Integral gains
    Ki_angle = 0.07
    # Derivative gains
    Kd_angle = 0.7

    # --- Tolerances ---
    distance_tolerance = 0.1  # meters
    angle_tolerance = 0.05    # radians (about 3 degrees)

    # --- PID State Variables ---
    integral_angle = 0.0
    last_angle_error = 0.0
    last_time = time.time()

    # --- Initial Target Read ---
    target_x, target_y, target_theta = read_target()
    print(f"Initial Target: (x={target_x:.2f}, y={target_y:.2f}, th={math.degrees(target_theta):.2f})")

    # --- Main Control Loop ---
    while True:
        # --- Time Step Calculation for PID ---
        current_time = time.time()
        dt = current_time - last_time
        if dt == 0:  # Avoid division by zero
            time.sleep(0.01)
            continue
        last_time = current_time

        # 1. Read current state and target
        current_x, current_y, current_theta = read_pos()
        target_x, target_y, target_theta = read_target()

        # 2. Calculate Errors
        distance_error = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        angle_to_target = math.atan2(target_y - current_y, target_x - current_x)

        # --- Check for Goal Completion ---
        final_orientation_error = normalize_angle(target_theta - current_theta)
        if distance_error < distance_tolerance and abs(final_orientation_error) < angle_tolerance:
            print("\nTarget Reached!")
            move(90, 90) # Send stop command
            break

        # --- Combined Control Logic ---
        angle_error_to_point = normalize_angle(angle_to_target - current_theta)

        # Decide whether to move forward or backward
        direction = 1.0
        if abs(angle_error_to_point) > (math.pi / 2):
            direction = 1.0
            angle_error_for_steering = angle_error_to_point #normalize_angle(angle_to_target - math.pi - current_theta)
        else:
            angle_error_for_steering = angle_error_to_point
        add_lin=0
        if math.degrees(abs(angle_error_for_steering)) <=15:
            add_lin=1
        # 3. PID Calculation for Angle
        integral_angle += angle_error_for_steering * dt
        derivative_angle = (angle_error_for_steering - last_angle_error) / dt
        last_angle_error = angle_error_for_steering

        angular_control = (Kp_angle * angle_error_for_steering) + \
                          (Ki_angle * integral_angle) + \
                          (Kd_angle * derivative_angle)

        # 4. Proportional Control for Distance (with capping)
        linear_control = Kp_distance * distance_error
        # Cap linear velocity contribution to leave room for steering within the 70-110 range
        max_linear_velocity = 15 
        linear_control = max(-max_linear_velocity, min(max_linear_velocity, linear_control))

        # 5. Convert control signals to wheel velocities
        base_speed = 90  # Stop speed
        left_wheel_speed = base_speed + (add_lin*direction * linear_control) - angular_control
        right_wheel_speed = base_speed + (add_lin*direction * linear_control) + angular_control

        # 6. Clamp wheel velocities to the allowed range [70, 110]
        left_wheel_speed = max(70, min(110, left_wheel_speed))
        right_wheel_speed = max(70, min(110, right_wheel_speed))

        # 7. Send commands to the robot
        move(left_wheel_speed, right_wheel_speed)
        log_error(distance_error, angle_error_for_steering)
        # --- Logging ---
        print(f"Pose:({current_x:.1f},{current_y:.1f},{math.degrees(current_theta):.1f}) "
              f"DistErr:{distance_error:.2f} AngleErr:{math.degrees(angle_error_for_steering):.1f} "
              f"Cmd:(L:{left_wheel_speed:.1f},R:{right_wheel_speed:.1f})", end='\r')

        time.sleep(0.05)


if __name__ == '__main__':
    main_controller()

