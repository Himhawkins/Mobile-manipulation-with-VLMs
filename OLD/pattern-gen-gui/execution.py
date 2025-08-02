import math
import time
import os

# --- File-Based Robot Interface ---
# These functions interact with text files to get robot state and send commands.

def initialize_files():
    """
    Creates initial state and target files if they don't exist.
    This allows the controller to run out-of-the-box with a sample pattern.
    """
    # Create a target file with a square pattern if it doesn't exist
    if not os.path.exists('pattern_target.txt'):
        print("Creating default pattern_target.txt file.")
        with open('pattern_target.txt', 'w') as f:
            f.write(f"5.0,0.0,{math.pi / 2}\n")  # Go to (5,0), face up
            f.write(f"5.0,5.0,{math.pi}\n")      # Go to (5,5), face left
            f.write(f"0.0,5.0,{-math.pi / 2}\n") # Go to (0,5), face down
            f.write(f"0.0,0.0,0.0\n")            # Go to (0,0), face right

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

def read_targets_from_file():
    """Reads all target destinations from pattern_target.txt into a list."""
    targets = []
    try:
        with open('pattern_target.txt', 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                x = float(parts[0])
                y = float(parts[1])
                theta = float(parts[2])
                targets.append((x, y, theta))
    except (IOError, IndexError, ValueError) as e:
        print(f"Error reading pattern_target.txt: {e}. Using no targets.")
    return targets

# --- Closed-Loop Controller ---

def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))

def adjust_wheel_speed(speed):
    """
    Applies custom non-linear adjustments to a wheel speed value.
    - Creates a dead zone around 90.
    - Increments/decrements values in specific ranges.
    """
    if 89.75 <= speed <= 90.25:
        # If the speed is very close to 90, set it exactly to 90
        return 90.0
    elif 90.25 < speed <= 95:
        # If the speed is in the range (90.25, 95], increment it by 1
        return speed + 1.0
    elif 85 <= speed < 89.75:
        # If the speed is in the range [85, 89.75), decrement it by 1
        return speed - 1.0
    else:
        # If the speed is outside these ranges, return it unchanged
        return speed

def main_controller():
    """
    Main closed-loop control function to drive the robot through a sequence of targets.
    """
    # Create necessary files with default values on first run
    initialize_files()

    # --- PID Controller Gains (Tuned for gentler motion) ---
    Kp_distance = 0.2
    Kp_angle = 4.0
    Ki_angle = 0.07
    Kd_angle = 0.7

    # --- Tolerances ---
    # How close the robot needs to be to a waypoint to consider it "reached"
    distance_tolerance = 15  # meters
    final_angle_tolerance = 0.1  # radians (about 6 degrees)

    # --- PID State Variables ---
    integral_angle = 0.0
    last_angle_error = 0.0
    last_time = time.time()

    # --- Read all targets from the file ---
    all_targets = read_targets_from_file()
    if not all_targets:
        print("No targets found in pattern_target.txt. Exiting.")
        return

    current_target_index = 0
    print(f"Loaded {len(all_targets)} targets.")

    # --- Main Control Loop ---
    while current_target_index < len(all_targets):
        # --- Get the current target from the list ---
        target_x, target_y, target_theta = all_targets[current_target_index]

        # --- Time Step Calculation for PID ---
        current_time = time.time()
        dt = current_time - last_time
        if dt == 0:
            time.sleep(0.01)
            continue
        last_time = current_time

        # 1. Read current state
        current_x, current_y, current_theta = read_pos()

        # 2. Calculate Errors
        distance_error = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        angle_to_target = math.atan2(target_y - current_y, target_x - current_x)

        # --- Check if the current target waypoint is reached ---
        if distance_error < distance_tolerance:
            print(f"\nWaypoint {current_target_index + 1} reached!")
            current_target_index += 1
            # Reset PID integrals to prevent windup when switching targets
            integral_angle = 0.0
            last_angle_error = 0.0
            # If that was the last target, break the loop
            if current_target_index >= len(all_targets):
                break
            else:
                # Briefly pause before moving to the next target
                move(90, 90)
                time.sleep(0.5)
                continue

        # --- Control Logic ---
        angle_error_to_point = normalize_angle(angle_to_target - current_theta)

        # Decide whether to move forward or backward
        direction = 1.0
        # If the target is behind the robot, it's more efficient to reverse
        if abs(angle_error_to_point) > (math.pi / 2):
            angle_error_for_steering = normalize_angle(angle_error_to_point - math.pi)
            direction = -1.0
        else:
            angle_error_for_steering = angle_error_to_point
            direction = 1.0

        # Only apply significant linear velocity when reasonably aligned
        add_lin = 0
        if math.degrees(abs(angle_error_for_steering)) <= 5:    #25
            add_lin = 1

        # 3. PID Calculation for Angle
        integral_angle += angle_error_for_steering * dt
        derivative_angle = (angle_error_for_steering - last_angle_error) / dt
        last_angle_error = angle_error_for_steering

        angular_control = (Kp_angle * angle_error_for_steering) + \
                          (Ki_angle * integral_angle) + \
                          (Kd_angle * derivative_angle)

        # 4. Proportional Control for Distance
        linear_control = Kp_distance * distance_error
        max_linear_velocity = 15
        linear_control = max(-max_linear_velocity, min(max_linear_velocity, linear_control))

        # 5. Convert control signals to wheel velocities
        base_speed = 90  # Stop speed
        left_wheel_speed = base_speed + (add_lin * direction * linear_control) - angular_control
        right_wheel_speed = base_speed + (add_lin * direction * linear_control) + angular_control
        
        # *** NEW: Apply the custom adjustment logic to each wheel speed ***
        left_wheel_speed = adjust_wheel_speed(left_wheel_speed)
        right_wheel_speed = adjust_wheel_speed(right_wheel_speed)

        # 6. Clamp wheel velocities to the allowed range [70, 110]
        left_wheel_speed = max(70, min(110, left_wheel_speed))
        right_wheel_speed = max(70, min(110, right_wheel_speed))

        # 7. Send commands to the robot
        move(left_wheel_speed, right_wheel_speed)
        log_error(distance_error, angle_error_for_steering)
        
        # --- Logging ---
        print(f"Target {current_target_index+1}:({target_x:.1f},{target_y:.1f}) "
              f"DistErr:{distance_error:.2f} AngleErr:{math.degrees(angle_error_for_steering):.1f} "
              f"Cmd:(L:{left_wheel_speed:.1f},R:{right_wheel_speed:.1f})", end='\r')

        time.sleep(0.05)

    # --- End of all targets ---
    print("\nAll targets reached! Finalizing position.")
    move(90, 90)
    
    # After reaching the last waypoint, turn to the final desired orientation
    # This block is optional but good for precise final alignment
    _, _, final_target_theta = all_targets[-1]
    while True:
        current_x, current_y, current_theta = read_pos()
        final_orientation_error = normalize_angle(final_target_theta - current_theta)
        if abs(final_orientation_error) < final_angle_tolerance:
            break
        
        turn_speed = 20 * final_orientation_error
        move(90 - turn_speed, 90 + turn_speed)
        time.sleep(0.05)
    
    print("Pattern complete. Stopping robot.")
    move(90, 90) # Send final stop command


if __name__ == '__main__':
    main_controller()