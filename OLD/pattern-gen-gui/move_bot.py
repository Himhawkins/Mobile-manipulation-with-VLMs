import serial
import time
import json

# --- Configuration ---
SERIAL_PORT = '/dev/ttyACM0'  # The serial port to connect to (e.g., COM3 on Windows)
BAUD_RATE = 115200            # The baud rate for the serial communication
INPUT_FILE = 'command.txt'    # The file to read the coordinates from
SEND_INTERVAL_S = 0.1         # Time to wait between sending commands, in seconds

def main_move_bot():
    """
    Main function to read from a file and send data to the serial port.
    """
    print(f"Attempting to connect to serial port {SERIAL_PORT} at {BAUD_RATE} baud...")

    try:
        # Establish the serial connection.
        # The 'with' statement ensures the port is automatically closed on exit.
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser:
            print(f"Successfully connected to {SERIAL_PORT}.")
            print("Starting to read from 'command.txt' and send data...")
            print("Press Ctrl+C to stop.")

            while True:
                try:
                    # Open and read the command file in each iteration
                    with open(INPUT_FILE, 'r') as f:
                        line = f.readline().strip()

                        # Proceed only if the line is not empty
                        if line:
                            parts = line.split(',')
                            if len(parts) == 2:
                                # Convert parts to integers
                                wx = int(parts[0].strip())
                                wy = int(parts[1].strip())
                                #wy = 180-wy
                                wx=180-wx
                                # Create the data structure (a dictionary)
                                data_to_send = {"left": wx, "right": wy}

                                # Convert the dictionary to a JSON string
                                json_string = json.dumps(data_to_send)

                                # Send the string over serial, encoded as bytes, with a newline
                                ser.write(json_string.encode('utf-8') + b'\n')
                                # print(f"Sent: {json_string}")
                            else:
                                print(f"Warning: Malformed line in {INPUT_FILE}: '{line}'")

                except FileNotFoundError:
                    print(f"Error: '{INPUT_FILE}' not found. Please create it.")
                except ValueError:
                    print(f"Error: Invalid number format in '{INPUT_FILE}'. Ensure values are integers.")
                except Exception as e:
                    print(f"An unexpected error occurred while reading the file: {e}")

                # Wait for the specified interval before the next send
                time.sleep(SEND_INTERVAL_S)

    except serial.SerialException as e:
        print(f"Error: Could not open serial port {SERIAL_PORT}.")
        print(f"Details: {e}")
        print("Please check the following:")
        print("1. Is the device connected to the correct port?")
        print("2. Do you have the necessary permissions to access the port?")
        print("3. Is the correct port name specified (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)?")
    except KeyboardInterrupt:
        print("\nProgram stopped by user. Closing serial port.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Before running, ensure you have the pyserial library installed:
    # pip install pyserial
    main_move_bot()
