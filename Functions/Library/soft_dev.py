import os

def create_code(code: str, output_file: str):
    """
    Saves a given code string to a file in the 'SOFT_DEV' subdirectory.

    This function first ensures that a directory named 'SOFT_DEV' exists in the
    current working directory, creating it if it does not. It then saves the
    provided code string into the specified output file inside that directory.

    Args:
        code (str): The string containing the Python code to be saved.
        output_file (str): The name of the file to save the code to (e.g., 'main.py').
    """
    # Define the target directory
    directory = "Functions/Library/SOFT_DEV"

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(directory, output_file)

    # Write the code string to the file
    try:
        with open(file_path, 'w') as f:
            f.write(code)
        print(f"✅ Code successfully saved to: {file_path}")
    except IOError as e:
        print(f"❌ Error saving file: {e}")

# --- Example Usage ---
