import os
import json

def load_agent_description(agent_name):
    """
    Loads an agent's description and function details from its dedicated folder.

    This function searches for a directory with the given agent_name inside an 'Agents'
    folder. It reads the agent's background from 'description.txt' and
    the associated function names and their libraries from 'functions.json'.

    Args:
        agent_name (str): The name of the agent to load. This should correspond
                          to a folder within the 'Agents' directory.

    Returns:
        tuple: A tuple containing:
            - background (str): The description of the agent.
            - functions_list (list): A list of strings, where each string
                                     details a function's name and its library.
    Returns:
        (None, None) if the agent folder, description file, or functions
        file cannot be found.
    """
    # Define the base path for the 'Agents' folder
    agents_folder_path = "Agents"
    agent_path = os.path.join(agents_folder_path, agent_name)

    # --- 1. Check if the agent's folder exists ---
    if not os.path.isdir(agent_path):
        print(f"Error: Agent folder '{agent_name}' not found at '{agent_path}'")
        return None, None

    # --- 2. Read the description.txt file ---
    description_file_path = os.path.join(agent_path, "description.txt")
    background = ""
    try:
        with open(description_file_path, 'r') as f:
            background = f.read()
    except FileNotFoundError:
        print(f"Error: 'description.txt' not found for agent '{agent_name}'")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading '{description_file_path}': {e}")
        return None, None

    # --- 3. Read the functions.json file ---
    functions_file_path = os.path.join(agent_path, "functions.json")
    functions_list = []
    try:
        with open(functions_file_path, 'r') as f:
            functions_data = json.load(f)
            # Process each function entry in the JSON file
            for func in functions_data:
                library = func.get("library")
                name = func.get("name")
                if library and name:
                    functions_list.append(f"Library: {library}, Name: {name}")
                else:
                    print(f"Warning: Skipping malformed function entry in '{functions_file_path}': {func}")
    except FileNotFoundError:
        print(f"Error: 'functions.json' not found for agent '{agent_name}'")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{functions_file_path}'")
        return None, None
    except Exception as e:
        print(f"An error occurred while processing '{functions_file_path}': {e}")
        return None, None

    return background, functions_list

def OLD_load_functions_description(function_list):
    """
    Loads detailed descriptions for a specific list of functions.

    This function reads from JSON files named after libraries within the
    'Functions/description' folder to retrieve detailed information about
    each function specified in the input list.

    Args:
        function_list (list): A list of strings from load_agent_description,
                              e.g., ["Library: shapes, Name: generate_shape"].

    Returns:
        dict: A dictionary where keys are function names and values are dicts
              containing the function's full description, inputs, output, etc.
              Returns an empty dictionary if no functions can be loaded.
    """
    functions_details = {}
    base_path = "Functions/description"

    if not function_list:
        return functions_details

    # Group function names by library to avoid opening the same file multiple times
    libraries_to_load = {}
    for func_string in function_list:
        try:
            # Parse the string "Library: shapes, Name: generate_shape"
            parts = func_string.split(', ')
            library = parts[0].replace('Library: ', '').strip()
            name = parts[1].replace('Name: ', '').strip()

            if library not in libraries_to_load:
                libraries_to_load[library] = []
            libraries_to_load[library].append(name)
        except IndexError:
            print(f"Warning: Skipping malformed function string: '{func_string}'")
            continue

    # Read each library file and extract the required function descriptions
    for library, names in libraries_to_load.items():
        file_path = os.path.join(base_path, f"{library}.json")
        try:
            with open(file_path, 'r') as f:
                library_data = json.load(f)
                for name in names:
                    if name in library_data:
                        functions_details[name] = library_data[name]
                    else:
                        print(f"Warning: Function '{name}' not found in library '{library}'.")
        except FileNotFoundError:
            print(f"Error: Library description file not found at '{file_path}'")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{file_path}'")
        except Exception as e:
            print(f"An error occurred while processing '{file_path}': {e}")

    return functions_details


def load_functions_description(function_list):
    """
    Loads detailed descriptions for a specific list of function names.

    This function searches through all JSON files in the 'Functions/description'
    directory to find and retrieve detailed information about each function
    specified in the input list. It expects JSON files to contain either a
    single function object or a list of function objects, where each object
    has a "name" key.

    Args:
        function_list (list): A list of function names (strings),
                              e.g., ["trace_targets", "another_function"].

    Returns:
        list: A list containing the full JSON objects for each function found.
              Returns an empty list if no functions are found.
    """
    # The return value is now a list of the found function objects.
    functions_details = []
    base_path = "Functions/description"

    if not function_list:
        return functions_details

    # Use a set for efficient lookup and removal of found functions.
    
    libraries_to_load = {}
    for func_string in function_list:
        try:
            # Parse the string "Library: shapes, Name: generate_shape"
            parts = func_string.split(', ')
            library = parts[0].replace('Library: ', '').strip()
            name = parts[1].replace('Name: ', '').strip()

            if library not in libraries_to_load:
                libraries_to_load[library] = []
            libraries_to_load[library].append(name)
        except IndexError:
            print(f"Warning: Skipping malformed function string: '{func_string}'")
            continue
    
    functions_to_find = []#set(function_list)
    print('dict::::')
    print(libraries_to_load)
    # Exit early if the base directory doesn't exist.
    if not os.path.isdir(base_path):
        print(f"Error: Base directory not found at '{base_path}'")
        return functions_details

    # Iterate through all files in the description directory.
    for library, names in libraries_to_load.items():
        functions_to_find.extend(names)
        file_path = os.path.join(base_path, f"{library}.json")
    # for filename in os.listdir(base_path):
    #     if not filename.endswith(".json"):
    #         continue

        # file_path = os.path.join(base_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                library_data = json.load(f)

            # Standardize the data to be a list for uniform processing.
            if isinstance(library_data, dict):
                data_to_process = [library_data]
            elif isinstance(library_data, list):
                data_to_process = library_data
            else:
                continue
            i=0
            #print('func obj i=',i, "***************************888")
            #print(data_to_process)
            # Process the list of function objects from the file.
            for func_obj in data_to_process:
                print('func obj i=',i, "***************************888")
                i+=1
                
                if isinstance(func_obj, dict) and 'name' in func_obj:
                    func_name = func_obj['name']
                    # # If this is a function we are looking for:
                    print(func_name)
                    # print("************************************")
                    # print(functions_to_find)
                    if func_name in functions_to_find:
                        # Append the entire function object to the list.
                        print('YOOOOO')
                        functions_details.append(func_obj)
                        functions_to_find.remove(func_name)

            # If all functions are found, we can stop searching.
            #if not functions_to_find:
            #    break

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{file_path}'")
        except Exception as e:
            print(f"An error occurred while processing '{file_path}': {e}")

    # Report any functions that were not found in any file.
    if functions_to_find:
        for func_name in functions_to_find:
            print(f"Warning: Function '{func_name}' not found in any library file.")

    return functions_details
