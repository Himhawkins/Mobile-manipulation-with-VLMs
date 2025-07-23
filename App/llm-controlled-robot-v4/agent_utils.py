import os
import json

def get_agent_folders():
    """
    Returns a list of subdirectories under 'Agents/'.
    """
    base_path = "Agents"
    try:
        return [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    except Exception:
        return []

def get_agent_functions(folder):
    """
    Loads the function configuration from an agent's functions.json file.
    Returns a dict mapping library names to function names.
    """
    json_path = os.path.join("Agents", folder, "functions.json")
    try:
        with open(json_path, "r") as f:
            items = json.load(f)
    except Exception:
        return {}
    groups = {}
    for entry in items:
        lib = entry.get("library")
        name = entry.get("name")
        if lib and name:
            groups.setdefault(lib, []).append(name)
    return groups

def get_all_functions(desc_dir="Functions/description"):
    """
    Reads all function descriptions from JSON files in the description directory.
    Returns a dict {library: [function names]}.
    """
    patterns = {}
    for fname in os.listdir(desc_dir):
        if not fname.endswith(".json"):
            continue
        key = os.path.splitext(fname)[0]
        path = os.path.join(desc_dir, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            patterns[key] = list(data.keys())
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    return patterns

def save_agent_to_disk(popup, old_folder, new_name, desc_text, check_groups):
    """
    Saves or renames an agent folder and writes the description and selected functions
    to disk. Used by the Agent Editor popup.
    
    Parameters:
    - popup: the CTkToplevel window to close if save is successful
    - old_folder: the previous agent name ("" if new)
    - new_name: new name entered by user
    - desc_text: description string
    - check_groups: list of CheckGroup objects for selected functions
    """
    agents_path = "Agents"
    old_path = os.path.join(agents_path, old_folder)
    new_path = os.path.join(agents_path, new_name)

    if not old_folder:
        os.makedirs(new_path, exist_ok=True)
    elif new_name != old_folder:
        try:
            os.rename(old_path, new_path)
        except Exception as e:
            print(f"Failed to rename folder: {e}")
            popup.destroy()
            return
    else:
        new_path = old_path

    try:
        with open(os.path.join(new_path, "description.txt"), "w") as f:
            f.write(desc_text)
    except Exception as e:
        print(f"Failed to save description: {e}")

    function_data = []
    for group in check_groups:
        group_data = group.get_selected()
        library_name = group.label
        for func_name, selected in group_data["options"].items():
            if selected:
                function_data.append({
                    "library": library_name,
                    "name": func_name
                })

    try:
        with open(os.path.join(new_path, "functions.json"), "w") as f:
            json.dump(function_data, f, indent=4)
    except Exception as e:
        print(f"Failed to save functions: {e}")
