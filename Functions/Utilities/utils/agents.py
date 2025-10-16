# vision_dashboard/utils/agents.py

import os
import json

AGENTS_DIR = "Agents"

def get_all_functions() -> dict:
    """
    Returns a dictionary of all available functions, grouped by category.
    In a real system, this might discover functions dynamically.
    """
    return {
        "Detection": ["detect_obstacles", "find_robot_pose", "read_qr_code"],
        "Motion": ["move_to_point", "rotate_to_angle", "pick_up_object", "drop_off_object"],
        "Navigation": ["plan_path", "avoid_obstacles"]
    }

def get_agent_functions(agent_name: str) -> dict:
    """Loads the selected functions for a given agent from its JSON file."""
    config_path = os.path.join(AGENTS_DIR, agent_name, "functions.json")
    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {}

def get_agent_folders() -> list:
    """Finds all subdirectories in the Agents folder."""
    if not os.path.exists(AGENTS_DIR):
        os.makedirs(AGENTS_DIR)
        return []
    return [d for d in os.listdir(AGENTS_DIR) if os.path.isdir(os.path.join(AGENTS_DIR, d))]

def save_agent_to_disk(popup, old_name, new_name, description, function_groups):
    """Saves the agent's description and function configuration."""
    # Rename folder if name changed
    if old_name and old_name != new_name and os.path.exists(os.path.join(AGENTS_DIR, old_name)):
        os.rename(os.path.join(AGENTS_DIR, old_name), os.path.join(AGENTS_DIR, new_name))

    agent_path = os.path.join(AGENTS_DIR, new_name)
    os.makedirs(agent_path, exist_ok=True)

    # Save description
    with open(os.path.join(agent_path, "description.txt"), "w") as f:
        f.write(description)

    # Save selected functions to JSON
    selected_functions = {}
    for group in function_groups:
        selected = group.get_selected()
        if selected: # Only save categories that have selections
            selected_functions[group.label] = selected
            
    with open(os.path.join(agent_path, "functions.json"), "w") as f:
        json.dump(selected_functions, f, indent=2)

    print(f"Agent '{new_name}' saved with functions.")