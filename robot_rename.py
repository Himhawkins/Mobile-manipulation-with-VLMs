import os
import json
from pathlib import Path

def rename_robot(robot_id: int, robot_name: str, json_path: str = "Data/robot_names.json") -> str:
    """
    Update or add a robot's name in Data/robot_names.json.

    Args:
        robot_id   (int): ID of the robot.
        robot_name (str): New name to assign.
        json_path (str): Path to the JSON file (default: Data/robot_names.json).

    Returns:
        str: Status message.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Load file if it exists
    data = {"robots": []}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict) or "robots" not in data:
                    data = {"robots": []}
        except Exception:
            data = {"robots": []}

    # Update or insert
    rid = int(robot_id)
    updated = False
    for entry in data["robots"]:
        try:
            if int(entry.get("id", -1)) == rid:
                entry["name"] = str(robot_name)
                updated = True
                break
        except Exception:
            continue

    if not updated:
        data["robots"].append({"id": rid, "name": str(robot_name)})

    # Save back
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return f"Robot {robot_id} renamed to '{robot_name}'."


def get_robot_name(robot_id: int, json_path: str = "Data/robot_names.json") -> str | None:
    """
    Fetch the robot's name given its ID.

    Args:
        robot_id (int): ID of the robot.
        json_path (str): Path to the JSON file.

    Returns:
        str | None: Robot name if found, else None.
    """
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "robots" not in data:
            return None
    except Exception:
        return None

    rid = int(robot_id)
    for entry in data["robots"]:
        try:
            if int(entry.get("id", -1)) == rid:
                return str(entry.get("name", None))
        except Exception:
            continue

    return None


def get_robot_id(robot_name: str, json_path: str = "Data/robot_names.json") -> int | None:
    """
    Fetch the robot's ID given its name.

    Args:
        robot_name (str): Name of the robot.
        json_path (str): Path to the JSON file.

    Returns:
        int | None: Robot ID if found, else None.
    """
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "robots" not in data:
            return None
    except Exception:
        return None

    for entry in data["robots"]:
        try:
            if str(entry.get("name", "")).lower() == str(robot_name).lower():
                return int(entry.get("id", None))
        except Exception:
            continue

    return None


if __name__ == "__main__":
    # Example usage:
    # msg = rename_robot(2, "Bravo")
    # print(msg)
    print(get_robot_name(1))   # "Alpha" if exists
    print(get_robot_name(3))   # None if not found
    print(get_robot_id("Alpha"))  # 1 if name exists
    print(get_robot_id("Charlie"))  # None if not found
