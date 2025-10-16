# vision_dashboard/utils/settings.py

import os
import json

ARENA_SETTINGS_PATH = "Settings/arena_settings.json"

def load_arena_settings():
    """Loads arena settings, creating a default file if missing or corrupt."""
    default = {"rows": 1, "cols": 1, "cells": {}}
    try:
        with open(ARENA_SETTINGS_PATH, "r") as f:
            settings = json.load(f)
            # Ensure essential keys are present
            for key, value in default.items():
                settings.setdefault(key, value)
            return settings
    except (json.JSONDecodeError, FileNotFoundError):
        if not os.path.exists(os.path.dirname(ARENA_SETTINGS_PATH)):
            os.makedirs(os.path.dirname(ARENA_SETTINGS_PATH))
        with open(ARENA_SETTINGS_PATH, "w") as f:
            json.dump(default, f, indent=2)
        return default