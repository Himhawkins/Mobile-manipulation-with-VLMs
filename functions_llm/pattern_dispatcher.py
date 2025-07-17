import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from geometric_functions import (
    read_grid_size,
    generate_circle_pattern,
    generate_rectangle_pattern,
    generate_trapezium_pattern,
    generate_parallelogram_pattern,
    generate_diamond_pattern,
    generate_lawnmower_pattern,
    generate_zigzag_pattern,
    generate_spiral_pattern,
    generate_ellipse_pattern
)

# === CONFIGURE GEMINI ===
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-pro")

# Mapping of shape keywords to local functions and their parameter names
FUNCTIONS = {
    "circle":       {"func": generate_circle_pattern,     "params": ["radius", "num_points", "grid_size"]},
    "rectangle":    {"func": generate_rectangle_pattern,  "params": ["width_rect", "height_rect", "num_points_per_side", "grid_size"]},
    "trapezium":    {"func": generate_trapezium_pattern,  "params": ["top_width", "bottom_width", "height", "num_points_per_side", "grid_size"]},
    "parallelogram":{"func": generate_parallelogram_pattern,"params": ["width_para", "height_para", "angle", "num_points_per_side", "grid_size"]},
    "diamond":      {"func": generate_diamond_pattern,     "params": ["width", "height", "num_points_per_side", "grid_size"]},
    "lawnmower":    {"func": generate_lawnmower_pattern,   "params": ["width_area", "height_area", "spacing", "grid_size"]},
    "zigzag":       {"func": generate_zigzag_pattern,      "params": ["width_area", "height_area", "num_zigs", "grid_size"]},
    "spiral":       {"func": generate_spiral_pattern,      "params": ["turns", "max_radius", "num_points", "grid_size"]},
    "ellipse":      {"func": generate_ellipse_pattern,     "params": ["major_axis", "minor_axis", "num_points", "grid_size"]},
}


def classify_and_extract(prompt: str) -> dict:
    """
    Ask Gemini which local function to use. If the response isn't valid JSON,
    strip any code fences and try again; on failure, fallback to use_local=False.
    Returns: { use_local: bool, shape: str?, args: dict }
    """
    system = "You know these local shape functions:\n"
    for name, meta in FUNCTIONS.items():
        system += f"  • {name}: {meta['func'].__name__}({', '.join(meta['params'])})\n"
    system += (
        "\nRespond with EXACT JSON (no markdown, no backticks).\n"
        "If the prompt matches one of these shapes, output:\n"
        "{ \"use_local\": true, \"shape\": \"circle\", \"args\": { /* params */ } }\n"
        "Otherwise output only: { \"use_local\": false }."
    )
    full_prompt = system + "\nUser Prompt: " + prompt
    resp = model.generate_content(full_prompt)
    text = resp.text.strip()
    # Strip code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first and last fence lines
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1])
        else:
            text = text.replace("```", "")
    try:
        decision = json.loads(text)
        if not isinstance(decision, dict) or 'use_local' not in decision:
            raise ValueError("Invalid JSON structure")
        return decision
    except Exception:
        print(f"[Warning] classify_and_extract failed to parse JSON, falling back to LLM generation.\nResponse was: {text}")
        return {"use_local": False}


def generate_via_llm(prompt: str) -> list:
    """
    Fallback: ask Gemini to generate the full points list JSON and parse it.
    """
    resp = model.generate_content(prompt)
    text = resp.text
    start, end = text.find('['), text.rfind(']') + 1
    snippet = text[start:end]
    data = json.loads(snippet)
    return [(p['x'], p['y'], p['theta']) for p in data]


def dispatch(prompt: str) -> list:
    """
    Main entry: use local math functions or fallback to raw LLM.
    """
    decision = classify_and_extract(prompt)
    if decision.get('use_local'):
        shape = decision['shape']
        args = decision.get('args', {}) or {}
        # Ensure grid_size default
        args.setdefault('grid_size', read_grid_size())
        # Default area bounds for certain shapes
        if shape in ('lawnmower', 'zigzag'):
            gw, gh = args['grid_size']
            args.setdefault('width_area', gw)
            args.setdefault('height_area', gh)
        fn = FUNCTIONS[shape]['func']
        return fn(**args)
    else:
        return generate_via_llm(prompt)
