# marker_sequence_parser.py

import json
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

# Allowed markers: "HOME" plus all letters A–Z
ALLOWED_MARKERS = {"HOME"} | {chr(c) for c in range(ord("A"), ord("Z") + 1)}

SYSTEM_INSTRUCTIONS = """
You are an assistant that reads a user's instruction about the order to visit arena markers.
Valid markers are "Home" and the letters A through Z.
Respond with EXACTLY a JSON array (no markdown, no backticks) listing the markers
in the order given by the user, capitalized (e.g. ["HOME", "A", "B", "HOME"]).
If the user mentions any invalid markers, ignore them.
If you cannot find any valid markers, return an empty list: [].
"""

def parse_marker_sequence(prompt: str, api_key: str) -> List[str]:
    """
    Parse a natural-language prompt into an ordered list of markers.

    :param prompt: The user's instruction, e.g. "Go from Home to A then C then back to Home."
    :param api_key: Your Google Gemini API key.
    :returns: List of uppercase marker labels, filtered to valid ones.
    """
    # configure & instantiate the Gemini model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-pro")

    # build the combined prompt
    full_prompt = SYSTEM_INSTRUCTIONS + "\nUser prompt: " + prompt

    # call the model
    resp = model.generate_content(full_prompt)
    text = resp.text.strip()

    # extract the JSON array substring
    start = text.find('[')
    end = text.rfind(']') + 1
    if start == -1 or end == -1:
        # fallback: no JSON array found
        return []

    snippet = text[start:end]
    try:
        raw = json.loads(snippet)
    except json.JSONDecodeError:
        return []

    # filter and normalize
    sequence: List[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        candidate = item.strip().upper()
        if candidate in ALLOWED_MARKERS:
            sequence.append(candidate)
    return sequence

if __name__ == "__main__":
    import os
    # simple demo
    load_dotenv
    gm_api_key = os.getenv("GEMINI_API_KEY")
    while True:
        prompt = input("Enter marker order (or 'quit'): ").strip()
        if prompt.lower() in {"q", "quit", "exit"}:
            break
        seq = parse_marker_sequence(prompt, gm_api_key)
        print("Parsed sequence:", seq)
