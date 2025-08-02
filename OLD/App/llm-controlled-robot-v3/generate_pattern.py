#!/usr/bin/env python3
"""
generate_pattern.py

Provides functions to configure Google Gemini LLM, generate a JSON list
of (x,y,theta) points from a natural-language prompt, parse that JSON into
Python tuples, and visualize the result. Includes a simple demo in __main__ without CLI arguments.
"""
import json
from typing import List, Tuple

import google.generativeai as genai
import matplotlib.pyplot as plt
import numpy as np


def configure_gemini(api_key: str, model_name: str = "models/gemini-2.5-pro") -> object:
    """
    Configure and return a Gemini GenerativeModel instance.

    :param api_key: API key for Google Gemini
    :param model_name: Gemini model identifier
    :returns: genai.GenerativeModel
    """
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def generate_points(model: object, prompt: str) -> str:
    """
    Ask the Gemini model to generate a JSON array of point objects.

    :param model: genai.GenerativeModel from configure_gemini
    :param prompt: natural-language description of the desired pattern
    :returns: raw JSON string
    """
    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Remove markdown fences if present
    if raw_text.startswith("```json"):
        lines = raw_text.splitlines()
        clean_text = "\n".join(lines[1:-1])
    else:
        clean_text = raw_text

    return clean_text


def parse_points(raw: str) -> List[Tuple[float, float, float]]:
    """
    Parse a JSON array string into a list of (x, y, theta) tuples.

    :param raw: JSON string representing a list of objects with 'x', 'y', 'theta'
    :returns: list of tuples of floats
    """
    data = json.loads(raw)
    points: List[Tuple[float, float, float]] = []
    for obj in data:
        x = float(obj.get("x"))
        y = float(obj.get("y"))
        theta = float(obj.get("theta"))
        points.append((x, y, theta))
    return points


def visualize_points(points: List[Tuple[float, float, float]], title: str = "Generated Path with Orientations"):
    """
    Visualize (x,y,theta) points with matplotlib.

    :param points: list of (x, y, theta)
    """
    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    theta_vals = [p[2] for p in points]

    plt.figure(figsize=(8, 8))
    plt.plot(x_vals, y_vals, 'mo-', label='Generated Path')
    for x, y, theta in zip(x_vals, y_vals, theta_vals):
        dx = 20 * np.cos(theta)
        dy = 20 * np.sin(theta)
        plt.arrow(x, y, dx, dy, head_width=10, head_length=10, fc='blue', ec='blue')

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    import sys

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    prompt = "Give 20 (x, y, theta) points that trace a rectangle of length 200px and width 100px in a 673x673 image. Use radians for theta. Give only the list of points in JSON format. Don't explain and don't hallucinate."

    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is required", file=sys.stderr)
        sys.exit(1)
    if not prompt:
        print("Error: GEMINI_PROMPT environment variable is required", file=sys.stderr)
        sys.exit(1)

    # Configure and generate
    model = configure_gemini(api_key)
    raw = generate_points(model, prompt)

    # Display raw JSON and parsed points
    print("Raw JSON response:")
    print(raw)
    try:
        points = parse_points(raw)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nParsed points:")
    for idx, pt in enumerate(points, start=1):
        print(f"{idx}: x={pt[0]}, y={pt[1]}, theta={pt[2]}")

    # Plot the points
    visualize_points(points)
