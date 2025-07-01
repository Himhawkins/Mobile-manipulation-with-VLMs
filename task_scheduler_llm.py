import subprocess
import json

# --------------------------------------------
# 1. Robot control functions (replace with your real logic)
# --------------------------------------------

def detect_target(marker):
    print(f"[detect_target] Detecting target for marker '{marker}'...")
    target_pos = (100, 200)  # Dummy value — replace with real detection
    with open("target.txt", "w") as f:
        f.write(f"{target_pos[0]},{target_pos[1]}")
    return target_pos

def detect_obstacles():
    print("[detect_obstacles] Detecting obstacles...")
    obstacle_data = [[0]*21 for _ in range(16)]  # Dummy 16x21 grid
    with open("obstacle.txt", "w") as f:
        for row in obstacle_data:
            f.write(",".join(map(str, row)) + "\n")
    return obstacle_data

def run_astar():
    print("[run_astar] Computing optimal path from robot_pos.txt to target.txt...")
    path = [(10, 10), (20, 20), (30, 30)]  # Dummy path — replace with A*
    return path

def goto(path):
    print("[goto] Following path to target...")
    for pt in path:
        print(f"  -> {pt}")
    print("[goto] Reached destination.")

# --------------------------------------------
# 2. Function to query Ollama3 with system+user prompt
# --------------------------------------------

def query_ollama(system_prompt, user_prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}",
            text=True,
            capture_output=True,
            check=True
        )
        response = result.stdout.strip()
        print("[RAW OLLAMA OUTPUT]:\n", response)

        # Try to extract code from markdown block
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response  # fallback: treat entire output as code
        return code
    except subprocess.CalledProcessError as e:
        print("[ERROR] Ollama3 failed:", e.stderr)
        return None


# --------------------------------------------
# 3. Main driver
# --------------------------------------------

def main():
    print("\n[Robot Scheduler] Type a natural command like:")
    print("  → Go to marker A")
    print("  → Go to marker B then D\n")

    user_input = input("Enter command: ").strip()

    # --- System prompt for the LLM ---
    system_prompt = """
    You are a Python automation agent that generates robot navigation scripts from natural language commands.

    The robot control system includes the following Python modules in the current directory:

    1. `find_target.py`:
    - Function: `find_target(prompt: str)`
    - Usage: Call `find_target("A")` to detect marker A and save its position to `target.txt`.

    2. `moon_obstacles.py`:
    - Function: `find_obstacles(object_prompt: str = "obstacles black rectangles solid")`
    - Usage: Call `find_obstacles("obstacles black rectangles solid")` to detect and save obstacles to `obstacles.txt`.

    3. `a4_min_dist.py`:
    - Function: `main()`
    - You must import it as `from a4_min_dist import main as run_astar` and call it using `run_astar()`.

    ---

    Your task is to:
    - Interpret commands like:
        - "Go to marker A"
        - "Go to marker B then D"
        - "Reach markers A, C, and F"
    - Generate a Python script that:
        - Imports all required functions **once at the top**.
        - For each marker, calls:
            1. `find_target("marker_name")`
            2. `find_obstacles("obstacles black rectangles solid")`
            3. `run_astar()`
        - Does **not include** the `goto()` function.
        - Does **not repeat** imports for each marker.
        - Does **not add comments or explanations**.

    ---

    ### Output Format Example

    If the user says:  
    `Go to marker A then C`

    You must respond with:

    ```python
    from find_target import find_target
    from moon_obstacles import find_obstacles
    from a4_min_dist import main as run_astar

    find_target("A")
    find_obstacles("obstacles black rectangles solid")
    run_astar()

    find_target("C")
    find_obstacles("obstacles black rectangles solid")
    run_astar()
	"""
    print("[INFO] Querying Ollama3 for task plan...\n")
    generated_code = query_ollama(system_prompt, user_input)

    if generated_code:
        print("[Generated Python Code]:\n")
        print(generated_code)
        with open("generated_task.py", "w") as f:
            f.write(generated_code)
        print("\n[INFO] Executing generated code...\n")
        exec(generated_code, globals())
    else:
        print("[ERROR] Failed to generate valid code.")

if __name__ == "__main__":
    main()
