
from Functions.Library.Agent.load import load_agent_description,load_functions_description
import Functions.Library.Agent.gemini import gemini_tool_list






if __name__ == "__main__":
    # To test this, you need a directory structure like this:
    

    agent_name = "Planning_Agent"
    background, functions = load_agent_description(agent_name)

    if background or functions:
        print(f"\nBackground for '{agent_name}':")
        print(background)
        print(f"\nFunctions for '{agent_name}':")
        for func_info in functions:
            print(f"- {func_info}")

        print("\n--- Loading function descriptions ---")
        function_details = load_functions_description(functions)
        if function_details:
            for name, details in function_details.items():
                print(f"\nDetails for function '{name}':")
                print(json.dumps(details, indent=2))
        else:
            print("No function descriptions were loaded.")


        gemini_formatted_tools = gemini_tool_list(function_details)
        print(json.dumps(gemini_formatted_tools, indent=2))

    print("\n" + "="*40 + "\n")

    print("--- Attempting to load a non-existent agent ---")
    non_existent_agent = "ghost_agent"
    background, functions = load_agent_description(non_existent_agent)
    if not background and not functions:
        print(f"Could not load agent '{non_existent_agent}' as expected.")
