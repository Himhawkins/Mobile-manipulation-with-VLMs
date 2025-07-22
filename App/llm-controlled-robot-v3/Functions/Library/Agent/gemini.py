
import json
from .load import load_agent_description,load_functions_description
try:
    import google.generativeai as genai
    from google.generativeai.types import FunctionDeclaration, Tool
except ImportError:
    print("Please install google-generativeai: pip install google-generativeai")
    genai = None
    FunctionDeclaration = None
    Tool = None

def gemini_tool_list(function_details):
    """
    Converts a dictionary of function details into a list of tools
    formatted for the Gemini API.

    Args:
        function_details (dict): A dictionary where keys are function names and
                                 values are their detailed descriptions.

    Returns:
        list: A list of dictionaries, with each dictionary representing a
              tool conforming to the Gemini API's tool schema.
    """
    gemini_tools = []

    type_mapping = {
        "str": "STRING",
        "string": "STRING",
        "int": "NUMBER",
        "integer": "NUMBER",
        "float": "NUMBER",
        "bool": "BOOLEAN",
        "boolean": "BOOLEAN",
    }
    
    for name, details in function_details.items():
        properties = {}
        required = []

        if "inputs" in details and isinstance(details["inputs"], list):
            for param in details["inputs"]:
                param_name = param.get("name")
                param_type = param.get("type", "string").lower()
                
                # Simple type mapping
                gemini_type = type_mapping.get(param_type, "STRING")
                
                properties[param_name] = {
                    "type": gemini_type,
                    "description": param.get("description", "")
                }
                
                # Heuristic for determining required parameters: if "optional" is not
                # mentioned in the description, assume it's required.
                if "optional" not in param.get("description", "").lower():
                    required.append(param_name)

        tool_declaration = {
            "name": name,
            "description": details.get("description", ""),
            "parameters": {
                "type": "OBJECT",
                "properties": properties,
                "required": required
            }
        }
        gemini_tools.append(tool_declaration)
        
    return gemini_tools



def call_gemini_agent(prompt, agent_name,model_ver='gemini-2.5-flash'):
    """
    Runs a full agent loop: load, format, and call Gemini with tools.

    Args:
        prompt (str): The user's prompt for the agent.
        agent_name (str): The name of the agent to run.
    """
    if not all([genai, FunctionDeclaration, Tool]):
        print("Google Generative AI library not installed. Cannot run agent.")
        return

    print(f"--- Running Agent: {agent_name} ---")
    
    # 1. Fetch agent details
    background, functions_list = load_agent_description(agent_name)
    if not (background or functions_list):
        print(f"Failed to load agent '{agent_name}'. Aborting.")
        return

    # 2. Fetch function details
    function_details = load_functions_description(functions_list)
    if not function_details:
        print("Could not load any function details. Aborting.")
        return

    # 3. Get Gemini tool list
    gemini_tool_definitions = gemini_tool_list(function_details)
    if not gemini_tool_definitions:
        print("Could not create Gemini tool definitions. Aborting.")
        return
        
    # 4. Create Gemini Tool objects
    try:
        function_declarations = [FunctionDeclaration(**tool) for tool in gemini_tool_definitions]
        agent_tool = Tool(function_declarations=function_declarations)
    except Exception as e:
        print(f"Error creating Gemini Tool object: {e}")
        return

    # 5. Combine prompt, background, and function details
    function_documentation = "\n".join(functions_list)
    full_prompt = (
        f"Agent Background: {background}\n\n"
        f"Function Documentation:\n{function_documentation}\n\n"
        f"User Request: {prompt}"
    )
    
    print("\n--- Calling Gemini API ---")
    try:
        # Configure your API key (replace with your actual key or set as an env var)
        # genai.configure(api_key="YOUR_API_KEY")

        # Initialize the model, providing the tool definition
        model = genai.GenerativeModel(
            model_name=model_ver,
            tools=[agent_tool]
        )
        
        response = model.generate_content(full_prompt)
        print("\n--- Gemini Response ---")
        print(response)

    except Exception as e:
        print(f"\nAn error occurred during the Gemini API call: {e}")
        print("Please ensure your API key is configured correctly.")

