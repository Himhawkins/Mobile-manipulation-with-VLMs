
import json

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
