
#from Functions.Library.Agent.load import load_agent_description,load_functions_description
from Functions.Library.Agent.gemini import call_gemini_agent

import json




if __name__ == "__main__":
    # To test this, you need a directory structure like this:
    

    agent_name = "Planning_Agent"
    
    user_prompt = " generate a circle of radius 40 and 60 points and then circle of raduis 40 and 20 points"
    call_gemini_agent(user_prompt, "Planning_Agent")

    