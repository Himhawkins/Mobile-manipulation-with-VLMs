#from google.generativeai.types import Part
#from Functions.Library.Agent.load import load_agent_description,load_functions_description
from Functions.Library.Agent.gemini import call_gemini_agent

import json

import cv2
import numpy as np


if __name__ == "__main__":
    # To test this, you need a directory structure like this:
    

    agent_name = "Planning_Agent"#"Planning_Agent"
    
    user_prompt = "Display the camera frame image"#" generate a circle of radius 40 and 60 points and then circle of raduis 40 and 20 points"
    
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    cv2.imwrite("Data/frame_img.png", frame)
    response=call_gemini_agent(user_prompt, agent_name) # frame optional
