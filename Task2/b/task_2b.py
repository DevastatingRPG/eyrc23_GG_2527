'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 2B of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''
############################## FILL THE MANDATORY INFORMATION BELOW ###############################

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_2b.py
# Functions:	    [`classify_event(image)` ]
###################################################################################################

# IMPORTS (DO NOT CHANGE/REMOVE THESE IMPORTS)
from sys import platform
import numpy as np
import subprocess
import shutil
import ast
import sys
import os

# Additional Imports
'''
You can import your required libraries here
'''
import cv2
import torch
from torchvision import transforms
from PIL import Image
# DECLARING VARIABLES (DO NOT CHANGE/REMOVE THESE VARIABLES)
detected_list = []
numbering_list = []
img_name_list = []

# Declaring Variables
'''
You can delare the necessary variables here
'''

# EVENT NAMES
'''
We have already specified the event names that you should train your model with.
DO NOT CHANGE THE BELOW EVENT NAMES IN ANY CASE
If you have accidently created a different name for the event, you can create another 
function to use the below shared event names wherever your event names are used.
(Remember, the 'classify_event()' should always return the predefined event names)  
'''
combat = "combat"
rehab = "humanitarianaid"
military_vehicles = "militaryvehicles"
fire = "fire"
destroyed_building = "destroyedbuilding"
###################################################################################################
###################################################################################################
''' 
	Purpose:
	---
	This function will load your trained model and classify the event from an image which is 
    sent as an input.
	
	Input Arguments:
	---
	`image`: Image path sent by input file 	
	
	Returns:
	---
	`event` : [ String ]
						  Detected event is returned in the form of a string

	Example call:
	---
	event = classify_event(image_path)
	'''
# def classify_event(image):
#     # if torch.cuda.is_available(): 
#     #     dev = "cuda:0" 
#     # else: 
#     #     dev = "cpu" 
#     # device = torch.device(dev)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(device)
#     image = cv2.imread(image)
#     sr = cv2.dnn_superres.DnnSuperResImpl_create()
#     path = "EDSR_x4.pb"   
#     sr.readModel(path)   
#     sr.setModel("edsr",4) 

#     scale = 0.5  #50% of original image
#     width = int(image.shape[1] * scale)
#     height = int(image.shape[0] * scale)
#     dim = (width, height)    
#     cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


#     result = sr.upsample(image)
#     model = torch.load('model.tf').to(device)
#     model.eval()
#     image_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize((224, 224), antialias=False),           
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]),
#         ])
#     with torch.inference_mode():
#       # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
#       transformed_image = image_transform(result).unsqueeze(dim=0)

#       # 7. Make a prediction on image with an extra dimension and send it to the target device
#       target_image_pred = model(transformed_image.to(device))

#     # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
#     target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

#     # 9. Convert prediction probabilities -> prediction labels
#     pred = torch.argmax(target_image_pred_probs, dim=1)
    
#     # pred = model.predict(image)
#     class_names = ['combat', 'destroyedbuilding', 'fire', 'humanitarianaid', 'militaryvehicles']
#     event = class_names[pred]
#     return event

# ...

def classify_event(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    image = cv2.imread(image)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "EDSR_x4.pb"   
    sr.readModel(path)   
    sr.setModel("edsr",4) 

    # edsr_model = torch.load(path).to(device)
    # edsr_model.eval()

    scale = 0.5  #50% of the original image
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)    
    cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    print("hello")
    # result = sr.upsample(image)
    result = image
    print("Hello")
    model = torch.load('model.tf')
    model = model.to(device)
    model.eval()

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=False),           
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    with torch.inference_mode():
        transformed_image = image_transform(result).unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    pred = torch.argmax(target_image_pred_probs, dim=1)

    class_names = ['combat', 'destroyedbuilding', 'fire', 'humanitarianaid', 'militaryvehicles']
    event = class_names[pred]
    return event

# ...


# ADDITIONAL FUNCTIONS
'''
Although not required but if there are any additonal functions that you're using, you shall add them here. 
'''

###################################################################################################
########################### DO NOT MAKE ANY CHANGES IN THE SCRIPT BELOW ###########################
def classification(img_name_list):
    for img_index in range(len(img_name_list)):
        img = "events/" + str(img_name_list[img_index]) + ".jpeg"
        detected_event = classify_event(img)
        print((img_index + 1), detected_event)
        if detected_event == combat:
            detected_list.append("combat")
        if detected_event == rehab:
            detected_list.append("rehab")
        if detected_event == military_vehicles:
            detected_list.append("militaryvehicles")
        if detected_event == fire:
            detected_list.append("fire")
        if detected_event == destroyed_building:
            detected_list.append("destroyedbuilding")
    shutil.rmtree('events')
    return detected_list

def detected_list_processing(detected_list):
    try:
        detected_events = open("detected_events.txt", "w")
        detected_events.writelines(str(detected_list))
    except Exception as e:
        print("Error: ", e)

def input_function():
    if platform == "win32":
        try:
            subprocess.run("input.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./input")
        except Exception as e:
            print("Error: ", e)
    img_names = open("image_names.txt", "r")
    img_name_str = img_names.read()

    img_name_list = ast.literal_eval(img_name_str)
    return img_name_list
    
def output_function():
    if platform == "win32":
        try:
            subprocess.run("output.exe")
        except Exception as e:
            print("Error: ", e)
    if platform == "linux":
        try:
            subprocess.run("./output")
        except Exception as e:
            print("Error: ", e)

###################################################################################################
def main():
    ##### Input #####
    img_name_list = input_function()
    #################

    ##### Process #####
    detected_list = classification(img_name_list)
    detected_list_processing(detected_list)
    ###################

    ##### Output #####
    output_function()
    ##################

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        if os.path.exists('events'):
            shutil.rmtree('events')
        sys.exit()
###################################################################################################
###################################################################################################
