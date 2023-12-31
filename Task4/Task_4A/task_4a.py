'''
*****************************************************************************************
*
*        		===============================================
*           		Geo Guide (GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 4A of Geo Guide (GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			2527
# Author List:		Shubham, Aditya
# Filename:			task_4a.py


####################### IMPORT MODULES #######################
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import cv2
import numpy as np
import copy
import torch
import torch.nn as nn
from torchvision import transforms  
import imutils     
import math
from cv2 import aruco  
import RRDBNet_arch as arch     
from torchvision.models import efficientnet_v2_s
from skimage import exposure, io, color

##############################################################



################# ADD UTILITY FUNCTIONS HERE #################


def detect_ArUco_details(image):
    ArUco_details_dict = {}
    ArUco_corners = {}
    
    ##############	ADD YOUR CODE HERE	##############
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    arucoParams = aruco.DetectorParameters()
    # GrayScale Conversion
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = aruco.detectMarkers(gray_image, aruco_dict, parameters=arucoParams)

    if ids is not None:
        for i in range(len(ids)):
            marker_id = int(ids[i][0])
            marker_center = [int(coord) for coord in list(np.mean(corners[i][0], axis=0).astype(int))]

            # Store details in dictionaries
            ArUco_details_dict[marker_id] = marker_center
            ArUco_corners[marker_id] = [[int(corner[0]), int(corner[1])] for corner in corners[i][0]]
    ##################################################
    
    return ArUco_details_dict, ArUco_corners 

def decrease_resolution(image, target_width, target_height):
    dim = (target_width, target_height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

##############################################################


def task_4a_return():
    """
    Purpose:
    ---
    Only for returning the final dictionary variable
    
    Arguments:
    ---
    You are not allowed to define any input arguments for this function. You can 
    return the dictionary from a user-defined function and just call the 
    function here

    Returns:
    ---
    `identified_labels` : { dictionary }
        dictionary containing the labels of the events detected
    """  
    identified_labels = {}  
    
##############	ADD YOUR CODE HERE	##############
    img = cv2.imread("eval.jpg")
    img = imutils.resize(img, width=960)
    _, ArUco_corners = detect_ArUco_details(img)
    marking_img = np.copy(img)
    corners = copy.deepcopy(ArUco_corners)
    events = [
        [[corners[7][1][0], corners[21][0][1]], [corners[21][0][0], corners[7][1][1]-10]],
        [corners[28][1], corners[14][0]],
        [corners[31][1], corners[11][3]], 
        [[corners[25][0][0], corners[34][0][1]], [corners[34][0][0], corners[25][0][1]]],    
        [corners[54][2], corners[40][0]]   
    ]

    i=1
    eventlist=[]
    letters = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    classconv = { "combat": "Combat", "destroyedbuilding": "Destroyed buildings", 
                 "humanitarianaid": "Humanitarian Aid and rehabilitation",
                 "militaryvehicles": "Military Vehicles", "fire": "Fire"}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = efficientnet_v2_s().to(device)
    model.classifier = torch.nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=5, bias=True),
        # nn.ReLU6(),
    ).to(device)
    model.load_state_dict(torch.load('w2.tf'))
    model.eval()
    image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=False),
            
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    model_path = 'RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth

    modelup = arch.RRDBNet(3, 3, 64, 23, gc=32)
    modelup.load_state_dict(torch.load(model_path), strict=True)
    modelup.eval()
    modelup = modelup.to(device)

    temp = 'output/temp.jpg'

    for tl, br in events:
        tl_adj = [tl[0] + 10, tl[1] + 7]
        br_adj = [br[0] - 10, br[1] - 4]
        roi = img[tl_adj[1]:br_adj[1], tl_adj[0]:br_adj[0]]
        

        # Apply a blur to the image
        blurred = cv2.blur(roi, (5, 5))
        # Apply a bilateral filter to the image
        filtered = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
        # Perform morphological opening
        kernel = np.ones((5,5),np.uint8)
        opened = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)

        gray = cv2.cvtColor(opened, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to the image
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours in the image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area in descending order, take the first one (the largest)
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # Get the bounding rectangle of the largest contour
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the image using the bounding rectangle, add some padding if needed
        padding = 0  # adjust this value according to your needs
        crop = roi[max(0, y-padding):min(y+h+padding, roi.shape[0]), max(0, x-padding):min(x+w+padding, roi.shape[1])]
        

        offset_x = tl_adj[0] + x
        offset_y = tl_adj[1] + y    


        eventlist.append(crop)
        cv2.imwrite(temp, crop)
        crop = cv2.imread(temp, cv2.IMREAD_COLOR)
        cv2.imshow("test", crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        crop = cv2.resize(crop, (150, 150))
        crop = cv2.fastNlMeansDenoisingColored(crop, None, h=10, templateWindowSize=7, searchWindowSize=21)

        
        # cv2.imshow("test", equalized_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # crop = cv2.resize(crop, (200, 200))
        # cv2.imshow("test", crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # crop = decrease_resolution(crop, 150, 150)
    
    
        # ESRGAN

        crop = crop * 1.0 / 255
        crop = torch.from_numpy(np.transpose(crop[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = crop.unsqueeze(0)
        img_LR = img_LR.to(device)
        with torch.no_grad():
            output = modelup(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite(temp, output)

        result = cv2.imread(temp, cv2.IMREAD_COLOR)  
        
        # result = cv2.fastNlMeansDenoisingColored(result, None, h=10, templateWindowSize=7, searchWindowSize=21) 
 
        # Define the sharpening kernel
        # kernel = np.array([[-1, -1, -1],
        #                 [-1, 9, -1],
        #                 [-1, -1, -1]])

        # # Apply the kernel to the image
        # sharpened = cv2.filter2D(result, -1, kernel)

        # cv2.imshow("sharpened Image", sharpened)
        # cv2.imshow("denoised Image", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        

        # result = cv2.

        with torch.inference_mode():
            # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
            transformed_image = image_transform(result).unsqueeze(dim=0)
            # 7. Make a prediction on image with an extra dimension and send it to the target device
            target_image_pred = model(transformed_image.to(device))

        # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

        # 9. Convert prediction probabilities -> prediction labels
        pred = torch.argmax(target_image_pred_probs, dim=1)

        # pred = model.predict(image)
        class_names = ['combat', 'destroyedbuilding', 'fire', 'humanitarianaid', 'militaryvehicles']
        event = class_names[pred]


        offset_x -= 10
        offset_y -= 10
        box = cv2.rectangle(marking_img, (offset_x, offset_y), (offset_x + w + 20, offset_y + h + 20), (0, 255, 0), 2)
        
        offset_y -= 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 2
        text = event
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)

        cv2.rectangle(marking_img, (offset_x, offset_y - text_height - 10), (offset_x + text_width, offset_y), (140, 133, 133), -1)
        cv2.putText(box, text, (offset_x, offset_y - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,0), thickness)
        identified_labels[letters[i]] = classconv[event]
        i+= 1

    cv2.imshow("Original Image", marking_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

##################################################
    return identified_labels


###############	Main Function	#################
if __name__ == "__main__":
    identified_labels = task_4a_return()
    print(identified_labels)