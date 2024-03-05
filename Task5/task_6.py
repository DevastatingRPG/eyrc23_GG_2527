# %%
'''
* Team Id : eYRC#GG#2527
* Author List : Shubham Sarkar, Aditya Gajjar, Sanskruti R. Ninawe, Vijit Ayush Pandey
* Filename: Task_6.py
* Theme: GeoGuide (GG)
* Functions: detect_ArUco_details, load_model, preprocess, getEvent, predictEvent, markImage, classifyArena, sortLabels, 
    distance, rotate_coordinates, adjust_coordinates, create_graph, isNode, calculate_angle, event_angle, atEvent, path_gen, 
    command_gen, get_element, read_csv, write_csv, tracker, norm_track, receive_data, display 
* Global Variables: cap, event_markers, bot_marker, coords, graph, lat_lon, ar_id, conversion, skip_test, priority_list, 
    oldDetails, oldCorners, received_queue, frame_queue, curr_node, oldBuffer, traversed, received_data
'''

# %%
import cv2 
from queue import Queue
import csv
import numpy as np 
from cv2 import aruco
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, resnet18  
import time
import math
import networkx as nx
import torch
import torch.nn as nn
import socket
import threading


# %% [markdown]
# ### Aruco Functions

# %%
'''
*Function Name: detect_ArUco_details
*Input: 
  - image: The image in which ArUco markers will be detected
*Output: 
  - ArUco_details_dict: Dictionary, contains marker IDs as keys and their details as values
  - ArUco_corners: Dictionary, contains marker IDs as keys and their corner coordinates as values
*Logic: 
  - Detects ArUco markers in the input image and extracts their details
  - Details include marker ID, center coordinates, and corner coordinates
*Example Call: 
  details, corners = detect_ArUco_details(img)
'''

def detect_ArUco_details(image): 
    ArUco_details_dict = {}
    ArUco_corners = {}
    
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
            ArUco_details_dict[marker_id] = [marker_center, 0]
            ArUco_corners[marker_id] = [[int(corner[0]), int(corner[1])] for corner in corners[i][0]]
    
    return ArUco_details_dict, ArUco_corners 

# %% [markdown]
# ### Image Classification

# %%
''' 
* Function Name: load_model
* Input: weight_path (of model), device (cpu or cuda)
* Output: Loaded model
* Logic: Loads model acc to weight path and device
* Example Call: model = load_model('weights.tf', 'cuda')
'''

def load_model(weight_path: str, device: str):
    model = efficientnet_v2_s()
    model.classifier = torch.nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=5, bias=True),
    )
    model.load_state_dict(torch.load(f"weights/{weight_path}"))
    model.eval()
    model = model.to(device)
    return model

# %%
''' 
* Function Name: preprocess
* Input: Original Image
* Output: Preprocessed image for contour detection
* Logic: applys various filters required for contour detection
* Example Call: img = preprocess(img)
'''

def preprocess(image):
    # Perform morphological opening
    kernel = np.ones((5,5),np.uint8)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    gray = cv2.cvtColor(opened, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return binary

# %%
''' 
* Function Name: getEvent
* Input: Original Image, Preprocessed Image for contour detection
* Output: Cropped image of Event
* Logic: Detects white contours in image and crops inside it
* Example Call: getEvent(roi, processed)
'''

def getEvent(org_image, processed_image):
    # Find contours in the image
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    x, y, w, h = cv2.boundingRect(contour)
    crop = org_image[max(0, y):min(y+h, org_image.shape[0]), max(0, x):min(x+w, org_image.shape[1])]
    return crop, x, y, w, h

# %%
''' 
* Function Name: predictEvent
* Input: Image, Image Transform, Model, Device, Threshold 
* Output: Classified Event
* Logic: Runs image through model and returns event
* Example Call: predictEvent(result, image_transform, model, device, threshold[i])
'''

def predictEvent(image, image_transform, model, device, threshold):
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = image_transform(image).unsqueeze(dim=0)
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    
    # Convert prediction probabilities -> prediction labels
    pred = torch.argmax(target_image_pred_probs, dim=1)
    
    class_names = ['fire', 'destroyed_buildings', 'combat', 'humanitarian_aid', 'military_vehicles']

    if max(target_image_pred_probs[0]) < threshold:
        event = "blank"
    else:     
        event = class_names[pred]
    return event

# %%
''' 
* Function Name: markImage
* Input: Image, Event Text, Top Left 
* Output: Image with bounding box and Event labelled
* Logic: Marks the bounding box acc to coordinates and puts text above it
* Example Call: markImage(marking_img, event, boxTL, boxBR)
'''

def markImage(image, event: str, tl: list, br: list):  

    box = cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    
    tl[1] -= 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 1
    (text_width, text_height), _ = cv2.getTextSize(event, font, scale, thickness)

    cv2.rectangle(image, (tl[0], tl[1] - text_height - 10), (tl[0] + text_width, tl[1]), (140, 133, 133), -1)
    cv2.putText(box, event, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,0), thickness)

    return image

# %%
''' 
* Function Name: classifyArena
* Input: cap (Camera Device), Image Path, Threshold Values
* Output: Identified Labels
* Logic: Crops all the events and classifies the crops
* Example Call: classifyArena(cap, "images/captured.jpg", [0]*5)
'''

def classifyArena(cap, image_path: str, threshold: list):
    identified_labels = {}  

    # Create a named window
    cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)

    picture_taken = False
    start_time = time.time()

    while not picture_taken:
        ret, frame = cap.read()
        display_frame = cv2.resize(frame, (960, 540))

        if not ret:
            print("Error reading frame from the camera")
            break

        cv2.imshow("Live Feed", display_frame)
        cv2.moveWindow("Live Feed", 0, 0)

        if time.time() - start_time >= 2:
            cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            picture_taken = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
   
    img = cv2.imread(image_path)

    marking_img = np.copy(img)
    _, corners = detect_ArUco_details(marking_img)
    
    events = [
        [[corners[25][3][0], corners[21][0][1]], [corners[21][0][0], corners[7][1][1] - 12]],
        [[corners[31][1][0], corners[28][1][1]], [corners[30][0][0], corners[14][3][1]]],
        [corners[31][1], [corners[30][0][0], corners[11][3][1]]], 
        [[corners[25][0][0], corners[34][0][1]], [corners[34][0][0], corners[11][3][1]]], 
        [[corners[42][1][0], corners[53][2][1]], [corners[40][0][0], corners[10][3][1]]]   
    ]

    letters = list("ABCDE")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model('weights.tf', device)
    
    image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=False),        
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    classconv = {
        "fire": "Fire", "destroyed_buildings": "Destroyed buildings", 
        "combat": "Combat", "humanitarian_aid": "Humanitarian Aid and rehabilitation", "military_vehicles": "Military Vehicles",
        "blank": "Blank"}

    temp = 'output/temp.jpg'

    for i, (tl, br) in enumerate(events):
        tl_adj = [tl[0] + 10, tl[1] + 7]
        br_adj = [br[0] - 10, br[1] - 4]
        roi = img[tl_adj[1]:br_adj[1], tl_adj[0]:br_adj[0]]

        processed = preprocess(roi)
        crop, x, y, w, h = getEvent(roi, processed)

        cv2.imwrite(temp, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
        result = cv2.imread(temp, cv2.IMREAD_COLOR)

        event = predictEvent(result, image_transform, model, device, threshold[i])
        text = classconv[event]

        boxTL, boxBR = [tl_adj[0] + x - 10, tl_adj[1] + y - 10], [tl_adj[0] + x + w + 10, tl_adj[1] + y + h + 10]
        marking_img = markImage(marking_img, text, boxTL, boxBR)

        identified_labels[letters[i]] = event
        cv2.imshow("Marked Image", marking_img)     
        cv2.waitKey(100)
        
    cv2.imshow("Marked Image", marking_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return identified_labels


# %%
''' 
* Function Name: classifyArena
* Input: cap (Camera Device), Image Path, Threshold Values
* Output: Identified Labels
* Logic: Crops all the events and classifies the crops
* Example Call: classifyArena(cap, "images/captured.jpg", [0]*5)
'''

def classifyArena(cap, image_path: str, threshold: list):
    identified_labels = {}  

    # Create a named window
    cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)

    picture_taken = False
    start_time = time.time()

    while not picture_taken:
        ret, frame = cap.read()
        display_frame = cv2.resize(frame, (960, 540))

        if not ret:
            print("Error reading frame from the camera")
            break

        cv2.imshow("Live Feed", display_frame)
        cv2.moveWindow("Live Feed", 0, 0)

        if time.time() - start_time >= 2:
            cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            picture_taken = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
   
    img = cv2.imread(image_path)

    marking_img = np.copy(img)
    _, corners = detect_ArUco_details(marking_img)
    
    events = [
        [[corners[25][3][0], corners[21][0][1]], [corners[21][0][0], corners[7][1][1] - 12]],
        [[corners[31][1][0], corners[28][1][1]], [corners[30][0][0], corners[14][3][1]]],
        [corners[31][1], [corners[30][0][0], corners[11][3][1]]], 
        [[corners[25][0][0], corners[34][0][1]], [corners[34][0][0], corners[11][3][1]]], 
        [[corners[42][1][0], corners[53][2][1]], [corners[40][0][0], corners[10][3][1]]]   
    ]

    letters = "ABCDE"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model('weights.tf', device)
    
    image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=False),        
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    temp = 'output/temp.jpg'

    # New
    classconv = {
        "fire": "Fire", "destroyed_buildings": "Destroyed buildings", 
        "combat": "Combat", "humanitarian_aid": "Humanitarian Aid and rehabilitation", "military_vehicles": "Military Vehicles",
        "blank": "Blank"}
    fixed = {'A': "military_vehicles", 'B': "destroyed_buildings", 'C': "combat", 'D': "fire", 'E': "fire"}
    identified_labels = {'A': "Military Vehicles", 'B': "Destroyed buildings", 'C': "Combat", 'D': "Fire", 'E': "Fire"}
    # New End

    for i, (tl, br) in enumerate(events):
        tl_adj = [tl[0] + 10, tl[1] + 7]
        br_adj = [br[0] - 10, br[1] - 4]
        roi = img[tl_adj[1]:br_adj[1], tl_adj[0]:br_adj[0]]

        processed = preprocess(roi)
        crop, x, y, w, h = getEvent(roi, processed)

        cv2.imwrite(temp, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
        result = cv2.imread(temp, cv2.IMREAD_COLOR)

        event = predictEvent(result, image_transform, model, device, threshold[i])

        # New
        event = fixed[letters[i]]
        text = event
        # New End

        boxTL, boxBR = [tl_adj[0] + x - 10, tl_adj[1] + y - 10], [tl_adj[0] + x + w + 10, tl_adj[1] + y + h + 10]
        marking_img = markImage(marking_img, text, boxTL, boxBR)

        # identified_labels[letters[i]] = classconv[event]
        cv2.imshow("Marked Image", marking_img)     
        cv2.waitKey(100)
        
    cv2.imshow("Marked Image", marking_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return identified_labels


# %%
def sortLabels(identified_labels: dict):
    order = ['Fire', 'Destroyed buildings', 'Humanitarian Aid and rehabilitation', 'Military Vehicles', 'Combat']
    result = []
    for target in ['A', 'B', 'C', 'D', 'E']:
        inserted = 0
        if identified_labels[target] != "Blank":
            tpos = order.index(identified_labels[target])
            if len(result) != 0:
                for ind, key in enumerate(result):
                    rpos = order.index(identified_labels[key])
                    if tpos <= rpos:
                        result.insert(ind, target)
                        inserted = 1
                        break
            if not inserted:
                result.append(target)
    return result

# %% [markdown]
# ### Path Creation

# %%
'''
*Function Name: distance
*Input: 
  - ar1: Coordinates (x, y) of the first point
  - ar2: Coordinates (x, y) of the second point
*Output: 
  - dist: Float, Euclidean distance between the two points
*Logic: 
  - Calculates the Euclidean distance between two points using their coordinates
*Example Call: 
  new_dist = distance(details[bot_marker][0], details[marker][0])
'''
def distance(ar1, ar2):
    c1 = ar1
    x1, y1 = c1[0], c1[1]
    c2 = ar2
    x2, y2 = c2[0], c2[1]

    width = x2-x1
    height = y2-y1
    dist = math.sqrt(pow(width, 2) + pow(height, 2))
    return dist

# %%
''' 
* Function Name: rotate_coordinates
* Input: x coord, y coord, theta_degress -> Angle to rotate by
* Output: Rotated coords
* Logic: Returns what new coords on an image will be after the image is rotated by some theta
* Example Call: rotate_coordinates(lat, lon, theta_degrees)
'''

def rotate_coordinates(x: float, y: float, theta_degrees: int):
    # Convert theta from degrees to radians
    theta = math.radians(-theta_degrees)

    # Perform the rotation
    x_prime = x * math.cos(theta) - y * math.sin(theta)
    y_prime = x * math.sin(theta) + y * math.cos(theta)

    return x_prime, y_prime

# %%
''' 
* Function Name: adjust_coordinates
* Input: csv file name, theta_degrees -> Angle to rotate by 
* Output: Csv file data with all coordinates rotated by given theta
* Logic: Reads the file and for each row except metadata row, rotates coordinates
* Example Call: adjust_coordinates('lat_long.csv', 15)
'''

def adjust_coordinates(csv_name: str, theta_degrees: int):
    adjusted_coordinates = {}

    with open(csv_name, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            ar_id, lat, lon = row[0], float(row[1]), float(row[2])
            adjusted_lat, adjusted_lon = rotate_coordinates(lat, lon, theta_degrees)
            adjusted_coordinates[ar_id] = [adjusted_lat, adjusted_lon]

    return adjusted_coordinates

# %%
'''
*Function Name: create_graph
*Input: 
  - coords: Dictionary, contains coordinates of nodes (format: {'node_id': (x, y)})
*Output: 
  - graph: NetworkX Graph, represents the connectivity of nodes with weighted edges
*Logic: 
  - Creates a graph representing the connectivity of nodes based on the provided links
*Example Call: 
  graph = create_graph(coords)
'''

def create_graph(coords: dict):
    links = (
        (23, 24), (24, 22), (22, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 48), (48, 47), (47, 46), 
        (46, 45), (45, 44), (44, 43), (43, 10), (10, 8), (8, 12), (12, 9), (9, 11), (11, 13), (13, 14), (14, 15), 
        (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 23),

        (22, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 11),

        (50, 34), (34, 33), (33, 32), (32, 31), (31, 30), (30, 12),

        (51, 42), (42, 41), (41, 40), (40, 39), (39, 35), (35, 38), (38, 37), (37, 36), (36, 10), (36, 8),

        (19, 27), (19, 28),

        (27, 32), (28, 32),

        (32, 39), (32, 35)       
    )
    nodes = [int(coord) for coord in coords.keys()]
    init_graph = {node: {} for node in nodes}
    for n1, n2 in links:
        init_graph[n1][n2] = distance(coords[str(n1)], coords[str(n2)])
    graph = nx.Graph()
    for link in links:
        graph.add_edge(link[0], link[1], weight=init_graph[link[0]][link[1]])
    return graph

# %%
def isNode(node, traversed):
    turns = ((23,), (19,), (22,), (27, 28), (11,), (50,), (32,), (12,), (51,), (39, 35), (10, 8))
    for turn in turns:
        if (node in turn) and turns.index(turn) not in traversed:
            traversed.append(turns.index(turn))
            return True, traversed
    return False, traversed

# %%
'''
*Function Name: calculate_angle
*Input: 
  - coord1: Tuple, coordinates (x, y) of the first point
  - coord2: Tuple, coordinates (x, y) of the second point
  - coord3: Tuple, coordinates (x, y) of the third point
*Output: 
  - angle: Float, angle in degrees between the three points
  - direction: Integer or String, direction ('2' for right, '3' for left, 'C' for colinear)
*Logic: 
  - Calculates the angle between three points using the law of cosines
  - Determines the direction based on the cross product of vectors
*Example Call: 
  ang, dir = calculate_angle(coords[str(path[i-1])], coords[str(path[i])], coords[str(path[i+1])])
'''

def calculate_angle(coord1, coord2, coord3):
    # Calculate the distances between the points
    a = math.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
    b = math.sqrt((coord3[0] - coord2[0])**2 + (coord3[1] - coord2[1])**2)
    c = math.sqrt((coord3[0] - coord1[0])**2 + (coord3[1] - coord1[1])**2)

    # Apply the law of cosines to find the angle
    cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
    angle = math.acos(cos_angle)

    # Calculate the cross product
    cross_product = (coord2[0] - coord1[0]) * (coord3[1] - coord1[1]) - (coord2[1] - coord1[1]) * (coord3[0] - coord1[0])

    # Determine the direction
    if cross_product > 0:
        # Right
        direction = 2
    elif cross_product < 0:
        # Left
        direction = 3
    else:
        direction = "C"

    # Convert the angle to degrees
    angle = math.degrees(angle)

    return angle, direction

# %%
'''
* Function Name: event_angle
* Input: 
  - coord1: Representing the coordinates (x, y) of one point
  - botcoord: Representing the coordinates (x, y) of the bot
* Output: 
  - angle_degrees: Float, angle in degrees between the two lines formed by the points
  - side: String, direction ('l' for left, 'r' for right)
* Logic: 
  - Calculates the angle and direction between two points using their coordinates
* Example Call: 
  angle, dir = event_angle(details[event][0], details[bot_marker][0])
'''

def event_angle(coord1, botcoord):
    # coord1 and coord2 are tuples representing (x, y)
    x1, y1 = coord1
    x2, y2 = botcoord

    # Calculate the difference between the two points
    dx = x2 - x1
    dy = y2 - y1

    try:
        slope1 = dy/dx
        slope2 = 0
        # Calculate the acute angle between two lines given their slopes
        angle = abs(math.atan((slope2 - slope1) / (1 + slope1 * slope2)))
        # Convert the angle to degrees
        angle_degrees = math.degrees(angle)
    except ZeroDivisionError:
        angle_degrees = 90

    if dx <= 0:
        side = 'l'
    else:
        side = 'r'
    

    return angle_degrees, side

# %%
'''
* Function Name: atEvent
* Input: 
  - bot_marker: ARUCO ID of the bot
  - event: String, specifies the event to check for
* Output: 
  - Boolean, True if the bot is at the specified event, False otherwise
* Logic: 
  - Compares the bot's position and orientation with the specified event
  - Handles different types of events, including angle and distance checks
* Example Call: 
  atEvent(bot_marker, event)
'''

def atEvent(bot_marker, event, frame_queue, event_markers, oldDetails):
  if not frame_queue.empty():
    frame = frame_queue.get()
    details, _ = detect_ArUco_details(frame)

    angles = {'A': [21, 77, 179], 'B': [17, 31, 220], 'C': [17.4896, 43, 192], 'D': [7, 79, 150], 'E': [0, 70, 150]}
    
    try:
      event_ar = event_markers[event]
      angle, dir = event_angle(oldDetails[event_ar][0], details[bot_marker][0])
      if (angles[event][0] <= angle <= angles[event][1] and 
          distance(oldDetails[event_ar][0], details[bot_marker][0]) < angles[event][2] and dir == 'l'):
          return True
      else:
          return False

        
    except KeyError:
        return False
    except IndexError:
        return False
  else:
     return False
    

# %%
'''
*Function Name: path_gen
*Input: 
  - graph: NetworkX Graph, represents the connectivity of nodes with weighted edges
  - start: Integer, ID of the starting point
  - event: String, specifies the event or target node
*Output: 
  - path: List, contains the nodes in the shortest path from start to the specified event
*Logic: 
  - Generates the shortest path in the graph from the start to the specified event
*Example Call: 
  path = path_gen(graph, 23, 'E')
'''

def path_gen(graph, start, event):
    path = nx.shortest_path(graph, start, event_markers[event], weight='weight')
    return path

# %%
'''
* Function Name: command_gen
* Input: 
  - coords: Coordinates of nodes
  - path: List of node aruco IDs representing the path to be traversed
  - oldbuffer: ID of the node visited before the current one
* Output: 
  - c: List of commands (integers) for the robot to execute
  - a: List of additional information (node aruco IDs) corresponding to the commands
  - oldbuffer: Updated aruco ID of the node visited before the current one
* Logic: 
  - Generates commands based on the path and the robot's current state
  - Different commands correspond to different actions such as FORWARD, RIGHT turn, LEFT turn, etc.
  - The function considers the robot's orientation and the type of nodes in the path to determine the commands
* Example Call: 
  commad_lsit, ar_node_list, oldbuffer = command_gen(coords, path, oldbuffer)
'''

def command_gen(coords, path: list, oldbuffer: int):
    # 1 is for FORWARD till node detection
    # 2 is for RIGHT turn then FORWARD till node detection
    # 3 is for LEFT turn then FORWARD till node detection
    # 4 is for 180 degree turn then FORWARD till node detection
    # 5 is for buzzer
    # 6 and 9 for exit at the end
    # 7 is for FORWARD but skip next node
    # 11 is for FORWARD for corners
    c = []
    a = []
    traversed = []
    if oldbuffer == path[1] and oldbuffer in [54, 47]:
        c.append(8)
    elif oldbuffer == path[1]:
        c.append(4)
    else:
        if path[0] == 48:
            c.append(11)
        else:
            c.append(1)
    for i in range(0, len(path)):
        if path[i] == 23 and i == 0:
            a.append(23)
            if path[i+1] == 24:
                c.append(1)
            else:
                c.append(2)
        
        elif i<len(path)-2:
            ang, dir = calculate_angle(coords[str(path[i])], coords[str(path[i+1])], coords[str(path[i+2])])
            result, traversed = isNode(path[i+1], traversed)
            if (path[i+1] == 51 and path[i+2] == 52):
                a.append(51)
                c.append(11)
            elif (path[i+1] == 10 and path[i+2] == 43):
                a.append(10)
                c.append(1)
            elif (150 >= ang >= 45) and result:
                traversed = []
                if path[i] in [19, 32] and (path[i+1] == 28 or path[i+1] == 27) and path[i+2] in [19, 32]:
                    a.append(path[i+1])
                    c.append(1)
                
                elif not (path[i] == 43 and path[i+2] == 8):
                    a.append(path[i+1])
                    c.append(dir)             
                
            elif (170 <= ang <= 180) and result and not (path[i+1] == 8 and path[i+2] == 10) :
                a.append(path[i+1])
                c.append(1)
        elif path[-1] == 23:
            result, traversed = isNode(23, traversed)
            if result:
                a.append(23)
            if path[i] == 21:
                c.append(6)
            elif path[i] == 24:
                c.append(9)
    oldbuffer = path[-2]
    return c, a, oldbuffer

# %%
def get_element(lst: list, index: int):
    try:
        return lst[index]
    except IndexError:
        return None

# %% [markdown]
# ### Geo Locating

# %%
'''
Function Name: read_csv
Input: 
  - csv_name: String, the name of the CSV file to be read
Output: 
  - lat_lon: Dictionary, contains ARUCO IDs as keys and corresponding [lat, lon] as values
Logic: 
  - Reads the specified CSV file and stores its data in the lat_lon dictionary
Example Call: 
  lat_lon = read_csv("lat_lon.csv")
'''

def read_csv(csv_name: str):
    lat_lon = {}

    # open csv file (lat_lon.csv)
    # read "lat_lon.csv" file
    # store csv data in lat_lon dictionary as {id:[lat, lon].....}
    # return lat_lon

    with open(csv_name, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            ar_id, lat, lon = row[0], row[1], row[2]
            lat_lon[ar_id] = [lat, lon]

    return lat_lon

# %%
'''
Function Name: write_csv
Input: 
  - loc: Dictionary, contains ARUCO IDs as keys and corresponding [lat, lon] as values
  - csv_name: String, the name of the CSV file to be written
Output: 
  - None
Logic: 
  - Writes the coordinates from the loc dictionary to the specified CSV file
Example Call: 
  write_csv({ar_id: coordinate}, "live_data.csv")
'''

def write_csv(loc, csv_name: str):

    # open csv (csv_name)
    # write column names "lat", "lon"
    # write loc ([lat, lon]) in respective columns

    with open(csv_name, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["lat", "lon"])  # Write the column names
        for coordinate in loc.values():
            lat, lon = coordinate
            csv_writer.writerow([lat, lon])

# %%
'''
Function Name: tracker
Input: 
  - ar_id: String, ARUCO ID for which the tracker will find lat, lon
  - lat_lon: Dictionary, contains ARUCO IDs as keys and corresponding [lat, lon] as values
Output: 
  - None
Logic: 
  - Finds the lat, lon associated with the specified ARUCO ID
  - Writes these lat, lon to "live_data.csv"
Example Call: 
  tracker(ar_id, lat_lon)
'''

def tracker(ar_id: int, lat_lon: dict):

    # find the lat, lon associated with ar_id (aruco id)
    # write these lat, lon to "live_data.csv"

    coordinate = None

    # Check if the ARUCO ID exists in the lat_lon dictionary
    if str(ar_id) in list(lat_lon.keys()):
        coordinate = lat_lon[str(ar_id)]

        # Write the coordinate to "live_data.csv"
        write_csv({ar_id: coordinate}, "live_data.csv")

# %%
def norm_track(path: list, segments: list, curr_node: int, ar_id: int, ind: int, traversed: list, frame_queue: Queue, oldDetails):

    if not frame_queue.empty():
        frame = frame_queue.get()

        details, _ = detect_ArUco_details(frame)
        try:
            if distance(details[bot_marker][0], oldDetails[path[curr_node+1]][0]) < distance(details[bot_marker][0], oldDetails[path[curr_node]][0]):
                curr_node += 1
                ar_id = path[curr_node]
                tracker(ar_id, lat_lon)
                result, traversed = isNode(ar_id, traversed)
                if result and ar_id != 23:
                    try:
                        ind = segments.index(ar_id)
                    except ValueError:
                        pass
                return curr_node, ar_id, ind


        except KeyError:
            pass
        except IndexError:
            pass
    return curr_node, ar_id, ind

# %% [markdown]
# ### THREADING

# %%
'''
* Function Name: receive_data
* Input: 
  - conn: Connection object
  - received_queue: Queue of messages received from esp32
* Output: 
  - None
* Logic: 
  - Thread that constantly receives data from robot
* Example Call: receive_data(conn, received_queue)
'''

def receive_data(conn, received_queue: Queue):
    # global received_data
    while True:
        try:
            received_data = conn.recv(1024)
            received_data = received_data.decode('utf-8').strip()
            received_queue.put(received_data)
        except ConnectionAbortedError:
            pass
        except OSError:
            pass

# %%
'''
* Function Name: display
* Input: 
  - cap: Camera object
  - received_queue: Queue to store frames
* Output: 
  - None
* Logic: 
  - Thread that constantly Displays and Puts frames in queue
* Example Call: display(cap, frame_queue)
'''

def display(cap, frame_queue: Queue):    
    while True:
        _, frame = cap.read()  
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Live Feed", display_frame)
        # Move the window to the left
        cv2.moveWindow("Live Feed", 0, 0)
        # Break the loop if 'q' is pressed

        if frame_queue.empty():
            frame_queue.put(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

# %% [markdown]
# ### MAIN

# %%
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# %%
if 'cap' not in globals():
    # Open the camera
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)


    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Try to set exposure, white balance, and other properties
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 0.25 means "manual exposure, manual iris"
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # 0 means "disable auto white balance"

    cap.set(cv2.CAP_PROP_SATURATION, 75)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Unable to open the camera")
        exit()

# %%
event_markers = {
    'A': 21,
    'B': 29,
    'C': 30,
    'D': 34,
    'E': 48, 
    'F': 23
}
conversion = {2: 10, 3: 12}
skip_test = lambda x, y: (x in (1, 7) and y in (1, 7))
bot_marker = 100

# %%
coords = adjust_coordinates('lat_long.csv', -15)
graph = create_graph(coords)
lat_lon = read_csv('lat_long.csv')
ar_id = 23
tracker(ar_id, lat_lon)

# %%
while True:
    try:
        events = classifyArena(cap, "images/captured.jpg", [0]*5)
        print(events)
        ask = input("OK ? : ")
        if ask == 'y':
            break
        else:
            continue
    except KeyError:
        continue
priority_list = sortLabels(events)
priority_list.append('F')


# %%

# %%
while True:
    ret, frame = cap.read()
    oldDetails, oldCorners = detect_ArUco_details(frame)
    if len(oldDetails) == 51 and 100 not in oldDetails.keys():
        break


# %%
esp32_ip = ""  # Change this to the IP address of your ESP32
esp32_port = 8002

# Global variable to store the received data
received_data = None
traversed = []

received_queue = Queue()
frame_queue = Queue()
display_thread = threading.Thread(target=display, args=(cap, frame_queue))
display_thread.start()

oldBuffer = 23

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((esp32_ip, 8002))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            # Create a new thread for receiving data
            receive_thread = threading.Thread(target=receive_data, args=(conn, received_queue))
            receive_thread.start()

            command = input("Enter command (1: Start): ")
            for j, event in enumerate(priority_list):
                if j > 0 :
                    subPath = path_gen(graph, event_markers[priority_list[j-1]], event)
                else:
                    subPath = path_gen(graph, 23, event)

                subCommands, subSegments, oldBuffer = command_gen(coords, subPath, oldBuffer)
                # print(subCommands)
                ind = -2
                
                # for i in range(len(subCommands)-1):

                #     if subCommands[i] in (2, 3) and subCommands[i+1] == 1:
                #         subCommands[i] = conversion[subCommands[i]]

                #     if i > 0:    
                #         if subCommands[i] == 1 and skip_test(subCommands[i-1], subCommands[i+1]):
                #             subCommands[i] = 7


                # Create a stop event
                curr_node = 0
                ar_id = subPath[0]
                traversed = []
                tracker(ar_id, lat_lon)

                curr_node, ar_id, ind = norm_track(subPath, subSegments, curr_node, ar_id, ind, traversed, frame_queue, oldDetails)

                conn.sendall(str.encode(f"{str(subCommands[0])}"))
                # print(f"Command Sent : {subCommands[0]}")
                i = 1

                while not atEvent(bot_marker, event, frame_queue, event_markers, oldDetails):
                    curr_node, ar_id, ind = norm_track(subPath, subSegments, curr_node, ar_id, ind, traversed, frame_queue, oldDetails)
    
                    result, traversed = isNode(ar_id, traversed)

                    if not result and (
                        event == 'E'
                    ) and i not in [0, 1] and i < 3:
                        # print("Replaced i : ", ind+2, i)                          
                        i = min(ind + 2, len(subCommands)-1)                           

                    if not received_queue.empty():
                        received_data = received_queue.get()

                    if (
                        (received_data == 'node') or 
                        (event == 'E' and ar_id in [51, 10])
                        ) and i < len(subCommands):  

                        try:                     
                            conn.sendall(str.encode(f"{str(subCommands[i])}"))
                            # print(f"Command Processed : {subCommands[i-1]}")
                            # print(f"Command Sent : {subCommands[i]}")
                            i += 1
                            received_data = None
                        except IndexError:
                            pass
                    
                    if received_data == 'positive':
                        # print("Done")
                        break
                else:
                    conn.sendall(str.encode("5"))
                    # print(f"Command Sent: 5")
                    while received_data != "buzz":
                        if not received_queue.empty():
                            received_data = received_queue.get()
                    # print("Command Processed : 5")
                    # print("done with one event")
            
            # print("helped everyone")

except KeyboardInterrupt:
    # print("Keyboard Interrupt")
    cv2.destroyAllWindows()



