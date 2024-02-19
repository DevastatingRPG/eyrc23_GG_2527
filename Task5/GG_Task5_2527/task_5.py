# %%
import cv2 
import csv
import numpy as np 
from cv2 import aruco
from torchvision import transforms
from torchvision.models import efficientnet_v2_s  
import time
import math
import networkx as nx
import torch
import torch.nn as nn
import socket
import threading
import pytesseract

# %%
pytesseract.pytesseract.tesseract_cmd = r'B:\Software\Tesseract\tesseract.exe'

# %% [markdown]
# ### Aruco Functions

# %%
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
            ArUco_details_dict[marker_id] = [marker_center, 0]
            ArUco_corners[marker_id] = [[int(corner[0]), int(corner[1])] for corner in corners[i][0]]
    ##################################################
    
    return ArUco_details_dict, ArUco_corners 

# %%
def mark_ArUco_image(image,ArUco_details_dict, ArUco_corners):

    for ids, details in ArUco_details_dict.items():
        center = details[0]

        corner = ArUco_corners[int(ids)]

        tl_tr_center_x = int((corner[0][0] + corner[1][0]) / 2)
        tl_tr_center_y = int((corner[0][1] + corner[1][1]) / 2) 

        display_offset = int(math.sqrt((tl_tr_center_x - center[0])**2+(tl_tr_center_y - center[1])**2))
        cv2.putText(image,str(ids),(center[0]+int(display_offset/2),center[1]),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    return image

# %% [markdown]
# ### Image Classification

# %%
def task_4a_return(image_path, threshold):
    global cap
    identified_labels = {}  
    
    ret, frame = cap.read()
    display_frame = cv2.resize(frame, (960, 540))

    # Create a named window
    cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)

    # Flag to check if the picture has been taken
    picture_taken = False
    # Get start time
    start_time = time.time()
    # Read and display frames from the camera

    while not picture_taken:
        ret, frame = cap.read()
        display_frame = cv2.resize(frame, (960, 540))

        if not ret:
            print("Error reading frame from the camera")
            break

        cv2.imshow("Live Feed", display_frame)

        # Move the window to the left
        cv2.moveWindow("Live Feed", 0, 0)

        if time.time() - start_time >= 5:
            cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            picture_taken = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()

    
    img = cv2.imread("images/captured.jpg")
    # os.remove(image_path)
    cv2.imshow("Marked", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    _, corners = detect_ArUco_details(img)
    # mark = mark_ArUco_image(img,  details, corners)

    marking_img = np.copy(img)
    _, corners = detect_ArUco_details(marking_img)
    
    events = [
        [[corners[25][3][0], corners[21][0][1]], [corners[21][0][0], corners[7][1][1]-10]],
        [[corners[31][1][0], corners[28][1][1]], [corners[30][0][0], corners[14][3][1]]],
        [corners[31][1], [corners[30][0][0], corners[11][3][1]]], 
        [[corners[25][0][0], corners[34][0][1]], [corners[34][0][0], corners[11][3][1]]], 
        [[corners[42][1][0], corners[52][1][1]], [corners[40][0][0], corners[10][3][1]-30]]   
    ]

    i=1
    eventlist=[]
    letters = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    classconv = { "combat": "Combat", "destroyedbuilding": "Destroyed buildings", 
                 "humanitarianaid": "Humanitarian Aid and rehabilitation",
                 "militaryvehicles": "Military Vehicles", "fire": "Fire", "blank": "Blank"}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = efficientnet_v2_s().to(device)
    
    model.classifier = torch.nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=5, bias=True),
    ).to(device)


    # model.classifier = torch.nn.Sequential(
    #     nn.Linear(1280, 256), # Additional linear layer with 256 output features
    #     # nn.ReLU(inplace=True),         # Activation function
    #     nn.Dropout(p=0.5, inplace=True),
    #     nn.Linear(256, 5)
    # ).to(device)

    model.load_state_dict(torch.load('weights/weights.tf'))
    # model.load_state_dict(torch.load('weights/w1.tf'))


    model.eval()
    
    image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=False),
            
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    temp = 'output/temp.jpg'

    for i, (tl, br) in enumerate(events):
        tl_adj = [tl[0] + 10, tl[1] + 7]
        br_adj = [br[0] - 10, br[1] - 4]
        roi = img[tl_adj[1]:br_adj[1], tl_adj[0]:br_adj[0]]

        reference = cv2.imread(f"images/empty/{i}.jpg")
        
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

        # Convert the cropped image to HSV color space
        # hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # lower_green = np.array([0, 50, 0])
        # upper_green = np.array([50, 100, 50])

        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        lower_green = np.array([200, 200, 200])
        upper_green = np.array([255, 255, 255])

        # Threshold the RGB image to get only green colors
        mask = cv2.inRange(img_rgb, lower_green, upper_green)

        # Calculate the percentage of green pixels
        green_percentage = (np.count_nonzero(mask) / (crop.shape[0] * crop.shape[1])) * 100
        # print(green_percentage)

        # If the majority of the cropped area is green, skip the classification
        threshold_percentage = 80  # Adjust the threshold as needed
        cv2.imwrite(temp, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
        crop = cv2.imread(temp)




        eventlist.append(crop)
        cv2.imwrite(temp, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
        result = cv2.imread(temp, cv2.IMREAD_COLOR)
        # result = super_resolution.cartoon_upsampling_4x(temp, temp )


        with torch.inference_mode():
            # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
            transformed_image = image_transform(result).unsqueeze(dim=0)
            # 7. Make a prediction on image with an extra dimension and send it to the target device
            target_image_pred = model(transformed_image.to(device))

        # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        
        # 9. Convert prediction probabilities -> prediction labels
        pred = torch.argmax(target_image_pred_probs, dim=1)
        class_names = ['combat', 'destroyedbuilding', 'fire', 'humanitarianaid', 'militaryvehicles']

        print(target_image_pred_probs[0])

        if max(target_image_pred_probs[0]) < threshold[i]:
            event = "blank"
        else:
            
            event = class_names[pred]

        offset_x = tl_adj[0] + x - 10
        offset_y = tl_adj[1] + y - 10    

        box = cv2.rectangle(marking_img, (offset_x, offset_y), (offset_x + w + 20, offset_y + h + 20), (0, 255, 0), 2)
        
        offset_y -= 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 1
        text = classconv[event]
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)

        cv2.rectangle(marking_img, (offset_x, offset_y - text_height - 10), (offset_x + text_width, offset_y), (140, 133, 133), -1)
        cv2.putText(box, text, (offset_x, offset_y - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,0), thickness)
        identified_labels[letters[i+1]] = classconv[event]
        # display_frame = cv2.resize(marking_img, (960, 540))
        cv2.imshow("Marked Image", marking_img)
        
        cv2.waitKey(500)  # delay for 500 milliseconds
        
    # display_frame = 
    cv2.imshow("Marked Image", marking_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return identified_labels


# %%
def sort_labels(identified_labels):
    global order
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
def rotate_coordinates(x, y, theta_degrees):
    # Convert theta from degrees to radians
    theta = math.radians(-theta_degrees)

    # Perform the rotation
    x_prime = x * math.cos(theta) - y * math.sin(theta)
    y_prime = x * math.sin(theta) + y * math.cos(theta)

    return x_prime, y_prime

# %%
def adjust_coordinates(csv_name, theta_degrees):
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
def create_graph(coords):
    links = (
        (23, 24), (24, 22), (22, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 48), (48, 47), (47, 46), 
        (46, 45), (45, 44), (44, 43), (43, 10), (10, 8), (8, 12), (12, 9), (9, 11), (11, 13), (13, 14), (14, 15), 
        (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 23),

        (22, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 11),

        (50, 34), (34, 33), (33, 32), (32, 31), (31, 30), (30, 12),

        (51, 42), (42, 41), (41, 40), (40, 39), (39, 35), (35, 38), (38, 37), (37, 36), (36, 10), (36, 8),

        (19, 27), (19, 28), (19, 32),

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
def atEvent(bot_marker, event):
    global event_markers
    global frame
    details, _ = detect_ArUco_details(frame)

    try:
        event = event_markers[event]
        angle, dir = event_angle(details[event][0], details[bot_marker][0])
        if (19 <= angle <= 71) and distance(details[event][0], details[bot_marker][0]) < 150 and dir == 'l':
            return True
        else:
            return False
        
    except KeyError:
        return False
    except IndexError:
        return False
    

# %%
def path_gen(graph, events):
    global event_markers
    path = [[] for _ in range(len(events)+1)]
    curr_node = 23
    for ind, event in enumerate(events):
        path[ind].extend(nx.shortest_path(graph, curr_node, event_markers[event], weight='weight'))
        curr_node = event_markers[event]
    path[-1].extend(nx.shortest_path(graph, curr_node, 23, weight='weight'))
    return path

# %%
def command_gen(coords, paths):
    # 1 is for FORWARD till node detection
    # 2 is for RIGHT turn then FORWARD till node detection
    # 3 is for LEFT turn then FORWARD till node detection
    # 4 is for 180 degree turn then FORWARD till node detection
    commands = []
    buffer = 0
    for path in paths:
        c = []
        traversed = []
        if buffer == path[1]:
            c.append(4)
        else:
            c.append(1)
        for i in range(0, len(path)):

            if path[i]==23 and i==0:
                if path[i+1] == 24:
                    c.append(1)
                else:
                    c.append(2)
            elif i<len(path)-2:
                ang, dir = calculate_angle(coords[str(path[i])], coords[str(path[i+1])], coords[str(path[i+2])])
                result, traversed = isNode(path[i+1], traversed)
                if (150 >= ang >= 45) and result:
                    traversed = []
                    if not (path[i] == 43 and path[i+2] == 8):
                        c.append(dir)
                    if path[i+2] == 32 and path[i+1] == 19:
                        c.append(1)
                    
                if (170 <= ang <= 180) and result:
                    c.append(1)
            elif path[-1] == 23:
                if path[i] == 21:
                    c.append(3)
                elif path[i] == 24:
                    c.append(1)
        buffer = path[-2]
        commands.append(c)
    return commands

# %%
def get_element(lst, index):
    try:
        return lst[index]
    except IndexError:
        return None

# %% [markdown]
# ### Geo Locating

# %%
def read_csv(csv_name):
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
def write_csv(loc, csv_name):

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
def tracker(ar_id, lat_lon):

    # find the lat, lon associated with ar_id (aruco id)
    # write these lat, lon to "live_data.csv"

    coordinate = None

    # Check if the ARUCO ID exists in the lat_lon dictionary
    if str(ar_id) in list(lat_lon.keys()):
        coordinate = lat_lon[str(ar_id)]

        # Write the coordinate to "live_data.csv"
        write_csv({ar_id: coordinate}, "live_data.csv")

# %%
def norm_track(path):
    global curr_node
    global bot_marker
    global lat_lon
    global frame
    global ar_id
    
    details, _ = detect_ArUco_details(frame)
    try:

        if distance(details[bot_marker][0], details[path[curr_node+1]][0]) < distance(details[bot_marker][0], details[path[curr_node]][0]):
            curr_node += 1
            ar_id = path[curr_node]
            tracker(ar_id, lat_lon)

        if distance(details[bot_marker][0], details[path[curr_node-1]][0]) < distance(details[bot_marker][0], details[path[curr_node]][0]):
            curr_node -= 1
            ar_id = path[curr_node]
            tracker(ar_id, lat_lon)

    except KeyError:
        pass
    except IndexError:
        pass

# %% [markdown]
# ### THREADING

# %%
# Function to handle data receiving
def receive_data(conn):
    global received_data
    while True:
        try:
            received_data = conn.recv(1024)
            received_data = received_data.decode('utf-8').strip()
        except ConnectionAbortedError:
            pass
        except OSError:
            pass

# %%
# Function to display Live Feed
def display():
    global ret
    global frame
    global cap
    
    while True:
        ret, frame = cap.read()  
        display_frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Live Feed", display_frame)
        # Move the window to the left
        cv2.moveWindow("Live Feed", 0, 0)
        # Break the loop if 'q' is pressed
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
    'E': 48
}
conversion = {1: 10, 2: 7, 3: 8}
bot_marker = 100
order = ['Fire', 'Destroyed buildings', 'Humanitarian Aid and rehabilitation', 'Military Vehicles', 'Combat']

# %%
coords = adjust_coordinates('lat_long.csv', -15)
graph = create_graph(coords)
lat_lon = read_csv('lat_long.csv')
ar_id = 23
tracker(ar_id, lat_lon)


# %%
while True:
    try:
        # events = task_4a_return("images/captured.jpg", [0.5, 0.5, 0.35, 0.25, 1])
        events = task_4a_return("images/captured.jpg", [0.45, 0.41, 0.35, 0.4, 0.45])
        # events = task_4a_return("images/captured.jpg", [0, 0, 0, 0, 0])

        # events = task_4a_return("images/captured.jpg", [1, 1, 1, 1, 1])
        print(events)
        ask = input("OK ? : ")
        if ask == 'y':
            break
        else:
            continue
    except Exception as e:
        print(e)
priority_list = sort_labels(events)
path = path_gen(graph, priority_list)
command_list = command_gen(coords, path)


# %%
# priority_list = list('EBADC')
# path = path_gen(graph, priority_list)
# command_list = command_gen(coords, path)

# %%
esp32_ip = ""  # Change this to the IP address of your ESP32
esp32_port = 8002

# Global variable to store the received data
received_data = None

display_thread = threading.Thread(target=display, args=())
display_thread.start()

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((esp32_ip, 8002))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            # Create a new thread for receiving data
            receive_thread = threading.Thread(target=receive_data, args=(conn,))
            receive_thread.start()

            command = input("Enter command (1: Move Forward): ")
            print("hallo")
            print(command_list)
            for j, (sub, subPath) in enumerate(zip(command_list, path)):
                # Create a stop event
                curr_node = 0
                ar_id = subPath[0]
                tracker(ar_id, lat_lon)

                norm_track(subPath)
                i = 0
                while i < len(sub):
                # for i in enumerate(sub):
                    subsub = sub[i]
                    norm_track(subPath)
                    received_data = None
                    if i == len(sub) - 1:
                        if get_element(priority_list, j) == 'E':
                            conn.sendall(str.encode(str(11)))
                            print(f"command: 11")
                        elif j == len(command_list) - 1:
                            if subsub == 3:
                                conn.sendall(str.encode(str(6)))
                                print(f"command: 6")
                            elif subsub == 1:
                                conn.sendall(str.encode(str(9)))
                                print(f"command: 9")
                        else:
                            conn.sendall(str.encode(str(conversion[subsub])))
                            print(f"command: {conversion[subsub]}")
                        if subsub != 1:
                            while received_data != "turned":
                                norm_track(subPath)
                                continue
                        print(f"command processed: {conversion[subsub]}")
                    else:
                        if get_element(priority_list, j-1) == 'E' and i==0:
                            conn.sendall(str.encode(str(12)))
                            print(f"command: 12")
                        else:
                            conn.sendall(str.encode(str(subsub)))
                            print(f"command: {subsub}")
                        while received_data != "node" and not (ar_id == 51 and get_element(priority_list, j) == 'E'):
                            norm_track(subPath)
                            continue
                        if (ar_id in [11, 29, 9]) and get_element(priority_list, j) == 'B':
                            i = len(sub) - 2
                        print(f"command processed: {subsub}")
                    norm_track(subPath)
                    print(received_data)
                    i += 1

                while True:
                    try:
                        norm_track(subPath)
                        if atEvent(bot_marker, priority_list[j]):
                            break
                    except IndexError:
                        norm_track(subPath)
                        continue

                norm_track(subPath)
                conn.sendall(str.encode(str(5)))
                print(f"Sent: 5")
                while received_data != "buzz":
                    norm_track(subPath)
                    continue
                norm_track(subPath)
                print("done with one event")
            
            print("helped everyone")

except KeyboardInterrupt:
    print("Keyboard Interrupt")
    cv2.destroyAllWindows()


