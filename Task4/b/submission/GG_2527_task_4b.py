# %%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import csv
import numpy as np 
import ctypes
from cv2 import aruco
import math 

# %%
def distance(ar1, ar2):
    c1 = ar1[0]
    x1 = c1[0], y1 = c1[1]
    c2 = ar2[0]
    x2 = c2[0], y2 = c2[1]

    width = x2-x1
    height = y2-y1
    dist = math.sqrt(pow(width, 2) + pow(height, 2))
    return dist
    

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

def tracker(ar_id, lat_lon):

    # find the lat, lon associated with ar_id (aruco id)
    # write these lat, lon to "live_data.csv"

    coordinate = None

    # Check if the ARUCO ID exists in the lat_lon dictionary
    if str(ar_id) in list(lat_lon.keys()):
        coordinate = lat_lon[str(ar_id)]
        
        # Write the coordinate to "live_data.csv"
        write_csv({ar_id: coordinate}, "live_data.csv")

    # also return coordinate ([lat, lon]) associated with respective ar_id.
    return coordinate

# %%
def calculate_angle(corners):
    p1, p2 = corners[0], corners[1]
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    return math.degrees(angle)

##############################################################

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
            marker_angle = int(calculate_angle(corners[i][0]))  # Calculate the angle

            # Store details in dictionaries
            ArUco_details_dict[marker_id] = [marker_center, marker_angle]
            ArUco_corners[marker_id] = [[int(corner[0]), int(corner[1])] for corner in corners[i][0]]
    ##################################################
    
    return ArUco_details_dict, ArUco_corners 


# %%
from task_2a import mark_ArUco_image

# %%
path = [23, 24, 22, 49, 34, 33, 35, 38, 37, 36, 8, 12, 9, 11, 29, 28, 27, 26, 25, 22, 24, 23]
bot_marker = 69
lat_lon = read_csv('lat_long.csv')
tracker(23, lat_lon)

# %%
def distance(ar1, ar2):
    c1 = ar1[0]
    x1, y1 = c1[0], c1[1]
    c2 = ar2[0]
    x2, y2 = c2[0], c2[1]

    width = x2-x1
    height = y2-y1
    dist = math.sqrt(pow(width, 2) + pow(height, 2))
    return dist
    

# %%
# Get screen size
# user32 = ctypes.windll.user32
screen_width = 1920

# Open the camera
cap = cv2.VideoCapture(0)

# Set the resolution to the maximum supported by the webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 0.25 means "manual exposure, manual iris"
cap.set(cv2.CAP_PROP_AUTO_WB, 1)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Unable to open the camera")
    exit()


 # Create a named window
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)

new_width = screen_width // 2
new_height = frame.shape[0] * new_width // frame.shape[1]

frame = cv2.resize(frame, (new_width, new_height))

# Flag to check if the picture has been taken
curr_node = 0
ar_id = path[0]
tracker(ar_id, lat_lon)
# Read and display frames from the camera
while curr_node < len(path) - 1:
    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if not ret:
        print("Error reading frame from the camera")
        break

    # Resize the frame to half of the screen width
    new_width = screen_width // 2
    new_height = frame.shape[0] * new_width // frame.shape[1]
    frame = cv2.resize(frame, (new_width, new_height))

    details, corners = detect_ArUco_details(frame)
    try:

        if distance(details[bot_marker], details[path[curr_node+1]]) < distance(details[bot_marker], details[path[curr_node]]):
            curr_node += 1
            ar_id = path[curr_node]
            tracker(ar_id, lat_lon)

        if distance(details[bot_marker], details[path[curr_node-1]]) < distance(details[bot_marker], details[path[curr_node]]):
            curr_node -= 1
            ar_id = path[curr_node]
            tracker(ar_id, lat_lon)

    except KeyError:
        pass

    cv2.imshow("Live Feed", frame)

    # Move the window to the left
    cv2.moveWindow("Live Feed", 0, 0)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()


