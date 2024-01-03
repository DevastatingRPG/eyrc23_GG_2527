import cv2
import time
import ctypes

# Get screen size
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Unable to open the camera")
    exit()

# Flag to check if the picture has been taken
picture_taken = False

# Read and display frames from the camera
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame from the camera")
        break

    # Resize the frame to half of the screen width
    new_width = screen_width // 2
    new_height = frame.shape[0] * new_width // frame.shape[1]
    frame = cv2.resize(frame, (new_width, new_height))

    cv2.imshow("Live Feed", frame)

    # Move the window to the left
    cv2.moveWindow("Live Feed", 0, 0)

    # Take a picture after 3 seconds
    if not picture_taken:
        time.sleep(3)
        cv2.imwrite('picture.jpg', frame)
        picture_taken = True

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()