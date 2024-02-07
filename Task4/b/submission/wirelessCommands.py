import socket
import time

# Set the IP address and port of the ESP32 server
# esp32_ip = "192.168.228.79"  # Change this to the IP address of your ESP32
esp32_ip = ""
esp32_port = 8002

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((esp32_ip, 8002))
    s.listen()
    conn, addr = s.accept()
    print("hello")
    with conn:
        print(f"Connected by {addr}")
        data = conn.recv(1024)
        print(data)
        command = input("Enter command (1: Move Forward, 2: Move Backward, 5: Stop): ")
        conn.sendall(str.encode(command))


        while True:

            data = conn.recv(1024)
            data = data.decode('utf-8')
            data = data.replace('\r', ' ')
            data = data.replace('\n', ' ')
            print(data)

            
            
            if command == "5":
                s.close()
                break
        

