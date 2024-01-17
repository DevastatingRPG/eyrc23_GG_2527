# import socket
# from time import sleep
# import signal		
# import sys		

# def signal_handler(sig, frame):
#     print('Clean-up !')
#     cleanup()
#     sys.exit(0)

# def cleanup():
#     s.close()
#     print("cleanup done")

# ip = "192.168.1.10"     #Enter IP address of laptop after connecting it to WIFI hotspot


# #We will be sending a simple counter which counts from 1 to 10 and then closes the socket

# #To undeerstand the working of the code, visit https://docs.python.org/3/library/socket.html
# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     print("helo")
#     s.bind((ip, 8002))
#     s.listen()
#     print("helo")
#     try:
#         conn, addr = s.accept()
#     except Exception as e:
#         print(e)
    
#     with conn:
#         print(f"Connected by {addr}")
#         while True:
#             print("\nOperation chosen will perform while ir does not detect anything")
#             op = int(input("\n1: Forward\n2: Backward\n3: Left\n4: Right\n5: Exit"))
#             if op == 5:
#                 break;
#             conn.sendall(str.encode(str(op)))
#             # conn.sendall(op)
#             sleep(1)
import socket
import time

# Set the IP address and port of the ESP32 server
esp32_ip = ""  # Change this to the IP address of your ESP32
esp32_port = 8002

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((esp32_ip, 8002))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            print(data)
            command = input("Enter command (1: Move Forward, 2: Move Backward, 5: Stop): ")
            conn.sendall(str.encode(str(command)))
            if command == "5":
                s.close()
                break
