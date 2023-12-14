//Motor pin connections
#define in1 16
#define in2 4
#define in4 2
#define in3 15
//ir pin connection
// #define irPin 17
unsigned long startTime = 0;
const unsigned long duration = 5000;  // 5 seconds in milliseconds

#include <WiFi.h>

// WiFi credentials
const char* ssid = "Airtel_prit_0694";       //Enter your wifi hotspot ssid
const char* password = "air73468";  //Enter your wifi hotspot password
const uint16_t port = 8002;
const char* host = "192.168.1.10";  //Enter the ip address of your laptop after connecting it to wifi hotspot

// External peripherals
// int buzzerPin = 15;
// int redLed = 2;


char incomingPacket[80];
WiFiClient client;
String msg;
int op;

void handleClientCommands();
void moveForward();
void moveBackward();
void turnLeft();
void turnRight();

void setup() {
  Serial.begin(115200);
  //Motor pins as output
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  //sensor pin as input
  // pinMode(irPin, INPUT);
  //Keeping all motors off initially
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

  //Connecting to wifi
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("...");
  }
  Serial.print("WiFi connected with IP: ");
  // Serial.println(WiFi.localIP());
}

void loop() {
  // Serial.println("Hello");

  // if (!client.connect(host, port)) {
  //   Serial.println("Connection to host failed");
  //   delay(200);
  //   return;
  // }
  // else{
  //   handleClientCommands();
  // }
  int connectionAttempts = 0;
  while (!client.connect(host, port) && connectionAttempts < 10) {
    Serial.println("Connection to host failed, retrying...");
    delay(200);
    connectionAttempts++;
  }

  if (connectionAttempts < 10) {
    handleClientCommands();
  } else {
    Serial.println("Exceeded connection attempts. Waiting before retrying...");
    delay(5000);  // Wait for 5 seconds before retrying
  }

  // if(!digitalRead(irPin)){
  //   //Stop
  //   digitalWrite(in1, LOW);
  //   digitalWrite(in2, LOW);
  //   digitalWrite(in3, LOW);
  //   digitalWrite(in4, LOW);
  // }
}

void handleClientCommands() {
    // Read the command from the client


    // Process the command
    while(client.connected()){
      msg = client.readStringUntil('\r');
      Serial.println("Received command: " + msg);
      op = msg.toInt();
      startTime = millis();
      while (millis() - startTime < duration) {
        if (op == 1) {
          moveForward();
        } else if (op == 2) {
          moveBackward();
        } else if (op == 5) {
          stopMotors();
        } else {
          stopMotors();
          // Unknown command, do nothing or handle accordingly
        }
      }

      // Send a response back to the client
      client.println("Command processed: " + msg);
    }

  
}

void moveForward() {
  // Move forward
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}

void moveBackward() {
  // Move backward
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}

void turnLeft() {
  // Turn left
  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);
  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);
}

void turnRight() {
  // Turn right
  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH);
}

void stopMotors() {
  // Stop
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}