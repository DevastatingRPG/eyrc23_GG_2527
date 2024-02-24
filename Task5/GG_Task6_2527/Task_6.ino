// motor1
#define in1 18
#define in2 5
#define ena1 17

// motor2
#define in3 19
#define in4 21
#define ena2 3

// irsen
#define irsen1 13 // Replace with the actual digital pin
#define irsen2 14 // Replace with the actual digital pin
#define irsen3 26 // Replace with the actual digital pin
#define irsen4 33 // Replace with the actual digital pin
#define irsen5 35 // Replace with the actual digital pin

#define buzzerPin 15
#define led 2

unsigned long startTime = 0;
unsigned long tame = 0;
unsigned long T = 0;

#include <WiFi.h>
const char *ssid = "DEVASTATINGRPG";   // Enter your wifi hotspot ssid
const char *password = "beansbestcat"; // Enter your wifi hotspot password
const uint16_t port = 8002;
const char *host = "192.168.137.1";

//Function prototypes

// Function to initialize the setup
void setup();

// Main function to run the robot operations
void loop();

// Function to handle connecting to the host and client commands
void handleClientCommands();

// Function to handle the received client commands
void handleMessage(int op);

// Function to execute straight movement until exiting the arena
void endStraight();

// Function to execute left turn movement until exiting the arena
void endLeft();

// Function to follow the line until reaching a node
void followLine();

// Function to follow a specific line for a particular event (E)
void followLine1();

// Function to check if the robot is on a node
bool isNode();

// Function to control the movement of the robot based on sensor readings
void track();

// Function to control the movement of the robot in a specific event lane (E)
void track1();

// Function to align the robot to the path if deviated to the right side
void alignRight(int speed1, int speed2);

// Function to align the robot to the path if deviated to the left side
void alignLeft(int speed1, int speed2);

// Function to perform a U-turn
void Uturn();

// Function to return from a special event lane (E)
void backFromE();

// Function to turn the robot right
void turnRight();

// Function to turn the robot left
void turnLeft();

// Function to signal an event visit
void eventSignal();

// Function to run certain actions at the end of the arena
void endSignal();

// Function to read sensor data from all sensors
void allSen();

// Function to control the motors for movement
void motorControl(int enablePin, int motor, int speed, int dir1, int dir2);

// Function to move the robot forward
void moveForward(int speed);

// Function to make the robot turn right
void goRight(int lSpeed, int rSpeed);

// Function to make the robot turn left
void goLeft(int lSpeed, int rSpeed);

// Function to adjust the robot's movement left
void adjustLeft(int Hspeed, int Lspeed);

// Function to adjust the robot's movement right
void adjustRight(int Hspeed, int Lspeed);

// Function to stop the motors
void stopMotors();


// Global Variables
int ir1, ir2, ir3, ir4, ir5;
char incomingPacket[80];
WiFiClient client;
String msg;
int op;
int counter = 0;
int flag = 0;

void setup()
{
  Serial.begin(115200);
  // Motor pins as output
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  // sensor pin as input
  pinMode(irsen1, INPUT);
  pinMode(irsen2, INPUT);
  pinMode(irsen3, INPUT);
  pinMode(irsen4, INPUT);
  pinMode(irsen5, INPUT);

  pinMode(buzzerPin, OUTPUT);
  pinMode(led, OUTPUT);
  // Keeping all motors off initially
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

  digitalWrite(buzzerPin, HIGH);
  digitalWrite(led, LOW);

  // Connecting to wifi
  WiFi.disconnect(true);
  WiFi.begin(ssid, password);

  while (counter < 10)
  {
    T = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - T < 8000)
    {
      delay(500);
      Serial.print("...");
    }
    if (WiFi.status() == WL_CONNECTED)
    {
      Serial.print("WiFi connected with IP: 2382347\n");
      flag = 1;
      break;
    }
    else
    {
      Serial.print("Connection Failed\n");
      counter++;
      delay(1000);
    }
  }
  if (!flag)
  {
    exit(1);
  }
}
void loop()
{
  int connectionAttempts = 0;
  while (!client.connect(host, port) && connectionAttempts < 10)
  {
    Serial.print("Connection to host failed, retrying...\n");
    delay(1000);
    connectionAttempts++;
    Serial.print(connectionAttempts);
  }

  if (connectionAttempts < 10)
  {
    handleClientCommands();
  }
  else
  {
    Serial.println("Exceeded connection attempts. Waiting before retrying...\n");
    delay(500); // Wait for 5 seconds before retrying
  }
}


void handleClientCommands()
{
  // Read the command from the client
  // Process the command
  while (client.connected())
  {
    Serial.print("connected");
    if (client.available())
    {
      // Read data from the server
      String message = client.readStringUntil('\n');
      Serial.println("Received message: " + message);
      // Process the received message
        if (!message.isEmpty())
  {
    op = NULL;
    // Convert the message to an integer
    op = message.toInt();
    Serial.println("Received command: " + String(op));
  }
      handleMessage(op);
    }
    else
      stopMotors();
  }
}

void handleMessage(int op)
{
  op = int(op);

  if (op == 1)
  {
    allSen();
    if (isNode())
    {
      while (isNode())
      {
        allSen();
        Serial.println("getting out");
        track();
      }
    }
    followLine();
  }
  else if (op == 2)
    turnRight();
  else if (op == 3)
    turnLeft();
  else if (op == 4)
    Uturn();
  else if (op == 5)
    eventSignal();
  else if (op == 6)
    endLeft();
  else if (op == 8) 
    backFromE();
  else if (op == 9)
    endStraight();
  
  else if (op == 11 || op == 7)
  {
    allSen();
    if (isNode())
    {
      while (isNode())
      {
        allSen();
        track();
      }
    }
    followLine1();
  }
  else  
    stopMotors();
}


void endStraight()         
{
  tame = millis();
  while (millis() - tame < 2000)
  {
    track();
  }
  endSignal();
  exit(0);
}

void endLeft()      
{
  tame = millis();
  while (millis() - tame < 300)
    track();
  goLeft(210, 210);
  tame = millis();
  while (millis() - tame < 600)
    continue;
  tame = millis();
  while (millis() - tame < 1000)
  {
    track();
  }
  endSignal();
  exit(0);
}

void followLine()       
{
  while (1)
  {
    allSen();
    if (isNode()){
      stopMotors();
      client.print("node");
      return;
    }
    else if (client.available()){
      stopMotors();
      return;
    }
    else
      track();
  }
}

void followLine1()
{
  while (1)
  {
    allSen();
    track1();
    if (client.available())
    {
      stopMotors();
      return;
    }
  }
}

bool isNode(){        
  allSen();
  if (ir2 && ir4){
    return true;
  }
  return false;
}


void track()            
{
  Serial.print("in track");
  allSen();
  if (isNode())
  {
    moveForward(250);
  }
  else if (ir2)
  {
    adjustLeft(255, 50); 
  }
  else if (ir4)
  {
    adjustRight(255, 50);
  }
  else if (ir1 && ir5)
  {
    moveForward(255);
  }
  else if (ir1 && !ir5)
  {
    adjustRight(255, 80);
  }
  else if (ir5 && !ir1)
  {
    adjustLeft(255, 40);
  }
  else if (ir3 && (ir5 || !ir1))
  {
    moveForward(255);
  }
  else if (ir3)
  {
    moveForward(255);
  }
  else
  {
    moveForward(255);
  }
}

void track1()
{
  Serial.print("in track");
  allSen();
  if (isNode())
  {
    stopMotors();
    client.print("node");
    while(!client.available()) continue;
    return;
  }
  else if (ir2)
  {
    adjustLeft(210, 60);
  }
  else if (ir4)
  {
    adjustRight(210, 60);
  }
  else if (ir3)
  {
    moveForward(255);
  }
  else if (ir1 && ir5)
  {
    moveForward(255);
  }
  else if (ir1 && !ir5)
  {
    if(op == 7)
      alignRight(255,0);
    else 
      alignRight(255,0);
  }
  else if (ir5 && !ir1)
  {
    alignLeft(255,0);  
  }
  else if (ir3 && (ir5 || !ir1))
  {
    moveForward(255);
  }
  else
  {
    moveForward(255);
  }
}

void alignRight(int speed1, int speed2)     
{
  adjustRight(speed1,speed2);
  while(!ir5){
    allSen();
    if(ir2 || ir4) return;
  } 
  adjustLeft(speed1,speed2);
  while(ir5) allSen();
}

void alignLeft(int speed1, int speed2)      
{
  adjustLeft(speed1 , speed2);
  while(!ir1){
    allSen();
    Serial.print("hello");
    if(ir2 || ir4) return;
  } 
  allSen();
  adjustRight(speed1, speed2);
  while(ir1) allSen();
}

void Uturn()
{
  goRight(255, 255);
  tame = millis();
  while (millis() - tame < 978)
  {
    continue;
  }
  stopMotors();
  allSen();
  if (!ir5)
  {
    adjustRight(255, 200);
    while (!ir5 && !ir2)
    {
      allSen();
    }
  }
  followLine();
}

void backFromE(){         
  goRight(255, 255);
  tame = millis();
  while (millis() - tame < 978)
  {
    continue;
  }
  stopMotors();
  allSen();
  if (!ir5)
  {
    adjustRight(255, 200);
    while (!ir5 && !ir2)
    {
      allSen();
    }
  }
  followLine1();
  
}

void turnRight()
{
  tame = millis();
  while (millis() - tame < 210)
  {
    track();
  }
  goRight(240, 240);
  tame = millis();
  // while (millis() - tame < 500)
  while (millis() - tame < 570)
    continue;
  stopMotors();
  client.print("turned");
  followLine();
  // }
  
}

void turnLeft()
{
  tame = millis();
  while (millis() - tame < 300)
    track();
  goLeft(240, 240);
  tame = millis();
  while (millis() - tame < 612)
    continue;
  stopMotors();
  allSen();
  followLine();
}

void eventSignal()         
{
  stopMotors();
  T = millis();
  while (millis() - T < 1000)
  {
    digitalWrite(buzzerPin, LOW);
  }
  digitalWrite(buzzerPin, HIGH);
  client.print("buzz");
}

void endSignal()
{
  stopMotors();
  client.print("node");
  T = millis();
  while (millis() - T < 5000)
  {
    stopMotors();
    digitalWrite(led, HIGH);
    digitalWrite(buzzerPin, LOW);
  }
  digitalWrite(led, LOW);
  digitalWrite(buzzerPin, HIGH);
  client.print("positive");
}

void allSen()     //All sensor reading
{
  ir1 = digitalRead(irsen1); // Front Left
  ir2 = digitalRead(irsen2); // Center Left
  ir3 = digitalRead(irsen3); // Front Center
  ir4 = digitalRead(irsen4); // Center Right
  ir5 = digitalRead(irsen5); // Front Right
}

//Motor Functions

void motorControl(int enablePin, int motor, int speed, int dir1, int dir2)
{
  analogWrite(enablePin, speed);

  if (motor == 1)
  {
    digitalWrite(in1, dir1);
    digitalWrite(in2, dir2);
  }
  else if (motor == 2)
  {
    digitalWrite(in3, dir1);
    digitalWrite(in4, dir2);
  }
}
void moveForward(int speed)
{
  // Move forward
  motorControl(ena1, 1, speed, HIGH, LOW);
  motorControl(ena2, 2, speed, HIGH, LOW);
}

void goRight(int lSpeed, int rSpeed)
{
  // Turn left
  motorControl(ena1, 2, lSpeed, LOW, HIGH);
  motorControl(ena2, 1, rSpeed, HIGH, LOW);
}

void goLeft(int lSpeed, int rSpeed)
{
  // Turn right
  motorControl(ena1, 2, lSpeed, HIGH, LOW);
  motorControl(ena2, 1, rSpeed, LOW, HIGH);
}

void adjustLeft(int Hspeed, int Lspeed)
{
  motorControl(ena1, 1, Lspeed, HIGH, LOW);
  motorControl(ena2, 2, Hspeed, HIGH, LOW);
}

void adjustRight(int Hspeed, int Lspeed)
{
  motorControl(ena1, 1, Hspeed, HIGH, LOW);
  motorControl(ena2, 2, Lspeed, HIGH, LOW);
}

void stopMotors()
{
  // Stop
  motorControl(ena1, 1, 0, LOW, LOW);
  motorControl(ena2, 2, 0, LOW, LOW);
}

