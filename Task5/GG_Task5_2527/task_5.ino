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
#define redLed 2

unsigned long startTime = 0;
unsigned long tame = 0;
unsigned long T = 0;

#include <WiFi.h>
// const char *ssid = "WARLIQ 5672";   // Enter your wifi hotspot ssid
// const char *password = "0h498Z8/"; // Enter your wifi hotspot password
// const uint16_t port = 8002;
// const char *host = "192.168.137.1";

const char *ssid = "DEVASTATINGRPG";   // Enter your wifi hotspot ssid
const char *password = "beansbestcat"; // Enter your wifi hotspot password
const uint16_t port = 8002;
const char *host = "192.168.137.1";

int ir1, ir2, ir3, ir4, ir5;
char incomingPacket[80];
WiFiClient client;
String msg;
int op;
int counter = 0;
int flag = 0;
int entered = 0;
bool tookRight = false;

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
  pinMode(redLed, OUTPUT);
  // Keeping all motors off initially
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

  digitalWrite(buzzerPin, HIGH);
  digitalWrite(redLed, LOW);

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

// void handleClientCommands()
// {
//     // Read the command from the client
//     // Process the command
//     while (client.connected())
//     {
//         msg = client.readStringUntil('\r');

//         if (!msg.isEmpty()){
//           op = msg.toInt();
//           Serial.println("changing op to : ");
//           Serial.println(op);
//         }
//         if (op == 1) {
//           allSen();
//           if (ir2 && ir4) {
//             while(ir2 && ir4){
//               track();
//             }
//           }
//           travel();
//         }
//         else if (op == 2 || op == 7) turnRight();
//         else if (op == 3 || op == 8) turnLeft();
//         else if (op == 4) turnU();
//         else if (op == 5) signal();
//         else if (op == 6) endLeft();
//         else if (op == 9) endStraight();
//         else if (op == 10) buzzStraight();
//         else if (op == 11) travel1();
//         else if (op == 12){
//           tame = millis();
//           while(millis() - tame < 5000) track();
//           travel1();
//           travel();
//         }
//         else stopMotors();
//     }
// }

void handleClientCommands()
{
  // Read the command from the client
  // Process the command
  while (client.connected())
  {
    if (client.available())
    {
      // Read data from the server
      String message = client.readStringUntil('\n');
      Serial.println("Received message: " + message);

      // Process the received message
      handleMessage(message);
    }
    else
      stopMotors();
  }
}

void handleMessage(String message)
{
  // Convert the message to an integer
  if (!message.isEmpty())
  {
    op = message.toInt();
    Serial.println("Received command: " + String(op));
  }

  if (op == 1)
  {
    allSen();
    if (ir2 && ir4)
    {
      while (ir2 && ir4)
      {
        track();
      }
    }
    travel();
  }
  else if (op == 2 || op == 7)
    turnRight();
  else if (op == 3 || op == 8)
    turnLeft();
  else if (op == 4)
    turnU();
  else if (op == 5)
    signal();
  else if (op == 6)
    endLeft();
  else if (op == 9)
    endStraight();
  else if (op == 10)
    buzzStraight();
  else if (op == 11)
    travel1();
  else if (op == 12)
  {
      tame = millis();
      while(millis() - tame < 7640){
        track();
      }
      tame = millis();
      while(millis() - tame <2500){
        track1();
      }
    travel();
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
  signalExtreme();
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
  while (millis() - tame < 1500)
  {
    track();
  }
  signalExtreme();
  exit(0);
}

void travel()
{
  while (1)
  {
    allSen();
    if (ir2 && ir4)
    {
      client.print("node");
      stopMotors();
      break;
    }
    else
      track();
  }
}

void travel1()
{
  Serial.println("in travel1");
  allSen();
  tame = millis();
  while(millis() - tame < 1500){
    track();
  }
  if (client.available()){
    return;
  }
  
  while (op == 11)
  {
    track1();
    if (client.available())
    {
      Serial.println("got new command");
      // // Read data from the server
      // String message = client.readStringUntil('\n');
      // Serial.println("Received message: " + message);

      // // Process the received message
      // handleMessage(message);
      stopMotors();
      // handleClientCommands();
      return;
    }
  }
}

void buzzStraight()
{
  while (op == 10)
  {
    track();
    if (client.available())
    {
      stopMotors();
      return;
    }
  }
}

void track()
{
  Serial.print("in track");
  allSen();
  if (ir2 && ir4)
  {
    moveForward(250);
  }
  else if (ir2)
  {
    adjustLeft(238, 50);
  }
  else if (ir4)
  {
    adjustRight(238, 50);
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
    adjustLeft(255, 80);
  }
  else if (ir3 && (ir5 || !ir1))
  {
    moveForward(255);
  }
  else if (ir3)
  {
    moveForward(200);
  }
  else
  {
    moveForward(200);
    // adjustRight(230,130);
  }
}

void track1()
{
  Serial.print("in track");
  allSen();
  if (ir2 && ir4)
  {

    stopMotors();
    return;
    // moveForward(250);
  }
  else if (ir2)
  {
    adjustLeft(210, 60);
  }
  else if (ir4)
  {
    adjustRight(210, 60);
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
    adjustLeft(255, 80);
  }
  else if (ir3 && (ir5 || !ir1))
  {
    moveForward(255);
  }
  else if (ir3)
  {
    moveForward(200);
  }
  else
  {
    // moveForward(200);
    adjustRight(230, 60);
  }
}

void turnU()
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
  // adjustRight(255,130);
  travel();
  client.print("done");
  Serial.print("turn u");
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
  while (millis() - tame < 500)
    continue;
  allSen();
  tame = millis();
  // while (millis() - tame < 1500)
  // {
  //   track();
  // }
  stopMotors();
  client.print("turned");
  while (op == 7)
  {
         Serial.println("got new command");
 
    // tame = millis();
    // while (millis() - tame < 1000)
    // {
    //   track();
    // }
    track();
    if (client.available())
    {
      stopMotors();
      return;
    }
  }
  if (op == 2)
  {
    travel();
  }
}

void turnLeft()
{
  tame = millis();
  while (millis() - tame < 300)
    track();
  goLeft(240, 240);
  tame = millis();
  while (millis() - tame < 570)
    continue;
  stopMotors();
  allSen();
  // if(!ir4){
  //   while(!ir4){
  //     allSen();
  //     goLeft();
  //   }
  // }
  tame = millis();
  while (millis() - tame < 1500)
  {
    track();
  }
  while (op == 8)
  {
    client.print("turned");
    tame = millis();
    // while (millis() - tame < 1000)
    // {
    //   track();
    // }
    // stopMotors();
    track();
    // msg = client.readStringUntil('\r');
    // if (!msg.isEmpty()){
    //   op = msg.toInt();
    //   Serial.println("changing op to : ");
    //   Serial.println(op);
    // }
    // if (op == 5){
    //   signal();
    //   delay(500);
    // }
    if (client.available())
    {
      stopMotors();
      return;
    }
  }
  if (op == 3)
  {
    travel();
  }
}

void signal()
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

void signalExtreme()
{
  stopMotors();
  T = millis();
  while (millis() - T < 5000)
  {
    stopMotors();
    digitalWrite(redLed, HIGH);
    digitalWrite(buzzerPin, LOW);
  }
  digitalWrite(redLed, LOW);
  digitalWrite(buzzerPin, HIGH);
}

void allSen()
{
  ir1 = digitalRead(irsen1); // Front Left
  ir2 = digitalRead(irsen2); // Center Left
  ir3 = digitalRead(irsen3); // Front Center
  ir4 = digitalRead(irsen4); // Center Right
  ir5 = digitalRead(irsen5); // Front Right
}

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

void moveBackward()
{
  // Move backward
  motorControl(ena1, 1, 150, LOW, HIGH);
  motorControl(ena2, 2, 150, LOW, HIGH);
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
