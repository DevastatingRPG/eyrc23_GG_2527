//motor1
#define in1 18
#define in2 5
#define ena1 17

//motor2
#define in3 19
#define in4 21
#define ena2 3

//irsen
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
const char *ssid = "WARLIQ 5672";   // Enter your wifi hotspot ssid
const char *password = "0h498Z8/"; // Enter your wifi hotspot password
const uint16_t port = 8002;
const char *host = "192.168.137.1";

int ir1, ir2, ir3, ir4, ir5;
char incomingPacket[80];
WiFiClient client;
String msg;
int op;
int counter = 0;
int i = -1;

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

    pinMode(buzzerPin,OUTPUT);
    pinMode(redLed,OUTPUT);
    // Keeping all motors off initially
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);

    digitalWrite(buzzerPin,HIGH);
    digitalWrite(redLed,LOW);

    // Connecting to wifi
    WiFi.disconnect(true);
    WiFi.begin(ssid, password);

    while (counter < 10)
    {
        T = millis();
        while (WiFi.status() != WL_CONNECTED && millis() - T < 8000)  delay(500);
        if (WiFi.status() == WL_CONNECTED) break;
        else  {
            counter++;
            delay(1000);
        }
    }
}

void loop(){
    int connectionAttempts = 0;
    while (!client.connect(host, port) && connectionAttempts < 10)  {
        delay(1000);
        connectionAttempts++;
    }
    if (connectionAttempts < 10)  handleClientCommands();
    else delay(500); // Wait for 0.5 seconds before retrying
}

void handleClientCommands()
{
    // Read the command from the client
    // Process the command
    while (client.connected())
    {
      client.setTimeout(10000);
        msg = client.readStringUntil('\r');
        if (!msg.isEmpty()){
          op = msg.toInt();
        }
        if (op == 1)  track();
        else if (op == 2) {
          turnRight();
          op = 1;
        }
        else if (op == 3) {
          turnLeft();
          op = 1;
        }
        else if (op == 4) turnU();
        else if (op == 5) signal();
        else stopMotors();
    }
}


void track(){
  allSen();
  if (ir2 && ir4) {
    stopMotors();
    client.print("at node");
    delay(1000);
  }
  else if(ir2) adjustLeft();
  else if (ir4) adjustRight();
  else if (ir3 && !ir5 && !ir1) moveForward();
  else if (ir1 && !ir5) adjustRight();
  else if (ir5 && !ir1) adjustLeft();
  else moveForward();
}


void turnRight(){
  moveForward();
  tame = millis();
  while(millis() - tame < 300) continue;
  goRight();
  tame = millis();
  while(millis() - tame < 800) continue;

}

void turnLeft(){
  moveForward();
  tame = millis();
  while(millis() - tame < 300) continue;
  goLeft();
  tame = millis();
  while(millis() - tame < 700) continue;
}

void turnU(){
  turnRight();
  tame=millis();
  while(millis() - tame < 1400) continue;
  stopMotors();
}

void signal()
{
    T = millis();
    while (millis() - T < 1000)
    {
        digitalWrite(buzzerPin, LOW);
    }
    digitalWrite(buzzerPin, HIGH);
}

void signalExtreme(){
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

void allSen(){
  ir1 = digitalRead(irsen1); // Front Left
  ir2 = digitalRead(irsen2); // Center Left
  ir3 = digitalRead(irsen3); // Front Center
  ir4 = digitalRead(irsen4); // Center Right
  ir5 = digitalRead(irsen5); // Front Right
}

void motorControl(int enablePin, int motor, int speed, int dir1, int dir2){
  analogWrite(enablePin, speed);
  
  if (motor == 1){
    digitalWrite(in1, dir1);
    digitalWrite(in2, dir2);
  }
  else if (motor == 2){
    digitalWrite(in3, dir1);
    digitalWrite(in4, dir2);
  }
}
void moveForward(){
  // Move forward
  motorControl(ena1, 1, 220, HIGH, LOW);
  motorControl(ena2, 2, 220, HIGH, LOW);
}

void moveBackward(){
  // Move backward
  motorControl(ena1, 1, 150, LOW, HIGH);
  motorControl(ena2, 2, 150, LOW, HIGH);
}

void goRight(){
  // Turn left
  motorControl(ena1, 2, 200, LOW, HIGH);
  motorControl(ena2, 1, 200, HIGH, LOW);
}

void goLeft(){
  // Turn right
  motorControl(ena1, 2, 200, HIGH, LOW);
  motorControl(ena2, 1, 200, LOW, HIGH);
}

void adjustLeft(){
  motorControl(ena1, 1, 100, HIGH, LOW);
  motorControl(ena2, 2, 200, HIGH, LOW);
}

void adjustRight(){
  motorControl(ena1, 1, 200, HIGH, LOW);
  motorControl(ena2, 2, 100, HIGH, LOW);
}

void stopMotors(){
  // Stop
  motorControl(ena1, 1, 0, LOW, LOW);
  motorControl(ena2, 2, 0, LOW, LOW);
}