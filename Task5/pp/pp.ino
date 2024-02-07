#define ena1 17
#define in1 18
#define in2 5

#define ena2 3
#define in3 19
#define in4 21


//max speed

const int speed = 178;

// LOW means black HIGH means color

// #define irPin 17
#define irsen1 13 // Replace with the actual digital pin
#define irsen2 14 // Replace with the actual digital pin
#define irsen3 26 // Replace with the actual digital pin
#define irsen4 33 // Replace with the actual digital pin
#define irsen5 35 // Replace with the actual digital pin

unsigned long startTime = 0;
unsigned long tame = 0;
unsigned long T = 0;

#include <WiFi.h>

// WiFi credentials
// const char *ssid = "Redmi 12 5G";   // Enter your wifi hotspot ssid
// const char *password = "vijit3457"; // Enter your wifi hotspot password
// const uint16_t port = 8002;
// const char *host = "192.168.228.186"; // Enter the ip address of your laptop after connecting it to wifi hotspot

const char *ssid = "WARLIQ 5672";   // Enter your wifi hotspot ssid
const char *password = "0h498Z8/"; // Enter your wifi hotspot password
const uint16_t port = 8002;
const char *host = "192.168.137.1";

// External peripherals
int buzzerPin = 15;
int redLed = 2;
int ir1, ir2, ir3, ir4, ir5;

char incomingPacket[80];
WiFiClient client;
String msg;
int op = -1;
int counter = 0;
int flag = 0;
int entered = 0;
bool first_detection = false;
int i = 0;
bool tookTurn();

//function declaration

void setup()
{
    Serial.begin(115200);
    // Motor pins as output
    pinMode(ena1, OUTPUT);
    pinMode(in1, OUTPUT);
    pinMode(in2, OUTPUT);

    pinMode(ena2, OUTPUT);
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
    analogWrite(ena1,0);


    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);
    analogWrite(ena2,0);

    // digitalWrite(buzzerPin,HIGH);
    digitalWrite(redLed,LOW);

    // Connecting to wifi
    // WiFi.disconnect(true);
    // WiFi.begin(ssid, password);

    // while (counter < 10)
    // {
    //     T = millis();
    //     while (WiFi.status() != WL_CONNECTED && millis() - T < 8000)
    //     {
    //         delay(500);
    //         Serial.print("...");
    //     }
    //     if (WiFi.status() == WL_CONNECTED)
    //     {
    //         flag = 1;
    //         break;
    //     }
    //     else
    //     {
    //         counter++;
    //         delay(1000);
    //     }
    // }
    // if (!flag)
    // {
    //     exit(1);
    // }
}


void loop()
{
  checkIR();
    // int connectionAttempts = 0;
    // while (!client.connect(host, port) && connectionAttempts < 10)
    // {
    //     Serial.print("Connection to host failed, retrying...\n");
    //     delay(1000);
    //     connectionAttempts++;
    //     Serial.print(connectionAttempts);
    // }

    // if (connectionAttempts < 10)
    // {
    //     // handleClientCommands();
    //     // WiFi.macAddress();
    //     checkIR();
    // }
    // else
    // {
    //     Serial.println("Exceeded connection attempts. Waiting before retrying...\n");
    //     delay(500); // Wait for 5 seconds before retrying
    // }
}

void checkIR(){
  // tame = millis();
  // while(millis() - tame < 1000){

  // }
  while(1){
    allSen();
    delay(1000);
  }

  

}

// void turnRightSen(){
//   goRight();
//   while(!ir5) allSen();
//   while(!ir3)
// }



// void stopAlign(){
//       if(ir3){
//       moveForward();
//     }
//     else {
//       adjustRight();
//       tame = millis();
//       while(millis() - tame < 1000){
//         if(allSenB()){
//           return exitTrack();
//         }
//       }
//       if(!allSenB()){
//         stopMotors();
//         tame = millis();
//         while(millis() - tame < 5000){
//           digitalWrite(buzzerPin,LOW);
//         }
//         digitalWrite(buzzerPin,HIGH);
//         exit(0);
//       }
//     }
// }
bool allSenB(){
  allSen();
  if(ir1 || ir2 || ir3 || ir4 || ir5) return true;
  else return false;
}


void allSen(){
  ir1 = digitalRead(irsen1); // Front Left
  ir2 = digitalRead(irsen2); // Center Left
  ir3 = digitalRead(irsen3); // Front Center
  ir4 = digitalRead(irsen4); // Center Right
  ir5 = digitalRead(irsen5); // Front Right
  Serial.println("sensing irs");
  Serial.println(String(ir1) + "," + String(ir2) + "," + String(ir3)+ "," + String(ir4)+ "," + String(ir5));
  Serial.println("sensed irs");
}

