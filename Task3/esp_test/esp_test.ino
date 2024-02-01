#include <WiFi.h>

// WiFi credentials
const char* ssid = "Airtel_prit_0694";                    //Enter your wifi hotspot ssid
const char* password =  "air73468";               //Enter your wifi hotspot password
const uint16_t port = 8002;
const char * host = "192.168.1.10";                   //Enter the ip address of your laptop after connecting it to wifi hotspot

// External peripherals 
int buzzerPin = 19;
int redLed = 21;
#define in1 21
#define in2 19
#define in3 2
#define in4 15


char incomingPacket[80];
WiFiClient client;

String msg = "0";
int counter = 0;



void setup(){
   
  // Serial.begin(115200);                          //Serial to print data on Serial Monitor

  // Output Pins
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  // //Connecting to wifi
  // WiFi.begin(ssid, password);

  // while (WiFi.status() != WL_CONNECTED) {
  //   delay(500);
  //   Serial.println("...");
  // }
 
  // Serial.print("WiFi connected with IP: ");
  // Serial.println(WiFi.localIP());
}


void loop() {

  // if (!client.connect(host, port)) {
  //   Serial.println("Connection to host failed");
  //   digitalWrite(buzzerPin, HIGH);           
  //   digitalWrite(redLed, LOW); 
  //   delay(200);
  //   return;
  // }

  while(1){
                    //Print data on Serial monitor
        digitalWrite(in1, HIGH);
        digitalWrite(in2, LOW);
        digitalWrite(in3, HIGH);
        digitalWrite(in4, LOW);
    
     
    }
}
