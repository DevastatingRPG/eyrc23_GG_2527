#include <WiFi.h>

// WiFi credentials
const char* ssid = "Airtel_prit_0694";                    //Enter your wifi hotspot ssid
const char* password =  "air73468";               //Enter your wifi hotspot password
const uint16_t port = 8002;
const char * host = "192.168.1.10";                   //Enter the ip address of your laptop after connecting it to wifi hotspot

// External peripherals 
int buzzerPin = 19;
int redLed = 21;
#define in1 16
#define in2 4
#define in3 2
#define in4 15


char incomingPacket[80];
WiFiClient client;

String msg = "0";
int counter = 0;



void setup(){
   
  Serial.begin(115200);                          //Serial to print data on Serial Monitor

  // Output Pins
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  //Connecting to wifi
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("...");
  }
 
  Serial.print("WiFi connected with IP: ");
  Serial.println(WiFi.localIP());
}


void loop() {

  if (!client.connect(host, port)) {
    Serial.println("Connection to host failed");
    digitalWrite(buzzerPin, HIGH);           
    digitalWrite(redLed, LOW); 
    delay(200);
    return;
  }

  while(1){
      msg = client.readStringUntil('\n');         //Read the message through the socket until new line char(\n)
      client.print("Hello from ESP32!");          //Send an acknowledgement to host(laptop)
      counter = msg.toInt();
      Serial.println(counter);                    //Print data on Serial monitor
      if(counter%2==0){
        digitalWrite(in1, HIGH);
        digitalWrite(in2, LOW);
        digitalWrite(in3, HIGH);
        digitalWrite(in4, LOW);
      }
      else{
        digitalWrite(in1, LOW);
        digitalWrite(in2, LOW);
        digitalWrite(in3, LOW);
        digitalWrite(in4, LOW);      
      }
     
    }
}
