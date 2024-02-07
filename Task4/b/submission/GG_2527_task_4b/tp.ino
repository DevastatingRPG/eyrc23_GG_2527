// if this doesn't work try adjusting percent se
// Motor pin connections
#define in1 16
#define in2 4
#define in4 2
#define in3 15

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
// const char *host = "192.168.228.79";
const char *host = "192.168.93.79";

const char *ssid = "Sanskruti's Galaxy S20 FE 5G";   // Enter your wifi hotspot ssid
const char *password = "Epiphany18"; // Enter your wifi hotspot password
const uint16_t port = 8002;
// const char *host = "192.168.1.13";

// External peripherals
int buzzerPin = 19;
// int buzzerPin = 400;
int redLed = 21;
int ir1, ir2, ir3, ir4, ir5;

char incomingPacket[80];
WiFiClient client;
String msg;
int op;
int counter = 0;
int flag = 0;
int entered = 0;

void handleClientCommands();
void menu();
void moveForward();
void moveBackward();
void turnLeft();
void turnRight();
void turnRightSlightly();
void turnLeftSlightly();
void isNode();

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

    // Serial.print(WiFi.localIP());
}

bool first_detection = false;
// int i = -2;
int i = 0;

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
    Serial.print("detected: " + ir1);
    while (client.connected())
    {
        client.println("Supp buddy me robit");
        msg = client.readStringUntil('\r');
        Serial.println("Received command: " + msg);
        op = msg.toInt();

        startTime = millis();
        while (millis() - startTime < 600000)
        {
            if (op == 1)
            {
              entered = 1;
                // Read sensor values
                ir1 = digitalRead(irsen1); // Front Left
                ir2 = digitalRead(irsen2); // Center Left
                ir3 = digitalRead(irsen3); // Front Center
                ir4 = digitalRead(irsen4); // Center Right
                ir5 = digitalRead(irsen5); // Front Right
                client.println("sensing irs");
                client.println(ir1);
                client.println(ir2);
                client.println(ir3);
                client.println(ir4);
                client.println(ir5);
                client.println("sensed irs");
                if(ir2) client.print("Z");
                if(ir4) client.print("Q");
                // Move forward only if all sensors are low
                client.println(first_detection);
                menu();
            }
            else
            {
                break;
                // Unknown command, do nothing or handle accordingly
            }
        }
        // stopMotors();
        if(entered){
          menu();
        }
         client.println("Number of nodes detected are: ");
         client.println(i);
        client.println("Command processed: " + op);

        // move1();
        // Send a response back to the client
    }
}
// void handleClientCommands()
// {
//     // Read the command from the client
//     // Process the command
//     Serial.print("detected: " + ir1);
//     while (client.connected())
//     {
//         client.println("Supp buddy me robit");
//         msg = client.readStringUntil('\r');
//         Serial.println("Received command: " + msg);
//         op = msg.toInt();
//             if (op == 1)
//             {
//                 // Read sensor values
//                 ir1 = digitalRead(irsen1); // Front Left
//                 ir2 = digitalRead(irsen2); // Center Left
//                 ir3 = digitalRead(irsen3); // Front Center
//                 ir4 = digitalRead(irsen4); // Center Right
//                 ir5 = digitalRead(irsen5); // Front Right
//                 client.println("sensing irs");
//                 client.println(ir1);
//                 client.println(ir2);
//                 client.println(ir3);
//                 client.println(ir4);
//                 client.println(ir5);
//                 client.println("sensed irs");
//                 // Move forward only if all sensors are low
//                 client.println(first_detection);
//                 menu();
//                 // adjustLeft();
//             }
//             else if(op == 2){
//               stopMotors();
//             }
//             else continue;
//         }
//         stopMotors();
//         client.println("Sorry got disconnected for a second");
//         client.println("Command processed: " + op);

//         // move1();
//         // Send a response back to the client
// }

void menu()
{
    if(i == -2){
      signal1();
      i++;
    }
    if(i == -1){
      startAlign();
      // alignStart();
    }
    else if(i == 11){
      stopAlign();
    }
    // moveForward();
    else if ((ir1 == 0 && ir2 == 0 && ir3 == 0 && ir4 == 0 && ir5 == 0) && first_detection && i==12)    //END
    {
        client.println("buzzzzzz");
        signalExtreme();
        client.println("sayonara");
        exit(0);
    }
    else
    {
        if (ir2 == 1 && ir4 == 1)     //node
        {
            i++;
            client.println("node");
            client.print(i);
            client.println("buzz");
            tame = millis();
            stopMotors();
            while(millis() - tame < 2000){
              continue;
            }
            stopMotors();
            signal();

            if (i == 3 || i == 5 || i == 6 || i == 8)
            {
                Serial.print("bout right\n");
                client.println("right");
                turnTimeRight();

            }
            else if(i == 20){
              goLeft();
              tame = millis();
              while(millis() - tame < 300){
                continue;
              }
              goRight();
              while(!ir3){
                ir3 = digitalRead(irsen3);
              }
              moveForward();
              while(ir2 && ir4){
                ir2 = digitalRead(irsen2);
                ir4 = digitalRead(irsen4);
                ir3 = digitalRead(irsen3);
                if (~ir3){
                  centerAlignLeft();
                }
              }
            }
            // else if(i == 1){
            //   goLeft();
            //   while(!ir3){
            //     ir3 = digitalRead(irsen3);
            //   }
            //   moveForward();
            //   while(ir2 && ir4){
            //     ir2 = digitalRead(irsen2);
            //     ir4 = digitalRead(irsen4);
            //     ir3 = digitalRead(irsen3);
            //     if (~ir3){
            //       goLeft();
            //       while (~ir3){
            //         ir3 = digitalRead(irsen3);
            //       }
            //     }
        
            //   }
            //   moveForward();
            // }

          
            else if (i == 4 || i == 10)
            {
                Serial.print("bout left\n");
                turnTimeLeft();
                Serial.print("left");
                client.println("left");
            }
            else
            {
                Serial.print("forward");
                moveForward();
                  while(ir2 && ir4){
                    moveForward();
                    ir2 = digitalRead(irsen2);
                    ir4 = digitalRead(irsen4);
                    client.print("stage 3");
                  }
                client.println("straight");
            }
    }
    
        else if (ir2 == 1 && ir3 == 1 && ir4 == 0)   //center
        {
          client.print("center adjusting left");
          centerAlignLeft();
        }
        else if(ir3 == 1 && ir2 == 0 && ir4 == 0){
          moveForward();
        }
        else if(ir3 == 1 && ir4 == 1 && ir2 == 0){   //center
          client.println("Right center adjust");
          centerAlignRight();
        }
        else if( ir2 == 1 && ir3 == 0 && ir4 == 0){
          client.println("adjusting left");
          centerAlignLeft();
        }
        else if ( ir4 == 1 && ir3 == 0 && ir2 == 0){
          client.println("adjusting right");
          centerAlignRight();
        }

        else if (ir5 == 1 && ir1 == 0){   
          client.print("corner adjusting left");
          cornerAlignLeft();
        } 
        else if (ir5 == 0 && ir1 == 1 )
        {
          client.print("corner adjusting right");
          cornerAlignRight();
        }

        else if (ir1==1 && ir4==1 ){
          client.println("adjust right for ir1 and ir4");
          cornerAlignRight();
        }
        else if(ir1 == 0 && ir2 == 0 && ir3 == 0 && ir4 == 0 && ir5 == 0 ){
          client.println("none detected");
          adjustRight();
        }
        else if ((ir1 == 1 && ir3 == 1 && ir5 == 1 ) || (ir1 == 1 && ir5 == 1 ) || (ir3 == 1))
        {
          client.print("just moving forward like it should");
            moveForward();
        }
        else{
          client.println("ERRORRR!!");
        }

    }
}


void startAlign(){
  moveForward();
  while(ir3){
    ir3 = digitalRead(irsen3);
    ir2 = digitalRead(irsen2);
    ir4 = digitalRead(irsen4);
    if(ir2 && ir4){
      i++;
      return;
    }
  }

  adjustLeft();
  while(!ir3){
    ir3 = digitalRead(irsen3);
    ir2 = digitalRead(irsen2);
    ir4 = digitalRead(irsen4);
    if(ir2 && ir4){
      i++;
      return;
    }
  }
  stopMotors();
}

void alignStart(){
  moveForward();
  while(ir3){
    ir3 = digitalRead(irsen3);
  }
  client.print("going there");
  moveForward();
  while(!ir1){
    client.print("Me here");
    ir1 = digitalRead(irsen1);
  }
  goLeft();
  while(!ir3){
    client.print("supp buddy");
    ir3 = digitalRead(irsen3);
  }
  moveForward();
  while(ir3){
    client.print("supp buddy");
    ir3 = digitalRead(irsen3);
  }
  // goLeft();
  // tame = millis();
  // while(millis() - tame < 400){
  //   client.print("bruh");
  //   ir3 = digitalRead(irsen3);
  // }
  // goRight();
  // while(!ir3){
  //   ir3 = digitalRead(irsen3);
  // }
    ir2 = digitalRead(irsen2);
    ir4 = digitalRead(irsen4);
    if(ir4 && ir2){
      i++;
      return;
    }
  i++;
}

// void startAlign(){
//   moveForward();
//   while(!ir2){
//     ir2 = digitalRead(irsen2);
//   }
//   goLeft();
//   while(!ir3){
//     ir3 = digitalRead(irsen3);
//   }
//   moveForward();
//   while(!ir2 || !ir4){
//     ir2 = digitalRead(irsen2);
//     ir4 = digitalRead(irsen4);
//   }
//   goLeft();
//   while(!ir3){
//     ir3 = digitalRead(irsen3);
//   }
//   stopMotors();
//   i++;
// }

void stopAlign(){
  while(millis() - tame < 3500){
    moveForward();
    while(ir3){
        ir3 = digitalRead(irsen3);   
    }
    goRight();
      while(!ir3){
        ir3 = digitalRead(irsen3);
      }
      if(~ir1 & ~ir2 & ~ir3 & ~ir4 & ~ir5){
        i++;
      }
  }
}

void centerAlignLeft(){       //adjusts if robot goes off track (RIGHT)
  adjustLeft();
  while(ir2){
    ir2 = digitalRead(irsen2);
    ir4 = digitalRead(irsen4);
    ir3 = digitalRead(irsen3);
    if(ir2 && ir4){
      stopMotors();
      client.println("At Node");
      break;
    }
  }
}

void centerAlignRight(){      //adjusts if robot goes off track (LEFT)
  adjustRight();
  while(ir4 && !ir5){
    ir2 = digitalRead(irsen2);
    ir4 = digitalRead(irsen4);
    ir5 = digitalRead(irsen5);
    if(ir2 && ir4){
      stopMotors();
      client.println("At Node");
      break;
    }
  }
  if(ir4 && ir5){
    adjustLeft();
    while(ir4){
      ir4 = digitalRead(irsen4);
    }
  }

}

void cornerAlignLeft(){       //adjusts if robot goes off track (RIGHT)
  adjustLeft();
  while(!ir1 && !ir3){
    ir1 = digitalRead(irsen1);
    ir4 = digitalRead(irsen4);
    ir2 = digitalRead(irsen2);
    ir3 = digitalRead(irsen3);
    if(ir2 && ir4){
      return;
    }
  }
}

void cornerAlignRight(){      //adjusts if robot goes off track (LEFT)
  adjustRight();
  while(ir1){
    ir1 = digitalRead(irsen1);
  }
}

void turnTimeLeft(){
  tame = millis();
  goLeft();
  while(millis() - tame < 1000){
    continue;
  }
  while(!ir3){
    goLeft();
    ir3 = digitalRead(irsen3);
  }
}

void turnTimeRight(){
  tame = millis();
  goRight();
  client.print(tame);
  while(millis() - tame < 800){
    continue;
  }
  while(!ir3){
    goRight();
    ir3 = digitalRead(irsen3);
  }

}
void turnRight(){
  adjustBackLeft();
  while(!ir5){
    ir5 = digitalRead(irsen5);
                        ir1 = digitalRead(irsen1); // Front Left
                ir2 = digitalRead(irsen2); // Center Left
                ir3 = digitalRead(irsen3); // Front Center
                ir4 = digitalRead(irsen4); // Center Right
                ir5 = digitalRead(irsen5); // Front Right
                client.println("sensing irs");
                client.println(ir1);
                client.println(ir2);
                client.println(ir3);
                client.println(ir4);
                client.println(ir5);
                client.println("sensed irs");
    client.print("stage 1");
  }
  goRight();
  while(!ir4){
    ir4 = digitalRead(irsen4);
                        ir1 = digitalRead(irsen1); // Front Left
                ir2 = digitalRead(irsen2); // Center Left
                ir3 = digitalRead(irsen3); // Front Center
                ir4 = digitalRead(irsen4); // Center Right
                ir5 = digitalRead(irsen5); // Front Right
                client.println("sensing irs");
                client.println(ir1);
                client.println(ir2);
                client.println(ir3);
                client.println(ir4);
                client.println(ir5);
                client.println("sensed irs");
    client.print("stage 2");
  }

   while(!ir3){
    ir3 = digitalRead(irsen3);
                        ir1 = digitalRead(irsen1); // Front Left
                ir2 = digitalRead(irsen2); // Center Left
                ir3 = digitalRead(irsen3); // Front Center
                ir4 = digitalRead(irsen4); // Center Right
                ir5 = digitalRead(irsen5); // Front Right
                client.println("sensing irs");
                client.println(ir1);
                client.println(ir2);
                client.println(ir3);
                client.println(ir4);
                client.println(ir5);
                client.println("sensed irs");
    client.print("stage 2");
  }
  client.print("ppppppppppp");
  moveForward();
  // while(ir2 & ir4){
  //   moveForward();
  //   ir2 = digitalRead(irsen2);
  //   ir4 = digitalRead(irsen4);
  //   client.print("stage 3");
  // }
}

// void turnRight(){
// adjustBackLeft();
//   while(ir2 && ir4){
//     ir5 = digitalRead(irsen5);
//                         ir1 = digitalRead(irsen1); // Front Left
//                 ir2 = digitalRead(irsen2); // Center Left
//                 ir3 = digitalRead(irsen3); // Front Center
//                 ir4 = digitalRead(irsen4); // Center Right
//                 ir5 = digitalRead(irsen5); // Front Right
//                 client.println("sensing irs");
//                 client.println(ir1);
//                 client.println(ir2);
//                 client.println(ir3);
//                 client.println(ir4);
//                 client.println(ir5);
//                 client.println("sensed irs");
//     client.print("stage 1");
//   }
//   goRight();
//   while(!ir3){
//     ir3 = digitalRead(irsen4);
//                         ir1 = digitalRead(irsen1); // Front Left
//                 ir2 = digitalRead(irsen2); // Center Left
//                 ir3 = digitalRead(irsen3); // Front Center
//                 ir4 = digitalRead(irsen4); // Center Right
//                 ir5 = digitalRead(irsen5); // Front Right
//                 client.println("sensing irs");
//                 client.println(ir1);
//                 client.println(ir2);
//                 client.println(ir3);
//                 client.println(ir4);
//                 client.println(ir5);
//                 client.println("sensed irs");
//     client.print("stage 2");
//   }
//   tame = millis();
// while(millis() - tame < 2000){
//   stopMotors();
// }
//    while(!ir3){
//     ir3 = digitalRead(irsen3);
//                         ir1 = digitalRead(irsen1); // Front Left
//                 ir2 = digitalRead(irsen2); // Center Left
//                 ir3 = digitalRead(irsen3); // Front Center
//                 ir4 = digitalRead(irsen4); // Center Right
//                 ir5 = digitalRead(irsen5); // Front Right
//                 client.println("sensing irs");
//                 client.println(ir1);
//                 client.println(ir2);
//                 client.println(ir3);
//                 client.println(ir4);
//                 client.println(ir5);
//                 client.println("sensed irs");
//     client.print("stage 2");
//   }
//   client.print("ppppppppppp");
//   moveForward();
//   while(!ir2 && !ir4){
//     ir2 = digitalRead(irsen2);
//   }
//   goRight();
//   while(!ir3){
//     ir3 = digitalRead(irsen3);
//   }
  // while(ir2 & ir4){
  //   moveForward();
  //   ir2 = digitalRead(irsen2);
  //   ir4 = digitalRead(irsen4);
  //   client.print("stage 3");
  // }
//   client.print("done turning right");
// }
void Right2(){
	moveBackward();
	while (ir2 & ir4){
		ir2 = digitalRead(irsen2);
		ir4 = digitalRead(irsen4);
	}
	stopMotors();
	goRight();
	while (ir3){
		ir3 = digitalRead(irsen3);
	}
	while (ir5){
		ir5 = digitalRead(irsen5);
	}
	while (~ir5){
		ir5 = digitalRead(irsen5);
	}
	while (ir5){
		ir5 = digitalRead(irsen5);
	}
	while (~ir5){
		ir5 = digitalRead(irsen5);
	}
	stopMotors();
	moveForward();
}


void turnLeft(){
  goLeft();
  while(!ir1){
    ir1 = digitalRead(irsen1);
  }
  while(!ir3){
    ir3 = digitalRead(irsen3);
  }
  moveForward();
  while(ir2){
    ir2 = digitalRead(irsen2);
  }
  stopMotors();
}

// void jiggle(){}

void signal()
{
    T = millis();
    while (millis() - T < 1000)
    {
      //  digitalWrite(redLed, HIGH);
        digitalWrite(buzzerPin, LOW);
    }
    digitalWrite(buzzerPin, HIGH);
    // digitalWrite(redLed, LOW);
}
void signal1()
{
    T = millis();
    while (millis() - T < 1000)
    {
       digitalWrite(redLed, HIGH);
        digitalWrite(buzzerPin, LOW);
    }
    digitalWrite(buzzerPin, HIGH);
    digitalWrite(redLed, LOW);
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
void ShubhamRight(){
	moveBackward();
	while (ir2 & ir4){
		ir2 = digitalRead(irsen2);
		ir4 = digitalRead(irsen4);
	}
	stopMotors();
	goRight();
	while (ir3){
		ir3 = digitalRead(irsen3);
	}
	while (~ir3){
		ir3 = digitalRead(irsen3);
	}
	stopMotors();
	moveForward();
}


void moveForward()
{
    // Move forward
    analogWrite(in1, 220);
    analogWrite(in2, 0);
    analogWrite(in3, 220);
    analogWrite(in4, 0);
}

void moveBackward()
{
    // Move backward
    analogWrite(in1, 0);
    analogWrite(in2, 150);
    analogWrite(in3, 0);
    analogWrite(in4, 150);
}

void goRight()
{
    // Turn left
    analogWrite(in1, 150);
    analogWrite(in2, 0);
    analogWrite(in3, 0);
    analogWrite(in4, 150);
    Serial.print("me working");
}

void goLeft()
{
    // Turn right
    analogWrite(in1, 0);
    analogWrite(in2, 150);
    analogWrite(in3, 150);
    analogWrite(in4, 0);
}

void adjustLeft()
{
    analogWrite(in1, 150);
    analogWrite(in2, 0);
    analogWrite(in3, 230);
    analogWrite(in4, 0);
}

void adjustRight()
{
    analogWrite(in1, 230);
    analogWrite(in2, 0);
    analogWrite(in3, 150);
    analogWrite(in4, 0);
}

void stopMotors()
{
    // Stop
    analogWrite(in1, 0);
    analogWrite(in2, 0);
    analogWrite(in3, 0);
    analogWrite(in4, 0);
}
void adjustBackRight(){
    analogWrite(in1, 0);
    analogWrite(in2, 190);
    analogWrite(in3, 0);
    analogWrite(in4, 100);
}

void adjustBackLeft(){
    analogWrite(in1, 0);
    analogWrite(in2, 0);
    analogWrite(in3, 0);
    analogWrite(in4, 150);
}

// bool isNode(){
//   return ((ir2 & ir4))
// }
