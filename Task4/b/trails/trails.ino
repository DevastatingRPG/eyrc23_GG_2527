// if this doesn't work try adjusting percent se
// Motor pin connections
#define in1 16
#define in2 4
#define in4 2
#define in3 15
// ir pin connection

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
const char *ssid = "Redmi 12 5G";   // Enter your wifi hotspot ssid
const char *password = "vijit3457"; // Enter your wifi hotspot password
const uint16_t port = 8002;
const char *host = "192.168.228.186"; // Enter the ip address of your laptop after connecting it to wifi hotspot

// External peripherals
int buzzerPin = 19;
int redLed = 21;
int ir1, ir2, ir3, ir4, ir5;
int type=1, t=1;

char incomingPacket[80];
WiFiClient client;
String msg;
int op;
int counter = 0;
int flag = 0;

void handleClientCommands();
void menu();
void moveForward();
void moveBackward();
void adjustLeft();
void adjustRight();
void turnLeft();
void turnRight();
void turnRightSlightly();
void turnLeftSlightly();

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

    // Keeping all motors off initially
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);

    pinMode(redLed, OUTPUT);
    digitalWrite(redLed, LOW);

    // // Connecting to wifi
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
    //         Serial.print("WiFi connected with IP: 2382347\n");
    //         flag = 1;
    //         break;
    //     }
    //     else
    //     {
    //         Serial.print("Connection Failed\n");
    //         counter++;
    //         delay(1000);
    //     }
    // }
    // if (!flag)
    // {
    //     exit(1);
    // }

    // Serial.print(WiFi.localIP());
}

bool first_detection = false;
int i = 0;

void jiggle();

void loop()
{
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
    // Read sensor values
    int start = millis();

    while (millis() - start < 10000)
        continue;

    ir1 = digitalRead(irsen1); // Front Left
    ir2 = digitalRead(irsen2); // Center Left
    ir3 = digitalRead(irsen3); // Front Center
    ir4 = digitalRead(irsen4); // Center Right
    ir5 = digitalRead(irsen5); // Front Right
    digitalWrite(redLed, HIGH);
    // handleClientCommands();
    // move1();

    // move1();

    // jiggle();

    move1();

    // stopMotors();
    // }
    // else
    // {
    //     Serial.println("Exceeded connection attempts. Waiting before retrying...\n");
    //     delay(500); // Wait for 5 seconds before retrying
    // }
}

void signal()
{
    T = millis();
    while (millis() - T < 1000)
    {
        digitalWrite(buzzerPin, HIGH);
    }
    digitalWrite(buzzerPin, LOW);
}

void signalExtreme()
{
    T = millis();
    while (millis() - T < 1000)
    {
        digitalWrite(redLed, HIGH);
        digitalWrite(buzzerPin, HIGH);
    }
    digitalWrite(redLed, LOW);
    digitalWrite(buzzerPin, LOW);
}

bool isNode()
{
    ir2 = digitalRead(irsen2);
    ir4 = digitalRead(irsen4);
    return ir2 & ir4;
}

bool toggle()
{
    ir1 = digitalRead(irsen1);
    ir3 = digitalRead(irsen3);
    ir5 = digitalRead(irsen5);
    if (ir1 == HIGH and ir3 == HIGH and ir5 == HIGH)
        return true;
    else
        return false;
}

void jiggle()
{
    while (1)
    {
        int tame = millis();
        adjustRight();
        while (millis() - tame < 1500)
            continue;
        stopMotors();

        tame = millis();
        adjustLeft();
        while (millis() - tame < 1500)
            continue;
        stopMotors();

        ir1 = digitalRead(irsen1); // Front Left
        ir2 = digitalRead(irsen2); // Center Left
        ir3 = digitalRead(irsen3); // Front Center
        ir4 = digitalRead(irsen4); // Center Right
        ir5 = digitalRead(irsen5);
    }
}

bool detectPath()
{
    if ((~ir1 & ~ir5) or ir3 == LOW)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void transition()
{
    moveForward();
    while (toggle())
    {
        continue;
    }
    // if (ir3 == HIGH){

    // }
}

bool offroad()
{
    ir1 = digitalRead(irsen1);
    ir3 = digitalRead(irsen3);
    ir5 = digitalRead(irsen5);
    // ir1 = digitalRead(irsen1);
    // ir1 = digitalRead(irsen1);
    if (ir3 & type == 1)
    {
        return false;
    }
    else if ((ir1 & ir5) & type == 2)
    {
        return false;
    }
    else if ((ir1 & ir3 & ir5) & type == 3)
    {
        return false;
    }
    else
        return true;
}

void adjust()
{
    if (type == 1)
    {
        moveForward();
        while (~ir2 & ~ir4)
        {
            // Read sensor values
            ir2 = digitalRead(irsen2); // Center Left
            ir4 = digitalRead(irsen4); // Center Right
        }
        stopMotors();
        if (ir4)
        {
            turnRight();
            while (~ir3)
            {
                // Read sensor values
                ir3 = digitalRead(irsen3); // Front Center
            }
            stopMotors();
        }
        else if (ir2)
        {
            turnLeft();
            while (~ir3)
            {

                // Read sensor values
                ir3 = digitalRead(irsen3); // Front Center
            }
            stopMotors();
        }
    }
    else if (type == 2)
    {
        if (ir1)
        {
            adjustRight();
            while (~ir5)
            {

                // Read sensor values
                ir5 = digitalRead(irsen5); // Front Right
            }
            stopMotors();
        }
        else if (ir5)
        {
            adjustLeft();
            while (~ir1)
            {

                // Read sensor values
                ir1 = digitalRead(irsen1); // Front Left
            }
            stopMotors();
        }
    }
    else if (type == 3)
    {
        moveBackward();
        while (~ir3)
        {
            ir1 = digitalRead(irsen1);
            ir2 = digitalRead(irsen2);
            ir3 = digitalRead(irsen3);
            ir4 = digitalRead(irsen4);
            ir5 = digitalRead(irsen5);
        }
        // while ((~ir2 & ~ir4)){
        //   ir2 = digitalRead(irsen2);
        //   ir4 = digitalRead(irsen4);
        // }
        stopMotors();


        analogBack();
        while (~ir2 & ~ir4){
          ir2 = digitalRead(irsen2);
          ir4 = digitalRead(irsen4);
        }
        stopMotors();

        if (ir4)
        {
            turnLeft();
            while (~ir3)
            {
                // Read sensor values
                ir3 = digitalRead(irsen3); // Front Center
            }
            stopMotors();
        }
        else if (ir2)
        {
            turnRight();
            while (~ir3)
            {

                // Read sensor values
                ir3 = digitalRead(irsen3); // Front Center
            }
            stopMotors();
        }
        // else if (ir3)
        // {
        //     moveForward();
        //     while (~ir1 & ~ir5)
        //     {
        //         ir1 = digitalRead(irsen1);
        //         ir5 = digitalRead(irsen5);
        //     }
        //     stopMotors();
        //     if (ir1)
        //     {
        //         adjustLeft();
        //         while (~ir5)
        //         {

        //             // Read sensor values
        //             ir5 = digitalRead(irsen5); // Front Right
        //         }
        //         stopMotors();
        //     }
        //     if (ir5)
        //     {
        //         adjustRight();
        //         while (~ir1)
        //         {
        //             ir1 = digitalRead(irsen1);
        //         }
        //         stopMotors();
        //     }
        // }
        // else if (ir1)
        // {
        //     adjustLeft();
        //     while (~ir5)
        //     {

        //         // Read sensor values
        //         ir5 = digitalRead(irsen5); // Front Right
        //     }
        //     stopMotors();
        // }
        // else if (ir5)
        // {
        //     adjustRight();
        //     while (~ir1)
        //     {
        //         ir1 = digitalRead(irsen1);
        //     }
        //     stopMotors();
        // }
    }
}

void move1()
{
    digitalWrite(redLed, LOW);
    ir1 = digitalRead(irsen1); // Front Left
    ir2 = digitalRead(irsen2); // Center Left
    ir3 = digitalRead(irsen3); // Front Center
    ir4 = digitalRead(irsen4); // Center Right
    ir5 = digitalRead(irsen5); // Front Right
    moveForward();
    while (!isNode())
    {
        if (ir1 & ir3 & ir5)
        {
          t = type;
            type = 3;
        }
        else if (ir3)
        {
          t = type;
            type = 1;
        }
        else if (ir1 & ir5)
        {
          t=type;
            type = 2;
        }
        if (offroad())
        {
            signal();
            adjust();
            moveForward();
        }
        // Read sensor values
        ir1 = digitalRead(irsen1); // Front Left
        ir2 = digitalRead(irsen2); // Center Left
        ir3 = digitalRead(irsen3); // Front Center
        ir4 = digitalRead(irsen4); // Center Right
        ir5 = digitalRead(irsen5); // Front Right
    }
    // stopMotors();
    signal();
}

void moveForward()
{
    // Move forward
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
}

void moveBackward()
{
    // Move backward
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
}

void analogBack(){
    digitalWrite(in1, 0);
    digitalWrite(in2, 100);
    digitalWrite(in3, 0);
    digitalWrite(in4, 100);
}

void turnLeft()
{
    // Turn left
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
}

void turnRight()
{
    // Turn right
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
    Serial.print("me working");
}

void adjustLeft()
{
    analogWrite(in1, 135);
    analogWrite(in2, 0);
    analogWrite(in3, 200);
    analogWrite(in4, 0);
}

void adjustRight()
{
    analogWrite(in1, 210);
    analogWrite(in2, 0);
    analogWrite(in3, 135);
    analogWrite(in4, 0);
}
void stopMotors()
{
    // Stop
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);
}
