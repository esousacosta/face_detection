#include <Servo.h>

int SERVO_PIN = 3;
Servo my_servo;
int led_pin = 13;
int angle;
int pot_pin = A0;

/* The servo motor is controlled via angle and its limits are [0, 180]. */

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  my_servo.attach(SERVO_PIN);
  my_servo.write(0);
}

void loop() {
  // put your main code here, to run repeatedly: 
  angle = analogRead(pot_pin);
  angle = map(angle, 0, 1023, 0, 180);
  my_servo.write(angle);
  delay(20);
}
