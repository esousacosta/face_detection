#include <Servo.h>

int SERVO_PIN = 3;
int led_pin = 13;
int pot_pin = A0;
int angle;
int max_buffer = 3;
Servo my_servo;

/* The servo motor is controlled via angle and its limits are [0, 180]. */

void setup() {
  static char buffer[max_buffer];
  Serial.begin(9600);
  my_servo.attach(SERVO_PIN);
  my_servo.write(0);
}

void loop() {
  // put your main code here, to run repeatedly: 
  /* angle = analogRead(pot_pin); */
  /* angle = map(angle, 0, 1023, 0, 180); */

  // I want a code that checks if a received number is sandwiched in between two chars: '<' and '>',
  // and reads, at most, 3 digits for that number; if the number has more than 3 digits, I want the
  // code to tell the user they have input an invalid number.
  
  if (Serial.available() > 0) {
  angle = Serial.read() - '0';
  my_servo.write(angle);
  Serial.println(angle, DEC);
  }
  delay(20);
}
