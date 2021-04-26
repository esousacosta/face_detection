#include <Servo.h>

int SERVO_PIN = 3;
int led_pin = 13;
unsigned int angle;
bool receiving = false;
Servo my_servo;

// Variables to control the data input from the Serial monitor
// the limit is four because the array is terminated by '\0'.
const byte max_buffer = 4;

// Defining the delimiting chars
const char start_byte = '<';
const char stop_byte = '>';

/* The servo motor is controlled via angle and its limits are [0, 180]. */

void setup() {
  Serial.begin(9600);
  my_servo.attach(SERVO_PIN);
  my_servo.write(0);
}

void loop() {
  static byte index = 0;
  static char buffer[max_buffer];

  // I want a code that checks if a received number is sandwiched in between two chars: '<' and '>',
  // and reads, at most, 3 digits for that number; if the number has more than 3 digits, I want the
  // code to tell the user they have input an invalid number. For instance, if I type: <150> in
  // the terminal, I want the value read to be 150.
  
  if (Serial.available() > 0) {
	temp = Serial.read();
	if (temp == start_byte) {
	  index = 0;
	  Serial.print("<");
	  receiving = true;
	} else if (temp == stop_byte) {
	  buffer[index] = '\0';
	  receiving = false;
	  index = -1;
	  Serial.println(">");
	  process_input_data(buffer, angle);
	} else {
	  if (receiving && index < max_buffer - 1) {
		buffer[index] = temp;
		Serial.print(temp);
		index++;
	  } else if (receiving && index >= max_buffer - 1) {
			Serial.println("Max buffer size exceeded! Discarding...");
			receiving = false;
			index = -1;
			clear_serial();
			clear_buffer(buffer);
	  }
	}
  }
  delay(20);
}


void process_input_data(char buffer[], unsigned int &angle)
{
  // This function doesn't work as the array is passed as a pointer to the
  // array's first element, messing up the memory access for memcpy.
  angle = atoi(buffer);
  control_servo(angle);
  clear_buffer(buffer);
  clear_serial();
}

void control_servo(unsigned int angle)
{
  // This function send the desired angle to the motor and
  // writes it on the serial montior.
  my_servo.write(angle);
  Serial.print("Writing angle = ");
  Serial.println(angle, DEC);
}

void clear_serial()
{
  // This function clears the serial input buffer in case of a badly feeded angle.
  char tmp;
  while (Serial.available() > 0)
	tmp = Serial.read();
}

void clear_buffer(char *buffer)
{
  // This function clears the buffer to make sure we don't read
  // reminiscent values from past inputs.
  for (int i = 0; i <= max_buffer; i++)
	buffer[i] = ' ';
}