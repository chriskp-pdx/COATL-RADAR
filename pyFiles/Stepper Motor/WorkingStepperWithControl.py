# Kamal Smith COATL Project 2/26/25 (Modified to run infinitely until we get a connection or turn off)

# This is the basic code to connect the I2C and began turning the motor.
# I am adding code so that the motor will began turning at the click of a button and can
# be stopped and started at will. I will mark the sections that we need to change if we 
# would like the speed of the motor to be faster or slower. 

import digitalio
import time
import board
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper

# setting up the Outputs on the board using CircuitPython naming
StartAndStop = board.A2 # for starting and stopping the motor
SpeedUp = board.A1 # to increase the speed of the motor
SpeedDown = board.A0 # to decrease the speed of the motor

# Setting up the digital IO
button = digitalio.DigitalInOut(StartAndStop)
button2 = digitalio.DigitalInOut(SpeedUp)
button3 = digitalio.DigitalInOut(SpeedDown)

# Setting Buttons as input with pull up resistors
button.direction = digitalio.Direction.INPUT
button.pull = digitalio.Pull.UP
button2.direction = digitalio.Direction.INPUT
button2.pull = digitalio.Pull.UP
button3.direction = digitalio.Direction.INPUT
button3.pull = digitalio.Pull.UP

# Create the I2C object using the STEMMA QT connector
i2c = board.STEMMA_I2C()

# Initialize the MotorKit at the default I2C address (0x60)
kit = MotorKit(i2c=i2c)

while True:
# This section is in progress
    if not button:

        print("Entered button section")


    elif not button2:

        print("Entered button2 section")
# Section in progress
    else:

        print("No button pressed, will continue on")

    print("Stepping forward a bit...")
    for _ in range(20):  # Step 20 times forward
        kit.stepper1.onestep(direction=stepper.FORWARD, style=stepper.SINGLE)
        # The stepper.DOUBLE is for utilizing both coils for more torque
        # Still figuring out the motor for this feature
        #kit.stepper1.onestep(direction=stepper.FORWARD, style=stepper.DOUBLE)
        time.sleep(0.01)  # Adjust to change speed

#    time.sleep(1)  # Pause so you can observe the move

#    print("Stepping backward a bit...")
#    for _ in range(20):  # Step 20 times backward
#        kit.stepper1.onestep(direction=stepper.BACKWARD, style=stepper.SINGLE)
#        time.sleep(0.01)

#    time.sleep(1)  # Pause before the next cycle
