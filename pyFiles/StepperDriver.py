import time
import board
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper

# Initialize the MotorKit instance for the default I2C bus
kit = MotorKit(i2c=board.I2C())

# Define stepper motor speed (in seconds between steps)
step_delay = 0.01  # Adjust this value to set the speed

# Function to drive stepper motor continuously
def drive_stepper_motor(steps, direction, style):
    for _ in range(steps):
        if direction == 'forward':
            kit.stepper1.onestep(direction=stepper.FORWARD, style=style)
        elif direction == 'backward':
            kit.stepper1.onestep(direction=stepper.BACKWARD, style=style)
        time.sleep(step_delay)

# Main loop to run the stepper motor continuously
try:
    while True:
        drive_stepper_motor(200, 'forward', stepper.SINGLE)  # Adjust steps as needed
        drive_stepper_motor(200, 'backward', stepper.SINGLE)  # Adjust steps as needed
except KeyboardInterrupt:
    print("Motor stopped by user")

# Release the stepper motor
kit.stepper1.release()
