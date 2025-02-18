# Kamal Smith COATL Project 2/17/25 (Modified to run infinitely until we get a connection or turn off)

# This is a simple code, for testing the IC2. I Used chat GPT for the backbone
# to get the basic code, and have added to this for testing purposes.

# This code will test GPIO3(SDA) and GPIO4(SCL) to see if there is a device connected to
# these pins. If the device is found, the red LED will light up and stay on indicating
# there is an I2C device found. Otherwise the LED will blink 3 times indicating that no
# I2C device is found. This code has been modified from the first version to continue
# to run and check if there is an I2C connection.

# Importing all the libraries that we may need here
# Library Documentation https://docs.circuitpython.org/en/latest/shared-bindings/index.html
import busio
import microcontroller
import digitalio
import time
import board

# I have removed this code and opted to just use the onboard RED LED on board
# Initialize the LED on a GPIO pin (change GPIO5 to your chosen pin)
# led = digitalio.DigitalInOut(microcontroller.pin.GPIO5)
# led.direction = digitalio.Direction.OUTPUT

# Setting up the small red LED as an output. This is to help indicate we have an I2C connection
LED = digitalio.DigitalInOut(board.LED)
LED.direction = digitalio.Direction.OUTPUT

# Initialize I2C on GPIO3 (SCL) and GPIO4 (SDA)
i2c = busio.I2C(microcontroller.pin.GPIO3, microcontroller.pin.GPIO4)

while True:
# Attempts to grab I2C lock, Returns true on success (type bool)
    while not i2c.try_lock():
        pass

    try:
    #LED.value = True
        devices = i2c.scan() # Scan all I2C addresses and return a list of those that respond
    #time.sleep(2)
    #LED.value = False

# If a device is found, turn on the red LED for 3 seconds then turn off
        if devices:
            LED.value = True
            time.sleep(3)
            print("I2C devices found:", [hex(device) for device in devices])

# Device not found, blink the red LED 3 times
        else:
            LED.value = True
            time.sleep(.5)
            LED.value = False
            time.sleep(.5)
            LED.value = True
            time.sleep(.5)
            LED.value = False
            time.sleep(.5)
            LED.value = True
            time.sleep(.5)
            LED.value = False
            time.sleep(.5)
            
            print("No I2C devices found.")
# Release the I2C lock
    finally:
        i2c.unlock()

        time.sleep(2)
