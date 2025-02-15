import board
import busio
import time

# Initialize the I2C bus on the ESP32-S3.
# Use the default board definitions for SCL and SDA; you can also specify alternate pins if needed.
i2c = busio.I2C(board.SCL, board.SDA)

# Wait until the I2C bus is available.
while not i2c.try_lock():
    pass

try:
    # Scan for devices and print any found addresses.
    devices = i2c.scan()
    if devices:
        print("I2C devices found:", [hex(device) for device in devices])
    else:
        print("No I2C devices found.")
finally:
    i2c.unlock()
