import board
import busio
import microcontroller
import digitalio
import time
from adafruit_bus_device.i2c_device import I2CDevice
#import acconeer.exptool as et
#from acconeer.exptool import a121

LED = digitalio.DigitalInOut(board.LED)
LED.direction = digitalio.Direction.OUTPUT

#indicator = digitalio.DigitalInOut(board.D13)
#indicator.direction = digitalio.Direction.OUTPUT

# XE125 sensor 7-bit I2C address (default if I2C_ADDR pin is not connected)
XE125_ADDR = 0x52

# A known register to read (Version register at 0x0000)
REG_VERSION = 0x0000

# Initialize I²C bus
#i2c = busio.I2C(SCL = board.SCL, SDA = board.SDA)
i2c = busio.I2C(microcontroller.pin.GPIO4, microcontroller.pin.GPIO3)
#i2c = busio.I2C(board.SCL, board.SDA)
MCU = digitalio.DigitalInOut(board.D11)
MCU.direction = digitalio.Direction.INPUT

WakeUp = digitalio.DigitalInOut(board.D12)
WakeUp.direction = digitalio.Direction.OUTPUT
WakeUp.value = True

time.sleep(3)

#if MCU.value:
while True:
        
    # Wait until the I²C bus is available
    while not i2c.try_lock():
        pass
    try:
        # Scan the bus for devices
            devices = i2c.scan()
    
            #if MCU.value = True
            if XE125_ADDR in devices:
                LED.value = True
                #WakeUp.value = True
                time.sleep(3)
                print("XE125 sensor detected on I²C bus.")
                #print (MCU.value)
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
                print("XE125 sensor NOT detected on I²C bus.")
                    #print (MCU.value)
    finally: 
        i2c.unlock()
        time.sleep(1)
#else:
    #print("MCU_INT not high")

    
def read_register(reg_addr):
    """
    Reads a 32-bit value from the specified register.
    The register address is two bytes (big-endian) followed by reading 4 data bytes.
    """
    addr = bytearray(2)
    addr[0] = (reg_addr >> 8) & 0xFF
    addr[1] = reg_addr & 0xFF
    try:
        i2c.writeto(XE125_ADDR, addr, stop=False)
        result = bytearray(4)
        i2c.readfrom_into(XE125_ADDR, result)
        value = (result[0] << 24) | (result[1] << 16) | (result[2] << 8) | result[3]
        return value
    except Exception as e:
        print("Error reading register 0x{:04X}:".format(reg_addr), e)
        return None

# Attempt to read the Version register
version = read_register(REG_VERSION)
if version is not None:
    major = (version >> 16) & 0xFFFF
    minor = (version >> 8) & 0xFF
    patch = version & 0xFF
    print("Version register: {}.{}.{}".format(major, minor, patch))
    print("Sensor and Feather are connected properly!")
else:
    print("Failed to read Version register; check wiring and power.")

#i2c.unlock()