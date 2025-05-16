#Chris Kane-Pardy
#Acconeer XE125 EVK Calibration & Sparse IQ Setup for Coffee Radar

import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np

#Connecting COM port to exptool via Python
client = a121.Client.open(
    serial_port="COM8" #Use Local Port to Your PC
)

#Assign a sensor ID from a board or module or evaluation kit (EVK)
sensor_id = 1

#Setup Base Sensor Config
sensor_config = a121.SensorConfig()
#Alter step length, multiply by 2.5mm for actual distance (must be a divisor or multiple of 24)
sensor_config.step_length = 1 #2.5mm step size
#Start Point = 2.5mm * start point
sensor_config.start_point = 15 #37.5mm start
#Total Distance = (num_point * step length * 2.5mm) from start point
sensor_config.num_points = 51 #165mm end
#SpF set to 1 for static objects
sensor_config.sweeps_per_frame = 1
#HWAAS to 25 for better SNR 
sensor_config.hwaas = 25
#Set to Profile 1 for close distance measurement accuracy
sensor_config.profile = et.a121.Profile.PROFILE_1 #Profile 1, DO NOT CHANGE
#PRF highest since our distance is short
sensor_config.prf = 19.5e6 #19.5 MHz, DO NOT CHANGE
#Confirm Sensor Config
client.setup_session(sensor_config)

#Start session
client.start_session()

#Collect one frame of data
data = client.get_next()

#Extract amplitude and corresponding distances
distances = [sensor_config.start_point * 2.5 + i * sensor_config.step_length * 2.5 for i in range(sensor_config.num_points)] #Correctly identify distances
amplitudes = data.frame[0].tolist()  # Extract amplitude values
realamplitudes = np.real(amplitudes)
imaginaryamplitudes = np.imag(amplitudes)
#combinedamplitudes = (amplitudes[0]^2 + amplitudes[1]^2)^(-2) #THIS IS CURRENTLY UNTESTED BUT CAN BE USED FOR ACCURACY COMPARISON w/ exptool

#Print labeled amplitude and distance values in tabulated format
print("Distance (mm)\tAmplitude")
for d, a, ra, ia in zip(distances, amplitudes, realamplitudes, imaginaryamplitudes):
    print(f"{d:.2f}\t{a:.2f}\t{ra:.2f}\t{ia:.2f}")

#Stop session and disconnect
client.stop_session()
client.close()
