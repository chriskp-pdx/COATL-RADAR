#Chris Kane-Pardy
#Acconeer XE125 EVK Calibration & Sparse IQ Setup for Coffee Radar

import acconeer.exptool as et
from acconeer.exptool import a121

#Connecting COM port to exptool via Python
client = a121.Client.open(
    serial_port="COM5" #Use Local Port to Your PC
)

#Assign a sensor ID from a board or module or evaluation kit (EVK)
sensor_id = 1

#Setup Base Sensor Config
sensor_config = a121.SensorConfig()
#Alter step length, multiply by 2.5mm for actual distance
sensor_config.step_length = 4
#Start Point = 2.5mm * start point
sensor_config.start_point = 0
#Total Distance = (num_point * step length * 2.5mm) from start point
sensor_config.num_points = 35
#SpF set to 1 for static objects
sensor_config.sweeps_per_frame = 1
#HWAAS to 25 for better SNR 
sensor_config.hwaas = 25
#Set to Profile 1
sensor_config.profile = et.a121.Profile.PROFILE_1
#PRF highest since our distance is short
sensor_config.prf = 19.5e6 #MHz
#Confirm Sensor Config
client.setup_session(sensor_config)

#Start session
client.start_session()

#Collect one frame of data
data = client.get_next()

#Extract amplitude and corresponding distances
distances = [sensor_config.start_point * 2.5 + i * sensor_config.step_length * 2.5 for i in range(sensor_config.num_points)]
amplitudes = data.frame[0].tolist()  # Extract amplitude values

#Print labeled amplitude and distance values
print("Distance (mm)\tAmplitude")
for d, a in zip(distances, amplitudes):
    print(f"{d:.2f}\t{a:.2f}")

#Stop session and disconnect
client.stop_session()
client.close()
