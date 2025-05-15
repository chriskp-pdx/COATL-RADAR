# Team 4 COATL RADAR BeanTube Testing
# This program by Kamal is a modified version of chris's program that would output the max amplitude along with
# other information from the IQ Data. Rather than giving other info, this program will simply
# output the max amplitude along with the distance that it found it at. 
# It will do this 4 times to ensure that you get 4 different data points at different times


import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import matplotlib.pyplot as plt
import time

# Function to compute permittivity
def process_xm125_data(calamp, beanamp, caldist, beandist):
    
    # Constants
    c = 3e8  # Speed of light (m/s)
    f = 60e9  # Frequency of XM125 (Hz)
    w = 2 * np.pi * f  # Angular frequency
    epsilonnot = 8.854 * 10 ** (-12) # Epsilon Not
    
    amp_cal = np.max(np.array(calamp))
    amp_bean = np.max(np.array(beanamp))
    
    max_index = np.argmax(np.abs(beanamp))
    d = beandist[max_index]
    
    # Compute attenuation
    alpha = np.log(amp_cal / amp_bean) / d  # Attenuation constant
    
    #Get Q and I Separately
    realbeans = np.real(amp_bean)
    imaginarybeans = np.imag(amp_bean)
    realcal = np.real(amp_cal)
    imaginarycal = np.imag(amp_cal)
    
    # Estimate phase shift (simplified approach)
    phical = np.arctan2(imaginarycal, realcal)
    phibean = np.arctan2(imaginarybeans, realbeans)
    beta =  (phibean - phical)/ d  # Phase constant
    
    # Compute permittivity
    epsilon_real = ((beta * c / w) ** 2) * (1 / epsilonnot)
    epsilon_imag = (2 * alpha * c) / (w * epsilonnot)
    
    return phical, phibean, amp_cal, amp_bean

# Setup sensor client
client = a121.Client.open(serial_port="COM5")
sensor_id = 1
sensor_config = a121.SensorConfig()
sensor_config.step_length = 1
sensor_config.start_point = 40
sensor_config.num_points = 50
sensor_config.sweeps_per_frame = 1
sensor_config.hwaas = 60
sensor_config.profile = et.a121.Profile.PROFILE_1
sensor_config.prf = 19.5e6
sensor_config.receiver_gain = 23


# Commenting this out for now, I tested without using the empty tube and it gave me the same results

# Configure and start session

#client.setup_session(sensor_config)
#client.start_session()
#data = client.get_next()

# Extract amplitude and distances

#Caldistances = [sensor_config.start_point * 2.5 + i * sensor_config.step_length * 2.5 for i in range(sensor_config.num_points)]
#Calamplitudes = data.frame[0].tolist()

# Stop session

#client.stop_session()

# When program is ready to read the amplitude, it will ask you to input "c"
while True:
    Continue_Condition = input("Please enter 'C' to continue: ")
    if Continue_Condition == "C" or Continue_Condition == "c":
        break

# Modifying this section to take 4 different readings
# The readings will be read one after another with a 2 second sleep in between to ensure it can grab a new data point

for i in range(4):
    client.setup_session(sensor_config)
    client.start_session()
    data = client.get_next()
    
    print(f"Reading {i+1}:")
    Beandistances = [sensor_config.start_point * 2.5 + j * sensor_config.step_length * 2.5 for j in range(sensor_config.num_points)]
    Beanamplitudes = np.abs(np.array(data.frame[0].tolist()))  # Ensure absolute values for amplitude
    
    max_index = np.argmax(Beanamplitudes)
    max_amp_bean = Beanamplitudes[max_index]
    max_dist_bean = Beandistances[max_index]
    
    client.stop_session()
    
    print(f"Max amplitude for Reading {i+1}: {max_amp_bean} at distance {max_dist_bean}")
    
    if i < 3:
        time.sleep(2)
