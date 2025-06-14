import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import matplotlib.pyplot as plt

# Function to compute permittivity
def process_xm125_data(calamp, beanamp, caldist, beandist):
    
    # Constants
    c = 3e8  # Speed of light (m/s)
    f = 60e9  # Frequency of XM125 (Hz)
    w = 2 * np.pi * f  # Angular frequency
    epsilonnot = 8.854 * 10 ** (-12) # Epsilon Not
    
    amp_cal = np.max(np.array(calamp))
    amp_bean = np.max(np.array(beanamp))
    
    max_index = np.argmax(calamp)
    d = caldist[max_index]
    
    # Compute attenuation
    alpha = np.log(amp_cal / amp_bean) / d  # Attenuation constant
    
    #Get Q and I Separately
    realbeans = np.real(amp_bean)
    imaginarybeans = np.imag(amp_bean)
    realcal = np.real(amp_cal)
    imaginarycal = np.imag(amp_cal)
    
    # Estimate phase shift (simplified approach)
    phical = np.arctan(imaginarycal/realcal)
    phibean = np.arctan(imaginarybeans/realbeans)
    beta =  (phibean - phical)/ d  # Phase constant
    
    # Compute permittivity
    epsilon_real = ((beta * c / w) ** 2) * (1 / epsilonnot)
    epsilon_imag = (2 * alpha * c) / (w * epsilonnot)
    
    return epsilon_real, epsilon_imag, beta, alpha

# Setup sensor client
client = a121.Client.open(serial_port="COM8")
sensor_id = 1
sensor_config = a121.SensorConfig()
sensor_config.step_length = 1
sensor_config.start_point = 80
sensor_config.num_points = 250
sensor_config.sweeps_per_frame = 1
sensor_config.hwaas = 25
sensor_config.profile = et.a121.Profile.PROFILE_1
sensor_config.prf = 19.5e6
sensor_config.receiver_gain = 12

# Configure and start session
client.setup_session(sensor_config)
client.start_session()
data = client.get_next()

# Extract amplitude and distances
Caldistances = [sensor_config.start_point * 2.5 + i * sensor_config.step_length * 2.5 for i in range(sensor_config.num_points)]
Calamplitudes = data.frame[0].tolist()

# Stop session
client.stop_session()

while True:
    Continue_Condition = input("Please enter 'C' to continue: ")
    if Continue_Condition == "C" or Continue_Condition == "c":
        break

# Configure and start session
client.setup_session(sensor_config)
client.start_session()
data = client.get_next()

# Extract amplitude and distances
Beandistances = [sensor_config.start_point * 2.5 + i * sensor_config.step_length * 2.5 for i in range(sensor_config.num_points)]
Beanamplitudes = data.frame[0].tolist()

# Stop session
client.stop_session()

# Compute permittivity
epsilon_real, epsilon_imag, beta, alpha = process_xm125_data(Calamplitudes, Beanamplitudes, Caldistances, Beandistances)
epsilon = np.sqrt((epsilon_real**2)+(epsilon_imag**2))
epsilon = np.real(epsilon)
print("The epsilon value is: ", epsilon, " Beta is: ", beta, "Alpha is: ", alpha)