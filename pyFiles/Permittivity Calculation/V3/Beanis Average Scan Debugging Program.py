import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import time

# Function to compute permittivity
def process_xm125_data(calamp, beanamp, caldist, beandist):
    # Constants
    c = 3e8  # Speed of light (m/s)
    f = 60e9  # Frequency of XM125 (Hz)
    w = 2 * np.pi * f  # Angular frequency
    epsilonnot = 8.854 * 10 ** (-12)  # Permittivity of free space
    
    # Get max amplitude and corresponding distance
    max_index = np.argmax(calamp)
    d = caldist[max_index]

    amp_cal = np.max(np.array(calamp))
    amp_bean = np.max(np.array(beanamp))
    
    # Debugging prints
    print(f"Max calibration amplitude: {amp_cal}")
    print(f"Max bean amplitude: {amp_bean}")
    print(f"Distance at max amplitude: {d}")

    # Compute attenuation
    if amp_bean == 0 or amp_cal == 0:  # Prevent log(0) issues
        print("Error: Zero amplitude detected, check signal quality.")
        return None, None, None, None

    alpha = np.log(amp_cal / amp_bean) / d  # Attenuation constant
    print(f"Computed alpha (attenuation constant): {alpha}")

    # Compute phase shift more robustly
    phical = np.angle(amp_cal)
    phibean = np.angle(amp_bean)
    beta = (phibean - phical) / d  # Phase constant
    print(f"Computed beta (phase constant): {beta}")

    # Compute permittivity
    epsilon_real = ((beta * c / w) ** 2) * (1 / epsilonnot)
    epsilon_imag = (2 * alpha * c) / (w * epsilonnot)

    # Debugging prints
    print(f"Epsilon real part: {epsilon_real}")
    print(f"Epsilon imaginary part: {epsilon_imag}")

    return epsilon_real, epsilon_imag, beta, alpha

# Function to collect and average multiple scans
def collect_averaged_scan(client, sensor_config, num_scans=10, delay=0.25):
    total_amplitudes = np.zeros(sensor_config.num_points)

    for _ in range(num_scans):
        client.start_session()
        data = client.get_next()
        client.stop_session()
        
        amplitudes = np.array(data.frame[0].tolist())
        total_amplitudes += amplitudes
        
        time.sleep(delay)  # Wait between scans

    averaged_amplitudes = total_amplitudes / num_scans
    distances = [sensor_config.start_point * 2.5 + i * sensor_config.step_length * 2.5 for i in range(sensor_config.num_points)]
    
    return averaged_amplitudes, distances

# Setup sensor client
client = a121.Client.open(serial_port="COM8")
sensor_id = 1
sensor_config = a121.SensorConfig()
sensor_config.step_length = 1
sensor_config.start_point = 23
sensor_config.num_points = 50
sensor_config.sweeps_per_frame = 1
sensor_config.hwaas = 25
sensor_config.profile = et.a121.Profile.PROFILE_1
sensor_config.prf = 19.5e6
sensor_config.receiver_gain = 12

# Configure and collect calibration scan
client.setup_session(sensor_config)
Calamplitudes, Caldistances = collect_averaged_scan(client, sensor_config)

while True:
    Continue_Condition = input("Please enter 'C' to continue: ")
    if Continue_Condition.lower() == "c":
        break

# Configure and collect actual scan
client.setup_session(sensor_config)
Beanamplitudes, Beandistances = collect_averaged_scan(client, sensor_config)

# Compute permittivity
epsilon_real, epsilon_imag, beta, alpha = process_xm125_data(Calamplitudes, Beanamplitudes, Caldistances, Beandistances)

if epsilon_real is not None:
    epsilon = np.sqrt((epsilon_real**2) + (epsilon_imag**2))
    epsilon = np.real(epsilon)
    print("The computed epsilon value is:", epsilon)
    print("Beta is:", beta)
    print("Alpha is:", alpha)
else:
    print("Error in computation, check debug prints above.")
