import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import time

# Function to compute permittivity
def process_xm125_data(calamp, beanamp):
    c = 3* 10 ** 8  # Speed of light (m/s)
    f = 60 * 10 **9  # Frequency of XM125 (Hz)
    w = 2 * np.pi * f  # Angular frequency
    epsilonnot = (8.854) * 10 ** (-12)  # Permittivity of free space
    
    d = 0.0125

    amp_cal = np.max(np.array(calamp))
    amp_bean = np.max(np.array(beanamp))
    
    print(f"Max calibration amplitude: {amp_cal}")
    print(f"Max bean amplitude: {amp_bean}")
    print(f"Distance at max amplitude: {d}")

    if amp_bean == 0 or amp_cal == 0:
        print("Error: Zero amplitude detected, check signal quality.")
        return None, None, None, None

    alpha = np.log(amp_cal / amp_bean) / d  
    print(f"Computed alpha (attenuation constant): {alpha}")

    phical = np.angle(amp_cal)
    phibean = np.angle(amp_bean)
    beta = (phibean - phical) / d  
    print(f"Computed beta (phase constant): {beta}")

    epsilon_real = ((beta * c / w) ** 2) * (1 / epsilonnot)
    epsilon_imag = (2 * alpha * c) / (w * epsilonnot)

    print(f"Epsilon real part: {epsilon_real}")
    print(f"Epsilon imaginary part: {epsilon_imag}")

    return epsilon_real, epsilon_imag, beta, alpha

# Function to collect and average multiple scans
def collect_averaged_scan(client, sensor_config, num_scans=10, delay=0.25):
    total_amplitudes = np.zeros(sensor_config.num_points, dtype=np.complex128)

    for _ in range(num_scans):
        client.start_session()
        data = client.get_next()
        client.stop_session()
        
        amplitudes = np.array(data.frame[0].tolist())
        total_amplitudes += amplitudes
        
        time.sleep(delay)  

    averaged_amplitudes = total_amplitudes / num_scans
    
    return averaged_amplitudes

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

# Collect calibration scan (10 scans averaged)
client.setup_session(sensor_config)
print("Starting calibration scan...")
Calamplitudes = collect_averaged_scan(client, sensor_config)
print("Calibration scan complete.")

# Wait for user input to proceed
input("Press Enter to proceed to actual scans...")

# Collect 5 groups of bean scans, each averaging 10 scans
num_groups = 5
Beanamplitudes = np.zeros(sensor_config.num_points, dtype=np.complex128)

for i in range(num_groups):
    print(f"Starting bean scan group {i+1}/{num_groups}...")
    
    client.setup_session(sensor_config)
    Beansum = collect_averaged_scan(client, sensor_config)
    Beanamplitudes += Beansum
    print(f"Bean scan group {i+1} complete.")

    # Require user input before moving to the next group
    if i < num_groups - 1:
        input("Press Enter to continue to the next scan group...")

# Compute permittivity using final averaged scan
Beanamplitudes /= num_groups
epsilon_real, epsilon_imag, beta, alpha = process_xm125_data(Calamplitudes, Beanamplitudes)

if epsilon_real is not None:
    epsilon = np.sqrt((epsilon_real**2) + (epsilon_imag**2))
    epsilon = np.real(epsilon)
    print("\nFinal Computed Epsilon Value:", epsilon)
    print("Beta:", beta)
    print("Alpha:", alpha)
else:
    print("Error in computation, check debug prints above.")
