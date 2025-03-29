import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import time
import scipy.constants as sp

# Constants
epsilonnot = sp.epsilon_0  # Vacuum permittivity
c = sp.c  # Speed of light (m/s)
f = 60e9  # Frequency of XM125 (60 GHz)
w = 2 * np.pi * f  # Angular frequency
d = 0.0125  # Distance (in meters)

# Function to compute permittivity
def process_xm125_data(calamp, beanamp):
    # Use the maximum values from averaged amplitudes
    max_amp_cal = np.max(np.abs(calamp))
    max_amp_bean = np.max(np.abs(beanamp))
    phase_cal = np.angle(calamp[np.argmax(np.abs(calamp))])  # Phase at max amplitude
    phase_bean = np.angle(beanamp[np.argmax(np.abs(beanamp))])  # Phase at max amplitude

    print(f"Max calibration amplitude: {max_amp_cal}")
    print(f"Max bean amplitude: {max_amp_bean}")
    print(f"Distance: {d} m")

    # Check if amplitudes are non-zero
    if max_amp_bean == 0 or max_amp_cal == 0:
        print("Error: Zero amplitude detected, check signal quality.")
        return None, None, None, None

    # Calculate attenuation constant (alpha) and phase constant (beta)
    alpha = np.log(max_amp_cal / max_amp_bean) / d  # Attenuation constant
    beta = (phase_bean - phase_cal) / d  # Phase constant

    # Print intermediate results
    print(f"Alpha (attenuation constant): {alpha}")
    print(f"Beta (phase constant): {beta}")

    # Compute permittivity (use only real part)
    epsilon_real = ((beta * c / w) ** 2) / epsilonnot

    print(f"Epsilon real part: {epsilon_real}")

    return epsilon_real, beta, alpha

# Function to collect and average multiple scans
def collect_averaged_scan(client, sensor_config, num_scans=10, delay=0.25):
    total_amplitudes = np.zeros(sensor_config.num_points, dtype=np.complex128)

    for _ in range(num_scans):
        client.start_session()
        data = client.get_next()
        client.stop_session()

        amplitudes = np.array(data.frame[0].tolist())
        total_amplitudes += amplitudes

        # Print raw data for debugging
        print(f"Raw amplitudes (first scan): {amplitudes}")
        
        time.sleep(delay)

    averaged_amplitudes = total_amplitudes / num_scans
    print(f"Averaged amplitudes: {averaged_amplitudes}")
    
    return averaged_amplitudes

# Setup sensor client
client = a121.Client.open(serial_port="COM8")

# Define sensor configuration
sensor_config = a121.SensorConfig()
sensor_config.step_length = 1
sensor_config.start_point = 40
sensor_config.num_points = 50
sensor_config.sweeps_per_frame = 1
sensor_config.hwaas = 60
sensor_config.profile = et.a121.Profile.PROFILE_1
sensor_config.prf = 19.5e6
sensor_config.receiver_gain = 23

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
epsilon_real, beta, alpha = process_xm125_data(Calamplitudes, Beanamplitudes)

if epsilon_real is not None:
    print("\nFinal Computed Epsilon Value (real part):", epsilon_real)
    print("Beta:", beta)
    print("Alpha:", alpha)
else:
    print("Error in computation, check debug prints above.")

# Close the client session
client.close()