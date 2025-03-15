import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8  # Speed of light (m/s)
f = 60e9  # Frequency of XM125 (Hz)
w = 2 * np.pi * f  # Angular frequency

# Function to compute permittivity
def process_xm125_data(amplitudes, distances):
    """
    Compute permittivity (epsilon' and epsilon'') from amplitude data.
    :param amplitudes: Measured amplitude values from the radar.
    :param distances: Corresponding distances in meters.
    :return: epsilon' and epsilon''
    """
    amp_meas = np.array(amplitudes)
    d = np.array(distances) / 1000  # Convert mm to meters
    
    # Compute weighted average of distances
    weights = d / np.sum(d)  # Normalize distances as weights
    d_weighted = np.sum(d * weights)  # Weighted average distance
    
    # Simulated reference amplitude (assumption: initial amplitude is reference)
    amp_ref = np.full_like(amp_meas, amp_meas[0])
    
    # Compute attenuation
    alpha = np.log(amp_ref / amp_meas) / d_weighted  # Attenuation constant
    
    # Estimate phase shift (simplified approach)
    delta_phi = np.pi / 4 * np.ones_like(alpha)  # Placeholder phase shift
    beta = delta_phi / d_weighted  # Phase constant
    
    # Compute permittivity
    epsilon_real = (beta * c / w) ** 2
    epsilon_imag = (2 * alpha * c) / w
    
    return epsilon_real, epsilon_imag

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

# Configure and start session
client.setup_session(sensor_config)
client.start_session()
data = client.get_next()

# Extract amplitude and distances
distances = [sensor_config.start_point * 2.5 + i * sensor_config.step_length * 2.5 for i in range(sensor_config.num_points)]
amplitudes = data.frame[0].tolist()

# Stop session
client.stop_session()
client.close()

# Compute permittivity
epsilon_real, epsilon_imag = process_xm125_data(amplitudes, distances)

# Plot results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(distances, epsilon_real, label="Real Permittivity (ε')")
plt.xlabel("Distance (mm)")
plt.ylabel("ε'")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(distances, epsilon_imag, label="Imaginary Permittivity (ε'')", color='r')
plt.xlabel("Distance (mm)")
plt.ylabel("ε''")
plt.legend()

plt.tight_layout()
plt.show()
