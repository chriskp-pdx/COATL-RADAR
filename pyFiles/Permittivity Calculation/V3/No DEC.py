import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import time

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
sensor_config.start_point = 23
sensor_config.num_points = 50
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
bean_scans = []

for i in range(num_groups):
    print(f"Starting bean scan group {i+1}/{num_groups}...")
    
    client.setup_session(sensor_config)
    Beanamplitudes = collect_averaged_scan(client, sensor_config)
    bean_scans.append(Beanamplitudes)

    print(f"Bean scan group {i+1} complete.")

    # Require user input before moving to the next group
    if i < num_groups - 1:
        input("Press Enter to continue to the next scan group...")

# Compute final averaged bean scan
Final_Beanamplitudes = np.mean(bean_scans, axis=0)

# Print final averaged amplitude
print("\nFinal Averaged Amplitude:", Final_Beanamplitudes)