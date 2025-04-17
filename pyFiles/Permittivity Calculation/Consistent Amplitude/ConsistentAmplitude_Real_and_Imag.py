# Consistent amplitude test with real and imaginary seperated

import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import time
import csv
import os

# Function to collect and average multiple scans
def MultiScanAverage(client, sensor_config, Scans, Delay= (500 * (10 ** (-9)))):
    total_amplitudes = np.zeros(sensor_config.num_points, dtype=np.complex128)

    for _ in range(Scans):
        client.start_session()
        data = client.get_next()
        client.stop_session()

        amplitudes = np.array(data.frame[0].tolist())
        total_amplitudes += amplitudes
        
        time.sleep(Delay)

    averaged_amplitudes = total_amplitudes / Scans
    return averaged_amplitudes

# Setup Sensor Client
client = a121.Client.open(serial_port="COM5")  # Adjust as necessary

# Define sensor configuration
sensor_config = a121.SensorConfig()
sensor_config.step_length = 1
sensor_config.start_point = 40
sensor_config.num_points = 50
sensor_config.sweeps_per_frame = 1
sensor_config.hwaas = 60
sensor_config.profile = et.a121.Profile.PROFILE_1
sensor_config.prf = 19.5e6
sensor_config.receiver_gain = 19
sensor_config.phase_enhancement = True

input("Press Enter to begin scans...")

BeanResults = []

# Repeat the full scan-and-process flow 10 times
for i in range(10):
    print(f"\n--- Processing iteration {i+1}/10 ---")

    # Collect and average 10 scans per group (50 scans each)
    ScanGroups = 10
    BeanAmplitudeSum = np.zeros(sensor_config.num_points, dtype=np.complex128)

    for j in range(ScanGroups):
        print(f"  Bean scan group {j+1}/{ScanGroups}...")
        client.setup_session(sensor_config)
        BeanData = MultiScanAverage(client, sensor_config, Scans=50)
        BeanAmplitudeSum += BeanData

    # Compute average amplitudes
    BeanAmplitudes = BeanAmplitudeSum / ScanGroups

    # Get index of max magnitude
    BeanMaxIndex = np.argmax(np.abs(BeanAmplitudes))
    BeanMaxComplex = BeanAmplitudes[BeanMaxIndex]

    real_part = BeanMaxComplex.real
    imag_part = BeanMaxComplex.imag
    magnitude = np.abs(BeanMaxComplex)
    phase_radians = np.angle(BeanMaxComplex)

    print(f"  Iteration {i+1}: Real={real_part}, Imag={imag_part}, Magnitude={magnitude}, Phase(rad)={phase_radians}")
    BeanResults.append((real_part, imag_part, magnitude, phase_radians))

# Close sensor session
client.close()

# Save results to CSV
csv_filename = "BeanMaxComplex_DetailedOutput.csv"
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Write header only if the file is new
    if not file_exists:
        writer.writerow(["RunID", "Iteration", "RealPart", "ImagPart", "Magnitude", "PhaseRadians"])

    for idx, (real_val, imag_val, mag_val, phase_val) in enumerate(BeanResults, 1):
        writer.writerow([time.strftime("%Y%m%d-%H%M%S"), idx, real_val, imag_val, mag_val, phase_val])

print(f"\nAppended {len(BeanResults)} entries to {csv_filename}.")
