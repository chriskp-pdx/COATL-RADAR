# Henry Sanders ECE 413 COATL RADAR 4/7/2025

import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import time
import csv
import os

# Function to collect and average multiple scans
def MultiScanAverage(client, sensor_config, Scans, Delay= 0.250):
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

# Function to normalize complex amplitude into a single scalar value
def NormalizeAmplitude(complex_amplitude):
    magnitude = np.abs(complex_amplitude)
    phase = np.angle(complex_amplitude)
    phase = np.unwrap(phase)
    normalized_value = magnitude * np.cos(phase)
    return normalized_value

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

BeanMaxNormalizedList = []

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
        #if j < ScanGroups - 1:
            #input("  Press Enter to continue to the next scan group...")

    # Normalize and compute BeanMaxNormalized
    BeanAmplitudes = BeanAmplitudeSum / ScanGroups
    NormalizedBeanAmplitudes = NormalizeAmplitude(BeanAmplitudes)
    BeanMaxIndex = np.argmax(NormalizedBeanAmplitudes)
    BeanMaxNormalized = NormalizedBeanAmplitudes[BeanMaxIndex]

    print(f"  BeanMaxNormalized for iteration {i+1}: {BeanMaxNormalized}")
    BeanMaxNormalizedList.append(BeanMaxNormalized)

# Close sensor session
client.close()

# Save results to CSV
csv_filename = "BeanMaxNormalized_Output.csv"
file_exists = os.path.isfile(csv_filename)

with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Write header only if the file is new
    if not file_exists:
        writer.writerow(["RunID", "Iteration", "BeanMaxNormalized"])

    for idx, value in enumerate(BeanMaxNormalizedList, 1):
        writer.writerow([time.strftime("%Y%m%d-%H%M%S"), idx, value])

print(f"\nAppended {len(BeanMaxNormalizedList)} BeanMaxNormalized values to {csv_filename}.")

