# Chris Kane-Pardy ECE 413 COATL RADAR 3/29/2025

# Import Acconeer Exploration Tool as Library & Other Libraries
import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import time
import csv
import os

# Function to collect and average multiple scans
def MultiScanAverage(client, sensor_config, Scans, Delay=0.0005):
    total_amplitudes = np.zeros(sensor_config.num_points, dtype=np.complex128)

    for _ in range(Scans):
        client.start_session()
        data = client.get_next()
        client.stop_session()

        amplitudes = np.array(data.frame[0].tolist())
        total_amplitudes += amplitudes
        
        time.sleep(Delay)
    
    return total_amplitudes / Scans

#Setup Sensor Client
client = a121.Client.open(serial_port="COM5") #Change COM Port to Match Your Local "Enhanced" Port

# Define sensor configuration
sensor_config = a121.SensorConfig()
sensor_config.step_length = 1
sensor_config.start_point = 40   
sensor_config.num_points = 50
sensor_config.sweeps_per_frame = 1
sensor_config.hwaas = 500
sensor_config.profile = et.a121.Profile.PROFILE_1
sensor_config.prf = 19.5e6
sensor_config.receiver_gain = 19
sensor_config.phase_enhancement = True

#Collect Calibration Scan (10 Empty Container Scans Averaged)
client.setup_session(sensor_config)
print("Starting calibration scan...")
CalibrationAmplitudes = MultiScanAverage(client, sensor_config, Scans = 50)
print("Calibration scan complete.")

#Wait for User to Input the Beans and Proceed
input("Press Enter to proceed to actual scans...")

#Collect 5 groups of bean scans, each averaging 10 scans
ScanGroups = 10
BeanAmplitudeSum = np.zeros(sensor_config.num_points, dtype=np.complex128)

RealBeanArray = []
ImaginaryBeanArray = []

for i in range(ScanGroups):
    print(f"Starting bean scan group {i+1}/{ScanGroups}...")
    
    client.setup_session(sensor_config)
    BeanData = MultiScanAverage(client, sensor_config, Scans = 50)
    BeanAmplitudeSum += BeanData
   
    ComplexBeanArray = CalibrationAmplitudes - BeanData
    RealBeanArray.append(np.real(ComplexBeanArray))
    ImaginaryBeanArray.append(np.imag(ComplexBeanArray))
    
    print(f"Bean scan group {i+1} complete.")

    #Require User Input Before Moving to the Next Group
    if i < ScanGroups - 1:
        input("Press Enter to continue to the next scan group...")

#Take Final Average of the Amplitude Scans
BeanAmplitudes = BeanAmplitudeSum / ScanGroups
FinalBeanAverage = CalibrationAmplitudes - BeanAmplitudes
RealBeanArray.append(np.real(FinalBeanAverage))
ImaginaryBeanArray.append(np.imag(FinalBeanAverage))

#Close the Client Session
client.close()

#Output Results
header = [f"Point_{i}" for i in range(sensor_config.num_points)]

import os

def SaveCombinedRowwiseCSV(filename, real_groups, imag_groups, labelPrefix, meanLabel):
    real_matrix = np.array(real_groups)
    imag_matrix = np.array(imag_groups)

    file_exists = os.path.exists(filename)
    file_has_data = os.path.getsize(filename) > 0 if file_exists else False

    extended_header = (
        ["Scan_Group"]
        + [f"Point_{i}_Real" for i in range(sensor_config.num_points)]
        + [f"Point_{i}_Imag" for i in range(sensor_config.num_points)]
    )

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_has_data:
            writer.writerow(extended_header)

        for i in range(len(real_groups)):
            label = f"{labelPrefix}_{i+1}" if i < ScanGroups else f"{meanLabel}"
            row = [label] + real_groups[i].tolist() + imag_groups[i].tolist()
            writer.writerow(row)

SaveCombinedRowwiseCSV("CombinedBeanData.csv", RealBeanArray, ImaginaryBeanArray, "Scan", "FinalAvg")
