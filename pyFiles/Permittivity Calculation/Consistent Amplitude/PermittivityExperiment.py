#Chris Kane-Pardy ECE 413 COATL RADAR 3/29/2025

#Import Acconeeer Exploration Tool as Library & Other Libraries
import acconeer.exptool as et
from acconeer.exptool import a121
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.constants as sp

#Constant Definitions
epsilonnot = sp.epsilon_0  # Vacuum permittivity
c = sp.c  # Speed of light (m/s)
f = 60 * (10 ** 9)  # Frequency of XM125 (60 GHz)
w = 2 * np.pi * f  # Angular frequency
d = 0.0125  # Distance (in meters)

def CalculatePermittivity(CalibratedAmp, CalibratedPhase, BeanAmp, BeanPhase):
    #This is possibly necessary if we're getting Gamma Values over 1 and need to normalize them: GammaMagnitude = 10 * np.log10(BeanAmp) - 10 * np.log10(CalibratedAmp)
    GammaMagnitude = BeanAmp / CalibratedAmp
    GammaPhase = BeanPhase - CalibratedPhase
    Gamma = GammaMagnitude * np.exp(1j * GammaPhase)
    #This is possibly necessary, will require testing: Gamma = np.clip(Gamma, -1, 1)
    EpsilonR = ((1 + Gamma) / (1 - Gamma)) ** 2
    return EpsilonR

# Function to collect and average multiple scans
def MultiScanAverage(client, sensor_config, Scans, Delay=0.25):
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

#Device Calibration, This Can be Configured as Desired, but is Currently Setup for the Beanis
#Setup Sensor Client
client = a121.Client.open(serial_port="COM6") #Change COM Port to Match Your Local "Enhanced" Port

#Define sensor configuration
sensor_config = a121.SensorConfig()
sensor_config.step_length = 1
sensor_config.start_point = 40
sensor_config.num_points = 50
sensor_config.sweeps_per_frame = 1
sensor_config.hwaas = 60
sensor_config.profile = et.a121.Profile.PROFILE_1
sensor_config.prf = 19.5e6
sensor_config.receiver_gain = 23

#Collect Calibration Scan (10 Empty Container Scans Averaged)
client.setup_session(sensor_config)
print("Starting calibration scan...")
CalibrationAmplitudes = MultiScanAverage(client, sensor_config, Scans = 50)
print("Calibration scan complete.")

#Find the Maximum Calibration Amplitude and Phase
CalibrationMaxIndex = np.argmax(CalibrationAmplitudes)
CalibrationMaxAmplitude = np.abs(CalibrationAmplitudes[CalibrationMaxIndex])
CalibrationMaxPhase = np.angle(CalibrationAmplitudes[CalibrationMaxIndex])

#Wait for User to Input the Beans and Proceed
input("Press Enter to proceed to actual scans...")

#Collect 5 groups of bean scans, each averaging 10 scans
ScanGroups = 5
BeanAmplitudeSum = np.zeros(sensor_config.num_points, dtype=np.complex128)

for i in range(ScanGroups):
    print(f"Starting bean scan group {i+1}/{ScanGroups}...")
    
    client.setup_session(sensor_config)
    BeanData = MultiScanAverage(client, sensor_config, Scans = 10)
    BeanAmplitudeSum += BeanData
    print(f"Bean scan group {i+1} complete.")

    #Require User Input Before Moving to the Next Group
    if i < ScanGroups - 1:
        input("Press Enter to continue to the next scan group...")

#Take Final Average of the Amplitude Scans
BeanAmplitudes = BeanAmplitudeSum / ScanGroups

#Find the Maximum Average Bean Amplitude and Phase
BeanMaxIndex = np.argmax(BeanAmplitudes)
BeanMaxAmplitude = np.abs(BeanAmplitudes[BeanMaxIndex])
BeanMaxPhase = np.angle(BeanAmplitudes[BeanMaxIndex])

print(f"Average Maximum Calibration Amplitude: {CalibrationMaxAmplitude}")
print(f"Average Maximum Calibration Phase: {CalibrationMaxPhase}")
print(f"Average Maximum Bean Amplitude: {BeanMaxAmplitude}")
print(f"Average Maximum Bean Phase: {BeanMaxPhase}")

#Calculate Dielectric Constant
DielectricConstant = np.abs(CalculatePermittivity(CalibrationMaxAmplitude, CalibrationMaxPhase, BeanMaxAmplitude, BeanMaxPhase))
print(f"Relative Permittivity: {DielectricConstant}")

#Close the Client Session
client.close()