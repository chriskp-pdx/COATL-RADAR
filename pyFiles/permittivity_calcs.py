# Example program that takes in I/Q data from the XM125 (both the sample and reference)
# Computes phase shift (Δ𝜙) and attenuation (𝛼)
# Calculates permittivity: 
# 𝜀′(real part) from phase velocity 
# 𝜀′′(imaginary part) from attenuation
# Plots the results


import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8  # Speed of light (m/s)
f = 60e9  # Frequency of XM125 (Hz)
w = 2 * np.pi * f  # Angular frequency

def process_xm125_data(iq_data, ref_iq_data, d):
    """
    Compute permittivity (epsilon' and epsilon'') from XM125 I/Q data.
    :param iq_data: Complex array of I/Q data (measurement with coffee beans)
    :param ref_iq_data: Complex array of I/Q data (free-space reference)
    :param d: Thickness of coffee bean sample (meters)
    :return: epsilon' and epsilon''
    """
    # Compute phase shift
    phase_meas = np.angle(iq_data)
    phase_ref = np.angle(ref_iq_data)
    delta_phi = np.unwrap(phase_meas - phase_ref)  # Phase difference
    beta = delta_phi / d  # Phase constant
    
    # Compute attenuation
    amp_meas = np.abs(iq_data)
    amp_ref = np.abs(ref_iq_data)
    alpha = np.log(amp_ref / amp_meas) / d  # Attenuation constant
    
    # Compute permittivity
    epsilon_real = (beta * c / w) ** 2
    epsilon_imag = (2 * alpha * c) / w
    
    return epsilon_real, epsilon_imag

# Example data (replace with real XM125 data)
num_samples = 100
iq_data = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)  # Simulated XM125 data
ref_iq_data = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)  # Simulated reference
d = 0.01  # Thickness of coffee bean sample (10 mm)

# Compute permittivity
epsilon_real, epsilon_imag = process_xm125_data(iq_data, ref_iq_data, d)

# Plot results
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epsilon_real, label="Real Permittivity (ε')")
plt.xlabel("Sample Index")
plt.ylabel("ε'")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epsilon_imag, label="Imaginary Permittivity (ε'')", color='r')
plt.xlabel("Sample Index")
plt.ylabel("ε''")
plt.legend()

plt.tight_layout()
plt.show()