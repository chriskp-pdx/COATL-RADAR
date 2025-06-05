import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV
df = pd.read_csv('Data for Roast Comparison.csv')

# Extract X-axis
x = np.arange(1, 51)

# Helper to extract row values
def extract_row(name):
    return df[df['Bean Data Type'] == name].iloc[0, 1:].astype(float).values

# Dark Roast
dark_real = extract_row("DarkRoastMeanReal")
dark_real_std = extract_row("DarkRoastStdReal")
dark_imag = extract_row("DarkRoastMeanImaginary")
dark_imag_std = extract_row("DarkRoastStdImaginary")

# Medium Roast
med_real = extract_row("MedRoastMeanReal")
med_real_std = extract_row("MedRoastStdReal")
med_imag = extract_row("MedRoastMeanImaginary")
med_imag_std = extract_row("MedRoastStdImaginary")

# Light Roast
light_real = extract_row("LightRoastMeanReal")
light_real_std = extract_row("LightRoastStdReal")
light_imag = extract_row("LightRoastMeanImaginary")
light_imag_std = extract_row("LightRoastStdImaginary")

# Plot Real Components
plt.figure(figsize=(12, 6))
plt.plot(x, dark_real, 'r-', label='Dark Roast Real')
plt.fill_between(x, dark_real - dark_real_std, dark_real + dark_real_std, color='r', alpha=0.3)
plt.plot(x, med_real, 'g-', label='Medium Roast Real')
plt.fill_between(x, med_real - med_real_std, med_real + med_real_std, color='g', alpha=0.3)
plt.plot(x, light_real, 'b-', label='Light Roast Real')
plt.fill_between(x, light_real - light_real_std, light_real + light_real_std, color='b', alpha=0.3)
plt.title('Real Amplitude Comparison')
plt.xlabel('Point')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# Plot Imaginary Components
plt.figure(figsize=(12, 6))
plt.plot(x, dark_imag, 'r-', label='Dark Roast Imag')
plt.fill_between(x, dark_imag - dark_imag_std, dark_imag + dark_imag_std, color='r', alpha=0.3)
plt.plot(x, med_imag, 'g-', label='Medium Roast Imag')
plt.fill_between(x, med_imag - med_imag_std, med_imag + med_imag_std, color='g', alpha=0.3)
plt.plot(x, light_imag, 'b-', label='Light Roast Imag')
plt.fill_between(x, light_imag - light_imag_std, light_imag + light_imag_std, color='b', alpha=0.3)
plt.title('Imaginary Amplitude Comparison')
plt.xlabel('Point')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
