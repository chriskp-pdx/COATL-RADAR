import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read CSV
df = pd.read_csv(r'C:\Users\Chris\Desktop\Spring 2025\Git4School\COATL-RADAR\Software\Data Collection\Datasets\Comparison Graph Data.csv')

# Filter real and imaginary values
yemenRealAmp = df["Average"].iloc[0:49]
yemenRealStd = df["Std Deviation"].iloc[0:49]
yemenImagAmp = df["Average"].iloc[50:99]
yemenImagStd = df["Std Deviation"].iloc[50:99]

guatemalaRealAmp = df["Average"].iloc[100:149]
guatemalaRealStd = df["Std Deviation"].iloc[100:149]
guatemalaImagAmp = df["Average"].iloc[150:199]
guatemalaImagStd = df["Std Deviation"].iloc[150:199]

rwandaRealAmp = df["Average"].iloc[200:249]
rwandaRealStd = df["Std Deviation"].iloc[200:249]
rwandaImagAmp = df["Average"].iloc[250:299]
rwandaImagStd = df["Std Deviation"].iloc[250:299]

# Create x-axis points
x = np.arange(0, 49)

# Plot Real Values (Yemen, Guatemala, Rwanda)
plt.figure(figsize=(12, 6))
plt.plot(x, yemenRealAmp, color='blue', linewidth=2, label='Yemen Real')
plt.fill_between(x, yemenRealAmp - yemenRealStd, yemenRealAmp + yemenRealStd, color='blue', alpha=0.4, label='Yemen Stdev')
plt.plot(x, guatemalaRealAmp, color='green', linewidth=2, label='Guatemala Real')
plt.fill_between(x, guatemalaRealAmp - guatemalaRealStd, guatemalaRealAmp + guatemalaRealStd, color='green', alpha=0.4, label='Guatemala Stdev')
plt.plot(x, rwandaRealAmp, color='red', linewidth=2, label='Rwanda Real')
plt.fill_between(x, rwandaRealAmp - rwandaRealStd, rwandaRealAmp + rwandaRealStd, color='red', alpha=0.4, label='Rwanda Stdev')
plt.xlabel('Point')
plt.ylabel('Amplitude')
plt.title('Mean Real Amplitude Values Across Radar Sweep')
plt.ylim(-50000, 50000)
plt.yticks(np.arange(-50000, 50001, 10000))
plt.legend()
plt.grid(True)
plt.show()

# Plot Imaginary Values (Yemen, Guatemala, Rwanda)
plt.figure(figsize=(12, 6))
plt.plot(x, yemenImagAmp, color='blue', linewidth=2, label='Yemen Imaginary')
plt.fill_between(x, yemenImagAmp - yemenImagStd, yemenImagAmp + yemenImagStd, color='blue', alpha=0.4, label='Yemen Stdev')
plt.plot(x, guatemalaImagAmp, color='green', linewidth=2, label='Guatemala Imaginary')
plt.fill_between(x, guatemalaImagAmp - guatemalaImagStd, guatemalaImagAmp + guatemalaImagStd, color='green', alpha=0.4, label='Guatemala Stdev')
plt.plot(x, rwandaImagAmp, color='red', linewidth=2, label='Rwanda Imaginary')
plt.fill_between(x, rwandaImagAmp - rwandaImagStd, rwandaImagAmp + rwandaImagStd, color='red', alpha=0.4, label='Rwanda Stdev')
plt.xlabel('Point')
plt.ylabel('Amplitude')
plt.title('Mean Imaginary Amplitude Values Across Radar Sweep')
plt.ylim(-50000, 50000)
plt.yticks(np.arange(-50000, 50001, 10000))
plt.legend()
plt.grid(True)
plt.show()