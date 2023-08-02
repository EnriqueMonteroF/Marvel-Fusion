import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import allantools

# Read CSV file, skipping the first 57 rows
data = pd.read_csv('C://Users//RFRUBEN//OneDrive - Teknologian Tutkimuskeskus VTT//Documents//csv//15.05.2023.csv', delimiter='\t', skiprows=57) # Green laser
# data1 = pd.read_csv('C://Users//RFRUBEN//OneDrive - Teknologian Tutkimuskeskus VTT//Documents//csv//24.07.2023, 10.02,  435,322074 THz.csv', delimiter='\t', skiprows=57) # Red laser
# data = pd.read_csv('C://Users//RFRUBEN//OneDrive - Teknologian Tutkimuskeskus VTT//Documents//csv//20.06.2023, 10.56,  15 798,00760 cm-1.csv', delimiter='\t', skiprows=57) # He-Ne laser

# Replace commas with decimal points
data.replace(',', '.', regex=True, inplace=True)

# Convert cell arrays to numeric arrays
wavelength = data.iloc[:, 1].astype(float).to_numpy()
time = data.iloc[:, 0].astype(float).to_numpy() / 1000  # Convert time to seconds

# Prompt the user to input the desired time range in seconds
start_time = float(input("Enter the start time (in seconds): "))
end_time = float(input("Enter the end time (in seconds): "))

# Filter data within the specified time range
time_mask = (time >= start_time) & (time <= end_time)
wavelength_filtered = wavelength[time_mask]
time_filtered = time[time_mask]

# Calculate the lower and upper bounds using IQR method on the filtered data
Q1 = np.percentile(wavelength_filtered, 25)
Q3 = np.percentile(wavelength_filtered, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
outlier_mask = (wavelength_filtered >= lower_bound) & (wavelength_filtered <= upper_bound)
wavelength_filtered = wavelength_filtered[outlier_mask]
time_filtered = time_filtered[outlier_mask]

# Calculate the average of the filtered wavelength data
wavelength_average = np.mean(wavelength_filtered)
# print(wavelength_average)

wavelength_nor = wavelength_filtered / wavelength_average
# print(wavelength_nor)

# Calculate the time interval between consecutive data points
time_interval = np.diff(time_filtered)

# Calculate the average sampling rate
average_rate = 1 / np.mean(time_interval)
# print(average_rate)

plt.plot(time_filtered, f)
plt.xlabel('Time (seconds)')
plt.ylabel('Wavelength Filtered')
plt.title('Wavelength Filtered vs Time')
plt.grid(True)
plt.show()

# Calculate the Allan deviation
taus, adevs, _, _ = allantools.adev(wavelength_nor, data_type="freq", rate=average_rate)

# Find the minimum Allan deviation and its corresponding tau value
min_adev = np.min(adevs)
min_adev_tau = taus[np.argmin(adevs)]

# Calculate the standard deviation of the wavelength_filtered data
wavelength_std = np.std(wavelength_filtered)

# Find the maximum Allan deviation and its corresponding tau value
max_adev = np.max(adevs)
max_adev_tau = taus[np.argmax(adevs)]

print(f'Standard Deviation of Wavelength Filtered: {wavelength_std:.2e}')
print(f'Highest Allan Deviation: {max_adev:.2e} at tau = {max_adev_tau:.2e}')
print(f'Minimum Allan Deviation: {min_adev:.2e} at tau = {min_adev_tau:.2e}')

# Plot the Allan deviation
fig, ax = plt.subplots()
ax.loglog(taus, adevs)
ax.set_xlabel('Averaging Time (s)')
ax.set_ylabel('Allan Deviation')
ax.set_title('Frequency Stability')
ax.grid(True)

# Set the x-axis tick label formatter to ScalarFormatter
ax.xaxis.set_major_formatter(ScalarFormatter())

# Annotate the lowest point on the plot
ax.annotate(f'Minimum Allan Deviation: {min_adev:.2e}', xy=(min_adev_tau, min_adev), xycoords='data',
            xytext=(0.5, 0.8), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->", lw=1.5, color='black'),
            fontsize=12, color='black')

plt.show()

# Plot the square root of the Allan deviation
fig, ax = plt.subplots()
ax.loglog(taus, np.sqrt(adevs))  # Taking the square root of adevs before plotting
ax.set_xlabel('Averaging Time (s)')
ax.set_ylabel('Square Root of Allan Deviation')
ax.set_title('Frequency Stability')
ax.grid(True)

# Set the x-axis tick label formatter to ScalarFormatter
ax.xaxis.set_major_formatter(ScalarFormatter())

# Annotate the lowest point on the plot
ax.annotate(f'Minimum Allan Deviation: {min_adev:.2e}', xy=(min_adev_tau, np.sqrt(min_adev)), xycoords='data',
            xytext=(0.5, 0.8), textcoords='axes fraction',
            arrowprops=dict(arrowstyle="->", lw=1.5, color='black'),
            fontsize=12, color='black')

plt.show()