'''
Description: 
version: 
Author: tangshiyi
Date: 2025-02-20 17:10:46
LastEditors: tangshiyi
LastEditTime: 2025-03-27 12:01:18
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import os

# Read data
folder_path = "sensor_data_320/"
files = os.listdir(folder_path)

for file in files:
    file_name = os.path.join(folder_path, file)
    # file_name = 'realtime_data.csv'
    df = pd.read_csv(file_name)
    # Add timestep as index (0, 1, 2, ...)
    df["timestep"] = np.arange(len(df))

    # Extract peaks from ax column
    peaks, _ = find_peaks(df['ax'], height=1200, distance=12)

    # # Ensure the first peak is greater than 2000
    # if len(peaks) > 0 and df['ax'].iloc[peaks[0]] < 2000:
    #     peaks = peaks[1:]

    print(f'Real Peak indexes: {peaks[::2]}')

    # Replace peak_times with timestep
    peak_times = df.iloc[peaks]['timestep'].values
    peak_values = df.iloc[peaks]['ax'].values
    print(f'Number of peaks is {len(peak_values)} in {file_name}')

    # Plot peak detection graph
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['timestep'], df['ax'], label='ax', color='b', alpha=0.6)
    ax.scatter(peak_times, peak_values, color='r', marker='o', label="Peaks")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Acceleration (ax)")
    ax.set_title(f"Peak Detection in {file_name}")
    ax.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


    # Plot subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # 1. IMU data (acceleration + gyroscope)
    axes[0].plot(df['timestep'], df['ax'], label='ax', color='b', alpha=0.6)
    axes[0].plot(df['timestep'], df['ay'], label='ay', color='g', alpha=0.6)
    axes[0].plot(df['timestep'], df['az'], label='az', color='r', alpha=0.6)
    axes[0].set_ylabel("IMU Acceleration")
    axes[0].set_title("IMU Data")
    axes[0].legend()

    # 2. Flex sensor data
    for i in range(1, 6):
        axes[1].plot(df['timestep'], df[f'flex{i}'], label=f'flex{i}')
    axes[1].set_ylabel("Flex Sensor")
    axes[1].set_title("Flex Sensor Data")
    axes[1].legend()

    # Mark `peak_times` (black dashed lines) on IMU & Flex plots
    for i, peak_time in enumerate(peak_times):
        linewidth = 2 if i % 2 == 0 else 1  # Thicker line for even indices
        for ax in axes:
            ax.axvline(peak_time, color='k', linestyle='--', alpha=0.5, linewidth=linewidth)

        # Label `timestep` on x-axis
        axes[1].text(peak_time, axes[1].get_ylim()[0], str(peak_time),
                    fontsize=8, rotation=45, ha='right', color='black')

    # General settings
    plt.xlabel("Timestep")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()