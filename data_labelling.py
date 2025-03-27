import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks

# Sliding window parameters
window_size = 15  # 
step_size = 1    

# 24 gesture transitions
# Rock: 0  Paper: 1  Scissor: 2  Up: 3  Down: 4  Left: 5  Right: 6
gesture_transitions = ["0→3", "3→1", "1→3", "3→2", "2→3", "3→0", 
                       "0→4", "4→1", "1→4", "4→2", "2→4", "4→0",
                       "0→5", "5→1", "1→5", "5→2", "2→5", "5→0",
                       "0→6", "6→1", "1→6", "6→2", "2→6", "6→0"] 


# Read data folder
folder_path = "sensor_data_320_testnew/"
files = os.listdir(folder_path)

dataset = []  # Store the final dataset
sample_id = 0

for file in files:
    file_name = os.path.join(folder_path, file)
    df = pd.read_csv(file_name)

    # Extract peaks from ax column
    peaks, _ = find_peaks(df['ax'], height=1200, distance=12)  # Set height threshold, can be adjusted
    # # Ensure the first peak is greater than 2000
    # if len(peaks) > 0 and df['ax'].iloc[peaks[0]] < 2000:
    #     peaks = peaks[1:]

    print('================================================================================')
    print(f"File: {file_name}, Detected Beats: {peaks}")


    # 24-class gesture transitions
    for i in range(1, len(peaks) - 2, 2):
        cur_beat_idx = peaks[i]
        next_beat_idx = peaks[i + 1]
        label_idx = (i - 1) // 2  # Adjust index
        label = gesture_transitions[label_idx]  # Get corresponding gesture label
        print('--------------------------------------------------')
        print(f'cur_beat_idx is {cur_beat_idx}')
        print(f'label is {label}')
        # print(f'window start between {cur_beat_idx} and {next_beat_idx - window_size}')
        # print(f'number of samples: {next_beat_idx - window_size - cur_beat_idx + 1}')
        
        # Use sliding window sampling between current beat-to-beat
        win_start = cur_beat_idx
        while win_start + window_size < next_beat_idx:
            win_end = win_start + window_size
            window_data = df.iloc[win_start:win_end]

            # Format transformation: each sensor’s data becomes "val1 val2 ..." string
            series_data = []
            for col in df.columns[1:12]:  # Skip timestamp
                series_data.append(" ".join(map(str, window_data[col].values)))  # Convert to string format
            
            # Save to dataset
            dataset.append([sample_id] + series_data + [label])  
            sample_id += 1  # Increment sample ID

            win_start += step_size



    # 'other' class: Rock, Paper, Scissors
    for i in range(0, 9, 4):
        cur_beat_idx = peaks[i]
        next_beat_idx = peaks[i + 1] if i + 1 < len(peaks) else len(df)  # Next beat or end of data
        if i == 0:
            label = "0→0"
        elif i == 4:
            label = "1→1"
        else:
            label = "2→2"
        # Use sliding window sampling between current beat-to-beat
        win_start = cur_beat_idx
        while win_start + window_size < next_beat_idx:
            win_end = win_start + window_size
            window_data = df.iloc[win_start:win_end]

            # Format transformation: each sensor’s data becomes "val1 val2 ..." string
            series_data = []
            for col in df.columns[1:12]:  # Skip timestamp
                series_data.append(" ".join(map(str, window_data[col].values)))  # Convert to string format
            
            # Save to dataset
            dataset.append([sample_id] + series_data + [label])  
            sample_id += 1  # Increment sample ID

            win_start += step_size

    # 'other' class: Up, Down, Left, Right
    for i in range(2, 14, 12):
        cur_beat_idx = peaks[i]
        next_beat_idx = peaks[i + 1] if i + 1 < len(peaks) else len(df)  # Next beat or end of data
        if i == 2:
            label = "3→3"
        elif i == 14:
            label = "4→4"
        elif i == 26:
            label = "5→5"
        else:
            label = "6→6"
        # Use sliding window sampling between current beat-to-beat
        win_start = cur_beat_idx
        while win_start + window_size < next_beat_idx:
            win_end = win_start + window_size
            window_data = df.iloc[win_start:win_end]

            # Format transformation: each sensor’s data becomes "val1 val2 ..." string
            series_data = []
            for col in df.columns[1:12]:  # Skip timestamp
                series_data.append(" ".join(map(str, window_data[col].values)))  # Convert to string format
            
            # Save to dataset
            dataset.append([sample_id] + series_data + [label])  
            sample_id += 1  # Increment sample ID

            win_start += step_size


# Generate column names
columns = ["Sample_ID"] + list(df.columns[1:12]) + ["Label"]  # Skip timestamp

# Convert to DataFrame and save
dataset_df = pd.DataFrame(dataset, columns=columns)
dataset_df.to_csv("gesture_dataset_320_testnew.csv", index=False)

print("Dataset saved: gesture_dataset_320_testnew.csv")
