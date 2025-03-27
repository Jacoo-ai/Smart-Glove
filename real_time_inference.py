'''
Description: 
version: 
Author: tangshiyi
Date: 2025-03-16 18:32:23
LastEditors: tangshiyi
LastEditTime: 2025-03-27 12:15:05
'''
from collections import deque
import csv
import queue
import threading
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.signal import medfilt, find_peaks
import serial
from sklearn.preprocessing import MinMaxScaler
import torch
from models.lstm import LSTMClassifier



# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 11
num_classes = 24
model = LSTMClassifier(input_size=11, hidden_size=64, num_layers=3, num_classes=num_classes, dropout=0.3).to(device)
model.load_state_dict(torch.load("best_pretrain_24classes_0.9333.pth"))  # Pre-trained multi-class model
model.eval()

# Set parameters 
WINDOW_SIZE = 15  # Approx. 300ms of data
BEAT_THRESHOLD = 500  # Peak threshold
BEAT_COOLDOWN = 15  # No repeated detection within 50 data points
MIN_PEAK_DISTANCE = 15

# Maintain data window
sensor_window = deque(maxlen=WINDOW_SIZE)
timesteps = deque(maxlen=WINDOW_SIZE)
last_beat_time = 0
latest_prediction = None
prediction_queue = queue.Queue()  # Prediction queue

# Record results
detected_beats = 0
global save_result
save_result = False
return_class_list  = []

with open('prediction_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    header = [
        "transition", 
        "gesture", "true gesture"
    ]
    writer.writerow(header)

# Configure serial port parameters
port = 'COM8'    
baudrate = 115200  
ser = serial.Serial(
    port='COM8',  
    baudrate=115200,
    timeout=0,
    xonxoff=False,
    rtscts=False,
    dsrdtr=False
)
ser.flushInput()  # Clear input buffer
ser.reset_input_buffer()  # Clear buffer to avoid data piling
print("Connected to {}".format(ser.name))

def data_collection_and_beat_detection():
    global last_beat_time
    global save_result
    global detected_beats
    
    timestep = 0
    buffer = ""
    while True:
        if ser.in_waiting > 0:
            buffer += ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                data = line.strip().split(',')

                if len(data) == 11:
                    try:
                        # Parse data
                        timestep += 1
                        ax, ay, az, gx, gy, gz = map(float, data[:6])
                        flex_values = list(map(int, data[6:11]))
                        new_data = [timestep, ax, ay, az, gx, gy, gz] + flex_values
                        # print(f"read new data {new_data}")

                        # Store in sliding window
                        sensor_window.append(new_data[1:])  # Remove timestamp
                        timesteps.append(new_data[0])

                        # Get recent ax signal
                        accel_values = [d[0] for d in sensor_window]  # Take ax

                        # Detect peaks
                        peaks, _ = find_peaks(accel_values, height=BEAT_THRESHOLD, distance=MIN_PEAK_DISTANCE)

                        if len(peaks) > 0:
                            peak_idx = peaks[-1]
                            peak_time = timesteps[peak_idx]  # Get latest peak time
                            # Avoid duplicate detection
                            if peak_time - last_beat_time > BEAT_COOLDOWN:
                                detected_beats += 1
                                last_beat_time = peak_time
                                print(f"🎵 Detected Beat! Time: {peak_time}")

                                # Send to prediction queue
                                start_idx = max(0, peak_idx - WINDOW_SIZE)  # Prevent out-of-bounds
                                sensor_data_batch = list(sensor_window)[start_idx:peak_idx+1]  # Take data before and at peak
                                prediction_queue.put(sensor_data_batch)
                                save_result = True
                                print("📩 Sent data to do predition `predict_real_time`")
            
                    except ValueError as e:
                        print("data error: {}".format(e))
        time.sleep(0.0001)  # Reduce CPU usage


def predict_real_time(is_fake=False):
    global save_result
    if is_fake:
        fake_data = np.array(np.zeros((WINDOW_SIZE, 11)), dtype=np.float32)
        predict_single_sample(sensor_data_batch=fake_data, isFake=True)
    else:
        while True:
            if save_result and not prediction_queue.empty():
                sensor_data_batch = prediction_queue.get()  # Get new data for prediction
                predict_single_sample(sensor_data_batch=sensor_data_batch)
                save_result = False
            
            time.sleep(0.0001)  # Allow CPU to switch threads


def process_single_sample(sensor_data_batch):
    sample_array = np.array(sensor_data_batch)
    # Preprocessing
    for j in range(sample_array.shape[1]):  # Loop through features
        if j < 6:  # IMU data (ax, ay, az, gx, gy, gz) -> Moving average filter
            sample_array[:, j] = np.convolve(sample_array[:, j], np.ones(5) / 5, mode='same')
        elif j >= 6 and j < 11:  # Flex data -> Median filter
            sample_array[:, j] = medfilt(sample_array[:, j], kernel_size=3)
    sample_array = scaler.transform(sample_array)  # Normalization

    return sample_array  # Shape: (window_size, num_features)
 
def predict_single_sample(sensor_data_batch, isFake=False):
    global detected_beats
    processed_data = process_single_sample(sensor_data_batch=sensor_data_batch)  # sensor_data_batch: (window_size, num_features)
    # Convert to PyTorch tensor
    input_tensor = torch.tensor(processed_data, dtype=torch.float32).unsqueeze(0).to(device)
    # print(input_tensor.shape)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        latest_prediction = reverse_dict.get(predicted_class, "Unknown Gesture")
        if detected_beats % 2 == 1:
            print(f"🧠 Result: {latest_prediction}")
            print("================================================")

            if not isFake:
                # Send prediction to Dash UI
                url = 'http://127.0.0.1:8050/update-prediction'
                data = {'predicted_info': [predicted_class, detected_beats]}
                try:
                    response = requests.post(url, json=data)
                    # print(f"✅ Sent to UI: {data}, Response: {response.json()}")
                except Exception as e:
                    print(f"⚠️ Error sending prediction: {e}")
            
    return predicted_class


def train_scaler(df):
    feature_columns = df.columns[1:12]  # Exclude Sample_ID and Label
    X = []
    for _, row in df.iterrows():
        sample = [np.array(row[col].split(), dtype=np.float32) for col in feature_columns]
        X.append(np.stack(sample, axis=1))  # (40, num_features)

    X = np.array(X)  # Convert to (num_samples, 40, num_features)

    # Smoothing and denoising
    for i in range(X.shape[0]):  # Loop through samples
        for j in range(X.shape[2]):  # Loop through features
            if j < 6:  # IMU data -> Moving average
                X[i, :, j] = np.convolve(X[i, :, j], np.ones(5) / 5, mode='same')
            elif j >= 6 and j < 11:  # Flex data -> Median filter
                X[i, :, j] = medfilt(X[i, :, j], kernel_size=3)

    # Normalize data
    scaler = MinMaxScaler()
    X = X.reshape(-1, X.shape[-1])  # (num_samples*window_size, num_features)
    X = scaler.fit_transform(X)  # Normalize
    X = X.reshape(-1, 15, len(feature_columns))  # Reshape back
    return scaler


def get_train_data(data_path):
    train_df = pd.read_csv(data_path)
    df_all = train_df
    label_col = df_all.columns[-1]  # Get the last column name
    target_labels = ['0→0', '1→1','2→2','3→3','4→4','5→5','6→6','7→7']  # Change this to your specific labels
    # Split the DataFrame based on the label list
    df_others = df_all[df_all[label_col].isin(target_labels)]
    df_24 = df_all[~df_all[label_col].isin(target_labels)]
    # Shuffle each subset
    df_others = df_others.sample(frac=1, random_state=42).reset_index(drop=True)
    # Sample from each subset
    sample_size = 900  # Adjust as needed (or use frac=0.2 for 20% sampling)
    df_others = df_others.sample(n=min(sample_size, len(df_others)), random_state=42)
    label_col = df_others.columns[-1]
    # Replace all values in the last column with a single label (e.g., "new_label")
    new_label = "other"  # Change this to the desired label
    df_others[label_col] = new_label
    df_data = pd.concat([df_24,df_others])
    return df_data


# Create scaler and fit on full training set
df_data = get_train_data('gesture_dataset_317.csv')
scaler = train_scaler(df=df_data)
label_map = {'Rock→Up': 0, 'Rock→Down': 1, 'Rock→Left': 2, 'Rock→Right': 3, 'Paper→Up': 4, 'Paper→Down': 5, 'Paper→Left': 6, 'Paper→Right': 7, 'Scissor→Up': 8, 'Scissor→Down': 9, 'Scissor→Left': 10, 'Scissor→Right': 11, 'Up→Rock': 12, 'Up→Paper': 13, 'Up→Scissor': 14, 'Down→Rock': 15, 'Down→Paper': 16, 'Down→Scissor': 17, 'Left→Rock': 18, 'Left→Paper': 19, 'Left→Scissor': 20, 'Right→Rock': 21, 'Right→Paper': 22, 'Right→Scissor': 23, 'other': 24}
reverse_dict = {v: k for k, v in label_map.items()}

print("Fake data to predict")
predict_real_time(is_fake=True)
time.sleep(2)
print("start!")

# === Start multithreading ===
thread_1 = threading.Thread(target=data_collection_and_beat_detection, daemon=True)
thread_2 = threading.Thread(target=predict_real_time, daemon=True)

thread_1.start()
thread_2.start()

# Keep main thread alive
while True:
    time.sleep(1)
