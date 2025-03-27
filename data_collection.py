'''
Description: 
version: 
Author: tangshiyi
Date: 2025-01-27 14:37:08
LastEditors: tangshiyi
LastEditTime: 2025-03-27 11:53:34
'''
# -*- coding: utf-8 -*-
import serial
import csv
import time
from datetime import datetime

# Configure serial port parameters
port = 'COM8'
baudrate = 115200

# Create a CSV file and write the header
filename = 'sensor_data' + str(time.time()) + '.csv'

with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    header = [
        'timestamp',
        'ax', 'ay', 'az', 'gx', 'gy', 'gz',
        'flex1', 'flex2', 'flex3', 'flex4', 'flex5',
        'emg1', 'emg2', 'emg3'
    ]
    writer.writerow(header)

# Connect to the serial port
ser = serial.Serial(port, baudrate, timeout=1)
print("Connected to {}".format(ser.name))

try:
    while True:
        if ser.in_waiting > 0:
            # Read a line of data
            line = ser.readline().decode('utf-8',  errors='ignore').strip()
            data = line.split(',')
            
            # Check data length
            if len(data) == 11:
                try:
                    # Parse data
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    ax = float(data[0])
                    ay = float(data[1])
                    az = float(data[2])
                    gx = float(data[3])
                    gy = float(data[4])
                    gz = float(data[5])
                    flex_values = list(map(int, data[6:11]))
                    
                    # Write to CSV
                    with open(filename, 'a', newline='') as f:
                        writer = csv.writer(f)
                        row = [timestamp] + [ax, ay, az, gx, gy, gz] + flex_values
                        writer.writerow(row)
                    
                    print("Saved: {}".format(row))
                except ValueError as e:
                    print("data error: {}".format(e))
            else:
                print("data len not correct: {}/11".format(len(data)))
except KeyboardInterrupt:
    print("exit")
finally:
    ser.close()
    print("port closed")
