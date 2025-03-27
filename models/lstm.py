'''
Description: 
version: 
Author: tangshiyi
Date: 2025-03-27 11:26:30
LastEditors: tangshiyi
LastEditTime: 2025-03-27 12:15:44
'''
import torch
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Correct hidden state initialization
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Hidden state
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)  # Cell state

        # Properly calling LSTM without extra arguments
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))

        # Extract last timestep's output for classification
        last_timestep_output = output[:, -1, :]  # Shape: (batch_size, hidden_size)
        return self.fc(last_timestep_output)  # Shape: (batch_size, num_classes)
    