import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Fcn(nn.Module):
    def __init__(self, in_channels):
        super(Fcn, self).__init__()
        self.fcn = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, inputs):
        return self.fcn(inputs)
    
class LstmFcn(nn.Module):
    def __init__(self, input_size, in_channels, units=64):
        super(LstmFcn, self).__init__()
        self.units = units
        self.fcn = Fcn(in_channels)  # Assuming Fcn is a custom PyTorch module
        self.lstm = nn.LSTM(input_size, self.units)

    def forward(self, inputs):
        lstm_out, _ = self.lstm(inputs)
        lstm_out = nn.Dropout(0.8)(lstm_out)
        fcn_out = self.fcn(inputs)
        print(fcn_out.shape)
        print(lstm_out.shape)
        return torch.cat([lstm_out, fcn_out], dim=1)