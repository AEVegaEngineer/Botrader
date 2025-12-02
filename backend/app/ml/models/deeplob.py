import torch
import torch.nn as nn

class DeepLOB(nn.Module):
    def __init__(self, seq_len, num_features, num_classes=3):
        super(DeepLOB, self).__init__()
        
        # num_features = 20 (5 levels * 2 sides * 2 vars)
        # We can treat this as a 1D image with 20 channels if we want, or just 1D time series.
        # DeepLOB paper uses 2D convs over (Time, Levels).
        # Here we implement a simplified 1D CNN + LSTM version.
        
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate output size after pooling
        # seq_len -> seq_len/2 -> seq_len/4 -> seq_len/8
        final_seq_len = seq_len // 8
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        # Transpose for Conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # x: (batch, 64, final_seq_len)
        # Transpose for LSTM: (batch, final_seq_len, 64)
        x = x.transpose(1, 2)
        
        out, _ = self.lstm(x)
        
        # Take last time step
        out = out[:, -1, :]
        
        out = self.fc(out)
        return out
