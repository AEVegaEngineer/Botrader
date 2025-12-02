import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class CNNClassifier(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim, kernel_size=3):
        super(CNNClassifier, self).__init__()
        
        # Input shape for Conv1d: (batch, input_dim, seq_len)
        # We need to transpose input in forward pass
        
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1) # Global Max Pooling
        self.fc = nn.Linear(num_filters, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        # Transpose to (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten: (batch, num_filters, 1) -> (batch, num_filters)
        x = x.squeeze(2)
        
        x = self.fc(x)
        return x
