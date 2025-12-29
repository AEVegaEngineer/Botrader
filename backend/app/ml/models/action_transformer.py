import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ActionTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, num_classes=3, dropout=0.1, max_len=5000):
        super(ActionTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """
        Args:
            src: [batch_size, seq_len, input_dim]
        Returns:
            output: [batch_size, num_classes]
        """
        # Permute for Transformer: [seq_len, batch_size, input_dim]
        src = src.permute(1, 0, 2)
        
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        
        # We take the output of the last time step
        # output shape: [seq_len, batch_size, d_model]
        output = output[-1, :, :]
        
        output = self.decoder(output)
        return output
