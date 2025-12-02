import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        """
        preds: (batch_size, num_quantiles)
        target: (batch_size) or (batch_size, 1)
        """
        assert preds.shape[1] == len(self.quantiles), "Preds dim 1 must match number of quantiles"
        
        loss = 0
        target = target.view(-1, 1) # Ensure target is (batch, 1)
        
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i:i+1]
            loss += torch.max((q-1) * errors, q * errors)
            
        return loss.mean()
