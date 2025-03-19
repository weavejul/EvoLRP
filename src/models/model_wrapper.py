import torch
import torch.nn as nn

class ModelWrapper(nn.Module):
    def __init__(self, base_model):
        """
        Wrapper for classification models to standardize forward pass.
        """
        super().__init__()
        self.model = base_model

    def forward(self, x):
        return self.model(x)