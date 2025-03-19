# lrp/base.py
from abc import ABC, abstractmethod
import torch.nn as nn

class LRP(ABC):
    def __init__(self, model: nn.Module):
        """
        Base LRP class. Each LRP method should implement the `compute_relevance` method.
        """
        self.model = model

    @abstractmethod
    def compute_relevance(self, input_tensor):
        """
        Compute LRP relevance scores for a given input.
        """
        pass