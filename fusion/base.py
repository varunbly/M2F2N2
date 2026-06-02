import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseFusionStrategy(nn.Module, ABC):
    @abstractmethod
    def forward(self, hp, hn, ht):
        """
        Fuse price, news, and time embeddings.
        Args:
            hp: Price embedding (B, T, d)
            hn: News embedding (B, T, d)
            ht: Time embedding (B, T, d)
        Returns:
            fused embedding (B, T, d)
        """
        pass
