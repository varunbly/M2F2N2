import torch
import torch.nn as nn
from .base import BaseFusionStrategy

class GatedFusion(BaseFusionStrategy):
    """
    Gated Cross-Modal Fusion from M2VN (§3.2)
    a = σ(W_g[r;t]), fused = a·r + (1-a)·t.
    """
    def __init__(self, d):
        super().__init__()
        self.gate_linear = nn.Linear(2 * d, 1)
        self.fusion_proj = nn.Linear(3 * d, d)

    def forward(self, hp, hn, ht):
        gate = torch.sigmoid(self.gate_linear(torch.cat([hp, hn], dim=-1)))  # (B,T,1)
        fused = gate * hp + (1 - gate) * hn
        bilin = hp * hn
        absdiff = torch.abs(hp - hn)
        h = self.fusion_proj(torch.cat([fused, bilin, absdiff], dim=-1))
        return h + ht
