import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.config import NUM_ASSETS, NEWS_DIM, TIME_FEATS, LATENT_DIM, ALIGN_DIM, TEMPERATURE
from fusion.gated import GatedFusion

class M2VNLite(nn.Module):
    """
    Simplified M2VN for position prediction on numin2 data.

    Architecture (following the paper):
      ℰ_price : (B,T,150) → (B,T,d)     price encoder
      ℰ_news  : (B,T,384) → (B,T,d)     news encoder
      ℰ_time  : (B,T,4)   → (B,T,d)     time encoder
      GatedFusion             → (B,T,d)  gated cross-modal fusion
      TemporalAgg             → (B,d)    attention-pooled summary
      PositionHead            → (B,50)   tanh-bounded positions

    Losses:
      ℒ_pred  = MSE(positions, target_returns)
      ℒ_align = InfoNCE(h_price, h_news)       [auxiliary]
      ℒ_total = ℒ_pred + λ · ℒ_align
    """

    def __init__(self, num_assets=NUM_ASSETS, num_price_features=NUM_ASSETS*3, news_dim=NEWS_DIM,
                 time_dim=TIME_FEATS, d=LATENT_DIM,
                 d_a=ALIGN_DIM, tau=TEMPERATURE, fusion_strategy=None):
        super().__init__()
        self.d = d

        # --- Encoders ---
        self.price_encoder = nn.Sequential(
            nn.Linear(num_price_features, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.news_encoder = nn.Sequential(
            nn.Linear(news_dim, d), nn.LayerNorm(d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, d), nn.GELU(),
        )

        # --- Fusion Strategy ---
        if fusion_strategy is None:
            self.fusion = GatedFusion(d)
        else:
            self.fusion = fusion_strategy

        # --- Temporal aggregation  (1-D conv + attention pooling) ---
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d, d, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(d, d, kernel_size=3, padding=1),
        )
        self.attn_query = nn.Linear(d, 1)

        # --- Alignment projectors  (for InfoNCE) ---
        self.price_proj = nn.Linear(d, d_a)
        self.news_proj  = nn.Linear(d, d_a)
        self.log_tau    = nn.Parameter(torch.tensor(float(np.log(tau))))

        # --- Position Head  (ANIL adapts only this) ---
        self.position_head = nn.Sequential(
            nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d, num_assets),
            # nn.Tanh(),
            Asinh(),
        )

    # ────────────────────────────────────────────────
    def _encode(self, price, news, time_feats):
        return (self.price_encoder(price),      # (B,T,d)
                self.news_encoder(news),         # (B,T,d)
                self.time_encoder(time_feats))   # (B,T,d)

    def _temporal_aggregate(self, h):
        """1-D conv residual + attention pooling → (B,d)."""
        h_conv = self.temporal_conv(h.transpose(1, 2)).transpose(1, 2)
        h = h + h_conv
        w = F.softmax(self.attn_query(h).squeeze(-1), dim=1)                # (B,T)
        return torch.bmm(w.unsqueeze(1), h).squeeze(1)                       # (B,d)

    def _infonce(self, hp, hn):
        """InfoNCE alignment loss over the look-back window (Eq. 13)."""
        B, T, _ = hp.shape
        zp = F.normalize(self.price_proj(hp), dim=-1)                        # (B,T,d_a)
        zn = F.normalize(self.news_proj(hn),  dim=-1)
        tau = self.log_tau.exp().clamp(min=0.01)

        loss = 0.0
        for b in range(B):
            logits = zp[b] @ zn[b].T / tau                                   # (T,T)
            labels = torch.arange(T, device=logits.device)
            loss += (F.cross_entropy(logits, labels)
                     + F.cross_entropy(logits.T, labels)) / 2.0
        return loss / B

    # ────────────────────────────────────────────────
    def forward(self, price, news, time_feats, compute_align=True):
        """
        Returns
        -------
        positions  : (B, num_assets)  in [-1, 1]
        align_loss : scalar  (0 if compute_align is False)
        """
        hp, hn, ht = self._encode(price, news, time_feats)

        align_loss = self._infonce(hp, hn) if compute_align else torch.tensor(0.0)

        h_fused = self.fusion(hp, hn, ht)               # (B,T,d)
        ctx     = self._temporal_aggregate(h_fused)     # (B,d)
        p_ctx   = self._temporal_aggregate(hp)          # (B,d)

        positions = self.position_head(torch.cat([ctx, p_ctx], dim=-1))
        return positions, align_loss

    # helpers for ANIL
    def head_param_names(self):
        return {n for n, _ in self.named_parameters()
                if n.startswith("position_head.")}

class Asinh(nn.Module):
    def forward(self, x):
        return torch.asinh(x)