import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from core.config import WINDOW, NEWS_DIM, TIME_FEATS

class MultiModalDataset(Dataset):
    """
    For each index i, returns:
      price      (WINDOW, NUM_ASSETS)  — lookback returns
      news       (WINDOW, NEWS_DIM)    — daily news embeddings
      time_feats (WINDOW, TIME_FEATS)  — calendar features
      target     (NUM_ASSETS,)         — next-day returns
    """

    def __init__(self, df: pd.DataFrame, news_embeddings: dict, earnings_dates: dict, window: int = WINDOW):
        self.returns  = df.values.astype(np.float32)       # (D, 50)
        self.dates    = df.index
        self.news     = news_embeddings
        self.window   = window
        self.earnings_feats = self._precompute_earnings(df, earnings_dates) # (D, 50, 2)
        # valid prediction indices: window … D-1
        self.valid_indices = list(range(window, len(self.returns)))

    def _precompute_earnings(self, df, earnings_dates):
        D = len(df)
        N = len(df.columns)
        feats = np.zeros((D, N, 2), dtype=np.float32)
        idx_times = pd.to_datetime(df.index.date).values
        for j, ticker in enumerate(df.columns):
            dates = earnings_dates.get(ticker, [])
            if not dates:
                feats[:, j, 0] = 1.0
                feats[:, j, 1] = 1.0
                continue
            edates = np.sort(pd.to_datetime(dates).values)
            idx = np.searchsorted(edates, idx_times)
            days_until = np.full(D, 1000.0)
            days_since = np.full(D, 1000.0)
            valid_next = idx < len(edates)
            if np.any(valid_next):
                diff = edates[idx[valid_next]] - idx_times[valid_next]
                days_until[valid_next] = diff.astype('timedelta64[D]').astype(float)
            valid_last = idx > 0
            if np.any(valid_last):
                diff = idx_times[valid_last] - edates[idx[valid_last]-1]
                days_since[valid_last] = diff.astype('timedelta64[D]').astype(float)
            feats[:, j, 0] = np.clip(days_since / 100.0, 0, 1)
            feats[:, j, 1] = np.clip(days_until / 100.0, 0, 1)
        return feats

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]

        # Price lookback
        price = self.returns[t - self.window : t]                       # (W, 50)
        
        # Earnings features lookback
        efeats = self.earnings_feats[t - self.window : t]               # (W, 50, 2)
        efeats = efeats.reshape(self.window, -1)                        # (W, 100)
        
        price_and_earnings = np.concatenate([price, efeats], axis=-1)   # (W, 150)

        # Target (next-day returns)
        target = self.returns[t]                                        # (50,)

        # News embeddings for each day in the window
        news = np.zeros((self.window, NEWS_DIM), dtype=np.float32)
        for k in range(self.window):
            d = self.dates[t - self.window + k]
            key = d.strftime("%d-%m-%Y")
            if key in self.news:
                news[k] = self.news[key]

        # Calendar features (normalised to [0, 1])
        tf = np.zeros((self.window, TIME_FEATS), dtype=np.float32)
        for k in range(self.window):
            d = self.dates[t - self.window + k]
            tf[k] = [d.dayofweek / 6.0,
                      d.month / 12.0,
                      d.day / 31.0,
                      d.quarter / 4.0]

        return (torch.from_numpy(price_and_earnings),
                torch.from_numpy(news),
                torch.from_numpy(tf),
                torch.from_numpy(target))

    def get_month_indices(self, year: int, month: int) -> list:
        """Return dataset indices whose prediction date falls in (year, month)."""
        out = []
        for i, t in enumerate(self.valid_indices):
            d = self.dates[t]
            if d.year == year and d.month == month:
                out.append(i)
        return out
