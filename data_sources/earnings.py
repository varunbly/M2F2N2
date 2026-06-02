import os
import pickle
from datetime import date, datetime
import pandas as pd
import numpy as np
from .base import BaseDataSource

class EarningsFetcher(BaseDataSource):
    def __init__(self, tickers, cache_path="/tmp/earnings_dates.pkl"):
        self.tickers = tickers
        self.cache_path = cache_path
        self.earnings_dates = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self.earnings_dates = pickle.load(f)

    def _save_cache(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.earnings_dates, f)

    def fetch_all(self):
        import yfinance as yf
        today = date.today()
        
        cache_mod_date = None
        if os.path.exists(self.cache_path):
            cache_mod_date = datetime.fromtimestamp(os.path.getmtime(self.cache_path)).date()
        
        missing = []
        for t in self.tickers:
            if t not in self.earnings_dates:
                missing.append(t)
            else:
                if cache_mod_date != today:
                    dates = self.earnings_dates[t]
                    if not dates or max(dates) < today:
                        missing.append(t)
                    
        if missing:
            print(f"  Fetching earnings dates for {len(missing)} out-of-date or missing tickers...")
            for i, ticker in enumerate(missing):
                try:
                    t = yf.Ticker(f"{ticker}.NS")
                    ed = t.earnings_dates
                    if ed is not None and not ed.empty:
                        dates = [d.tz_localize(None).date() if hasattr(d, 'tz_localize') else d.date() for d in ed.index]
                        self.earnings_dates[ticker] = sorted(list(set(dates)))
                    else:
                        self.earnings_dates[ticker] = []
                except Exception as e:
                    print(f"    ✗ {ticker}: {e}")
                    self.earnings_dates[ticker] = []
                if (i+1) % 10 == 0:
                    self._save_cache()
            self._save_cache()
            print(f"  ✓ Total earnings dates fetched/updated.")
        else:
            print("  ✓ All earnings dates are already up-to-date.")
        return self.earnings_dates
