import os
import pickle
import hashlib
import numpy as np
from core.config import NEWS_DIM, NEWS_CACHE_PATH
from .base import BaseDataSource

class NewsEmbedder(BaseDataSource):
    """
    Mirrors M2VN's use of TiMaGPT:
      - For every trading date, concatenate all published articles
      - Embed with a frozen language model
      - Cache embeddings to disk for reproducibility
    We use all-MiniLM-L6-v2 as a practical stand-in for TiMaGPT.
    Temporal integrity is preserved: only news published on or before
    each date is used.
    """

    def __init__(self, client, cache_path=NEWS_CACHE_PATH):
        self.client     = client
        self.cache_path = cache_path
        self.embeddings = {}
        self._model     = None
        self._load_cache()

    # ── sentence-transformer (lazy init) ──
    def _init_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            print("  ✓ Loaded sentence-transformers model (all-MiniLM-L6-v2)")
        except ImportError:
            print("  ⚠ sentence-transformers not installed — using hash fallback")
            self._model = "fallback"

    def embed_text(self, text: str) -> np.ndarray:
        self._init_model()
        if self._model == "fallback":
            h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
            rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
            return rng.randn(NEWS_DIM).astype(np.float32) * 0.1
        return self._model.encode(text, show_progress_bar=False).astype(np.float32)

    # ── disk cache ──
    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                self.embeddings = pickle.load(f)
            print(f"  Loaded {len(self.embeddings)} cached news embeddings")

    def _save_cache(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.embeddings, f)

    # ── main entry ──
    def fetch_all(self) -> dict:
        all_dates = self.client.get_all_news_dates()
        missing   = [d for d in all_dates if d not in self.embeddings]
        if not missing:
            print(f"  All {len(self.embeddings)} news dates already cached")
            return self.embeddings

        print(f"  Embedding {len(missing)} news dates "
              f"(already cached: {len(self.embeddings)})…")
        for i, date in enumerate(sorted(missing)):
            try:
                news = self.client.get_news(date)
                text = news.get("text", "") if isinstance(news, dict) else str(news)
                self.embeddings[date] = self.embed_text(text)
            except Exception as e:
                print(f"    ✗ {date}: {e}")
                self.embeddings[date] = np.zeros(NEWS_DIM, dtype=np.float32)
            if (i + 1) % 50 == 0:
                print(f"    … {i+1}/{len(missing)}")
                self._save_cache()

        self._save_cache()
        print(f"  ✓ Total news embeddings: {len(self.embeddings)}")
        return self.embeddings
