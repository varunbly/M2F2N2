from .news import NewsEmbedder
from .earnings import EarningsFetcher

DATA_SOURCES = {
    "news": NewsEmbedder,
    "earnings": EarningsFetcher
}
