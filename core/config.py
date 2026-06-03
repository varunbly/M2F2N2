# core/config.py
VERSION         = "V1.0.2"
WINDOW          = 100
NUM_ASSETS      = 50
NEWS_DIM        = 384
LATENT_DIM      = 50
ALIGN_DIM       = 25
TIME_FEATS      = 4
INNER_LR        = 0.005
META_LR         = 0.0005
INNER_STEPS     = 3
LAMBDA_ALIGN    = 0.1
TEMPERATURE     = 0.07
META_EPOCHS     = 10
BATCH_SIZE      = 8
MAX_WEIGHT      = 0.07
TXN_COST        = 0.001

NEWS_CACHE_PATH = "/tmp/numin2_news_embeddings.pkl"
DATA_PATH       = "/tmp/consolidated_daily_returns.parquet"
PLOT_PATH       = f"{WINDOW}-{NUM_ASSETS}-{NEWS_DIM}-{LATENT_DIM}-{ALIGN_DIM}-{TIME_FEATS}-{INNER_LR}-{META_LR}-{INNER_STEPS}-{LAMBDA_ALIGN}-{TEMPERATURE}-{META_EPOCHS}-{BATCH_SIZE}-{MAX_WEIGHT}-{TXN_COST}-{VERSION}.png"
