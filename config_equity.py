# config_equity.py — P2-ETF-DLINEAR-CROSS
# Configuration for Option A: Equity ETFs

MODULE          = "A"
LABEL           = "Equity ETFs"
HF_DATASET_REPO = "P2SAMAPA/etf-dlinear-cross-data"
HF_SUBDIR       = "equity"
PARQUET_FILE    = "ohlcv_equity.parquet"
METADATA_FILE   = "metadata_equity.json"
RESULTS_DIR     = "results/equity"

# ETF Universe
TICKERS = [
    "SPY",   # S&P 500
    "QQQ",   # NASDAQ 100
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "GDX",   # Gold Miners
    "IWM",   # Russell 2000 Small Cap
    "XES",   # Oil & Gas Equipment & Services
]

START_DATE = "2006-01-01"

# Train / Val / Test split
# Test  = most recent full calendar year
# Val   = year before test
# Train = everything before val
TEST_YEAR  = 2023
VAL_YEAR   = 2022

# Model hyperparameters
SEQ_LEN    = 96       # input sequence length (trading days lookback)
PRED_LEN   = 1        # predict next 1 trading day
LABEL_LEN  = 0        # no overlap (as per paper)
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 0.001
GAMMA      = 10       # smoothing coefficient for tanh(gamma * x) ~ sign(x)

# DLinear specific
DLINEAR_INDIVIDUAL = False   # shared linear weights across all ETFs

# Crossformer specific
CROSS_D_MODEL  = 64
CROSS_N_HEADS  = 2
CROSS_E_LAYERS = 2
CROSS_D_FF     = 128
CROSS_SEG_LEN  = 12
CROSS_WIN_SIZE = 2
CROSS_DROPOUT  = 0.2

# Features used from OHLCV
FEATURE_COLS = ["Close", "Volume"]   # base; data_loader adds derived features

# Number of ETFs (N) — output layer will be N+1 (incl. Hold node)
N_ASSETS = len(TICKERS)
