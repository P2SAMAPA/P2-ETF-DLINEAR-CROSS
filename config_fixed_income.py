# config_fixed_income.py — P2-ETF-DLINEAR-CROSS
# Configuration for Option B: Fixed Income / Commodity ETFs

MODULE          = "B"
LABEL           = "Fixed Income / Commodity ETFs"
HF_DATASET_REPO = "P2SAMAPA/etf-dlinear-cross-data"
HF_SUBDIR       = "fixed_income"
PARQUET_FILE    = "ohlcv_fixed_income.parquet"
METADATA_FILE   = "metadata_fixed_income.json"
RESULTS_DIR     = "results/fixed_income"

# ETF Universe
TICKERS = [
    "TLT",   # 20+ Year Treasury Bond
    "VNQ",   # Real Estate (REITs)
    "GLD",   # Gold
    "SLV",   # Silver
    "LQD",   # Investment Grade Corporate Bonds
    "HYG",   # High Yield Corporate Bonds
    "MBB",   # Mortgage-Backed Securities
    "PFF",   # Preferred Stock
]

START_DATE = "2008-01-01"

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
