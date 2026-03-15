# config_equity.py — P2-ETF-DLINEAR-CROSS
# Configuration for Option A: Equity ETFs
# Phase 3: XES removed from universe

import math
from datetime import datetime, date

MODULE          = "A"
LABEL           = "Equity ETFs"
HF_DATASET_REPO = "P2SAMAPA/etf-dlinear-cross-data"
HF_SUBDIR       = "equity"
PARQUET_FILE    = "ohlcv_equity.parquet"
METADATA_FILE   = "metadata_equity.json"
RESULTS_DIR     = "results/equity"

# ETF Universe — XES removed in Phase 3
# XES (Oil & Gas Equipment & Services) suspected of distorting PRC loss
# due to low price (~$25) vs other ETFs (SPY ~$550, QQQ ~$500)
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
    # "XES" removed — Phase 3 experiment
]

START_DATE = "2006-01-01"

# ── Rolling train/val/test split ──────────────────────────────────────────────
SPLIT_TEST_RATIO  = 0.15
SPLIT_VAL_RATIO   = 0.10

# Display labels only
_today = date.today()
TEST_YEAR = _today.year - 1
VAL_YEAR  = _today.year - 2

# Model hyperparameters
SEQ_LEN    = 96
PRED_LEN   = 1
LABEL_LEN  = 0
BATCH_SIZE = 32
EPOCHS     = 300
LR         = 0.005
GAMMA      = 10

# Anti-collapse bias init
OUTPUT_BIAS_INIT = 1.0

# Disable Hold node — forces model to always allocate
USE_HOLD     = False

# ── Model variants to train ───────────────────────────────────────────────────
MODEL_VARIANTS = [
    ("dlinear",     "L2", "PRC"),   # Phase 2/3 baseline
    ("crossformer", "L2", "PRC"),   # Phase 2/3 baseline
    ("dlinear",     "L2", "RET"),   # Phase 2/3 best performer
    ("crossformer", "L2", "RET"),   # Phase 2/3 best performer
    ("mole",        "L2", "PRC"),   # Phase 4: MoLE-DLinear + Price loss
    ("mole",        "L2", "RET"),   # Phase 4: MoLE-DLinear + Return loss
]

# MoLE hyperparameters (Phase 4)
MOLE_N_HEADS      = 4     # number of DLinear experts
MOLE_HEAD_DROPOUT = 0.0   # head dropout rate (0 = off, 0.2 = regularised)

# DLinear
DLINEAR_INDIVIDUAL = False

# Crossformer
CROSS_D_MODEL  = 64
CROSS_N_HEADS  = 2
CROSS_E_LAYERS = 2
CROSS_D_FF     = 128
CROSS_SEG_LEN  = 12
CROSS_WIN_SIZE = 2
CROSS_DROPOUT  = 0.2

FEATURE_COLS = ["Close", "Volume"]
N_ASSETS     = len(TICKERS)   # now 9 instead of 10
