# config_equity.py — P2-ETF-DLINEAR-CROSS
# Configuration for Option A: Equity ETFs

import math
from datetime import datetime, date

MODULE          = "A"
LABEL           = "Equity ETFs"
HF_DATASET_REPO = "P2SAMAPA/etf-dlinear-cross-data"
HF_SUBDIR       = "equity"
PARQUET_FILE    = "ohlcv_equity.parquet"
METADATA_FILE   = "metadata_equity.json"
RESULTS_DIR     = "results/equity"

# ETF Universe
TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE",
    "XLV", "XLI", "GDX", "IWM", "XES",
]

START_DATE = "2006-01-01"

# ── Rolling train/val/test split ──────────────────────────────────────────────
# Computed dynamically at runtime based on today's date.
# Approximate trading days from today backwards:
#   Test  : last 10% of data  (~1 year)
#   Val   : prior 10%         (~1 year)
#   Train : remaining 80%
#
# These are computed in data_loader.py using row counts, not calendar years.
# SPLIT_RATIOS drives the split — do not use TEST_YEAR/VAL_YEAR for splitting.
SPLIT_TEST_RATIO  = 0.10
SPLIT_VAL_RATIO   = 0.10
# Train ratio = 1 - test - val = 0.80 (implicit)

# Keep these for display/logging purposes only
_today = date.today()
TEST_YEAR = _today.year - 1      # approx label only
VAL_YEAR  = _today.year - 2      # approx label only

# Model hyperparameters
SEQ_LEN    = 96
PRED_LEN   = 1
LABEL_LEN  = 0
BATCH_SIZE = 32
EPOCHS     = 50   # faster iterations while tuning
LR         = 0.005  # higher LR to escape hold-collapse
GAMMA      = 10

# ── Anti-collapse: initialise output layer bias away from zero ────────────────
# Prevents model collapsing to all-HOLD at the start of training
OUTPUT_BIAS_INIT = 1.0  # stronger push away from zero

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

USE_HOLD     = False  # disable Hold node — forces model to always trade

# ── Model variants to train ───────────────────────────────────────────────────
# Each entry: (model_arch, loss_variant, loss_type)
# loss_type: "PRC" = price diff (best in paper), "RET" = return % diff
MODEL_VARIANTS = [
    ("dlinear",     "L2", "PRC"),   # DLinear + StockLoss-L2 Price
    ("crossformer", "L2", "PRC"),   # Crossformer + StockLoss-L2 Price
    ("dlinear",     "L2", "RET"),   # DLinear + StockLoss-L2 Return
    ("crossformer", "L2", "RET"),   # Crossformer + StockLoss-L2 Return
]

FEATURE_COLS = ["Close", "Volume"]
N_ASSETS     = len(TICKERS)
