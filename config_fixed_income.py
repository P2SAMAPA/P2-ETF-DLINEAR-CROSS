# config_fixed_income.py — P2-ETF-DLINEAR-CROSS
# Configuration for Option B: Fixed Income / Commodity ETFs

from datetime import date

MODULE          = "B"
LABEL           = "Fixed Income / Commodity ETFs"
HF_DATASET_REPO = "P2SAMAPA/etf-dlinear-cross-data"
HF_SUBDIR       = "fixed_income"
PARQUET_FILE    = "ohlcv_fixed_income.parquet"
METADATA_FILE   = "metadata_fixed_income.json"
RESULTS_DIR     = "results/fixed_income"

# ETF Universe
TICKERS = [
    "TLT", "VNQ", "GLD", "SLV",
    "LQD", "HYG", "MBB", "PFF",
]

START_DATE = "2008-01-01"

# ── Rolling train/val/test split ──────────────────────────────────────────────
SPLIT_TEST_RATIO  = 0.10
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
EPOCHS     = 50
LR         = 0.005
GAMMA      = 10

# Anti-collapse bias init
OUTPUT_BIAS_INIT = 1.0

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
