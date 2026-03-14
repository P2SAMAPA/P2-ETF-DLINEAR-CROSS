"""
data_loader.py — P2-ETF-DLINEAR-CROSS
========================================
Loads OHLCV parquet from HuggingFace, computes derived features,
and returns train/val/test tensors ready for model input.

Features computed per ETF:
  - Close price (PRC)         ← used in StockLoss-L2 PRC
  - Daily return
  - Volume change %
  - 5-day moving average
  - 20-day moving average
  - RSI (14-day)

Output tensors:
  X : (samples, seq_len, N * n_features)   ← model input
  Y : (samples, N)                          ← price differences PRC_{t+1} - PRC_t
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_l  = loss.ewm(com=window - 1, min_periods=window).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(ohlcv_df: pd.DataFrame, tickers: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a MultiIndex OHLCV DataFrame (ticker, field),
    return:
      features_df : (dates, N * n_features)  — scaled model input
      prices_df   : (dates, N)               — raw Close prices for loss computation
    """
    feature_frames = []
    price_frames   = []

    for ticker in tickers:
        if ticker not in ohlcv_df.columns.get_level_values(0):
            raise ValueError(f"Ticker {ticker} not found in dataset")

        tk = ohlcv_df[ticker].copy()

        close  = tk["Close"]
        volume = tk["Volume"] if "Volume" in tk.columns else pd.Series(np.nan, index=tk.index)

        ret         = close.pct_change()
        vol_change  = volume.pct_change()
        ma5         = close.rolling(5).mean()
        ma20        = close.rolling(20).mean()
        rsi         = compute_rsi(close, 14)

        feat = pd.DataFrame({
            f"{ticker}_close":      close,
            f"{ticker}_return":     ret,
            f"{ticker}_vol_change": vol_change,
            f"{ticker}_ma5":        ma5,
            f"{ticker}_ma20":       ma20,
            f"{ticker}_rsi":        rsi,
        }, index=tk.index)

        feature_frames.append(feat)
        price_frames.append(close.rename(ticker))

    features_df = pd.concat(feature_frames, axis=1)
    prices_df   = pd.concat(price_frames, axis=1)

    return features_df, prices_df


# ── Dataset class ─────────────────────────────────────────────────────────────

class ETFDataset(Dataset):
    """
    Sliding window dataset.
    X[i] = features[i : i+seq_len]          shape (seq_len, n_features)
    Y[i] = price_diff[i+seq_len]             shape (N,)
             = Close[i+seq_len] - Close[i+seq_len-1]
    """
    def __init__(self, features: np.ndarray, prices: np.ndarray, seq_len: int):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.prices   = torch.tensor(prices,   dtype=torch.float32)
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X = self.features[idx : idx + self.seq_len]          # (seq_len, n_feat)
        # Price diff at the prediction step
        p_next = self.prices[idx + self.seq_len]             # (N,)
        p_curr = self.prices[idx + self.seq_len - 1]         # (N,)
        Y = p_next - p_curr                                   # (N,)
        return X, Y


# ── Main loader ───────────────────────────────────────────────────────────────

def load_data(
    cfg,
    seq_len:    int  = 96,
    batch_size: int  = 32,
    token:      str  = None,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Download parquet from HF, compute features, split, scale, return DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, n_features
    """
    token = token or os.getenv("HF_TOKEN")

    # Download parquet from HF
    print(f"📥 Loading {cfg.PARQUET_FILE} from HuggingFace...")
    local = hf_hub_download(
        repo_id=cfg.HF_DATASET_REPO,
        repo_type="dataset",
        filename=f"{cfg.HF_SUBDIR}/{cfg.PARQUET_FILE}",
        token=token,
        force_download=False,
    )
    ohlcv_df = pd.read_parquet(local)
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index).tz_localize(None)
    print(f"   Loaded {len(ohlcv_df)} rows, {ohlcv_df.shape[1]} columns")

    # Build features
    features_df, prices_df = build_features(ohlcv_df, cfg.TICKERS)

    # Drop rows with NaN (from rolling windows / RSI warmup)
    valid = features_df.dropna().index.intersection(prices_df.dropna().index)
    features_df = features_df.loc[valid]
    prices_df   = prices_df.loc[valid]

    # Train / Val / Test split by year
    test_mask  = features_df.index.year == cfg.TEST_YEAR
    val_mask   = features_df.index.year == cfg.VAL_YEAR
    train_mask = features_df.index.year < cfg.VAL_YEAR

    feat_train = features_df[train_mask].values
    feat_val   = features_df[val_mask].values
    feat_test  = features_df[test_mask].values

    prc_train  = prices_df[train_mask].values
    prc_val    = prices_df[val_mask].values
    prc_test   = prices_df[test_mask].values

    # Scale features (fit on train only)
    scaler = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_val   = scaler.transform(feat_val)
    feat_test  = scaler.transform(feat_test)

    n_features = feat_train.shape[1]

    # Datasets
    train_ds = ETFDataset(feat_train, prc_train, seq_len)
    val_ds   = ETFDataset(feat_val,   prc_val,   seq_len)
    test_ds  = ETFDataset(feat_test,  prc_test,  seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"   Train: {len(train_ds)} samples | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"   Features per timestep: {n_features}")

    return train_loader, val_loader, test_loader, n_features, scaler
