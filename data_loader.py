"""
data_loader.py — P2-ETF-DLINEAR-CROSS
========================================
Loads OHLCV parquet from HuggingFace, computes derived features,
and returns train/val/test tensors using a rolling 80/10/10 split
based on row counts — NOT hardcoded calendar years.

Features per ETF (6 total):
  - Close price (PRC)
  - Daily return
  - Volume change %
  - 5-day moving average
  - 20-day moving average
  - RSI (14-day)
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
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_l = loss.ewm(com=window - 1, min_periods=window).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(ohlcv_df: pd.DataFrame, tickers: list) -> tuple:
    feature_frames = []
    price_frames   = []

    for ticker in tickers:
        if ticker not in ohlcv_df.columns.get_level_values(0):
            raise ValueError(f"Ticker {ticker} not found in dataset")

        tk     = ohlcv_df[ticker].copy()
        close  = tk["Close"]
        volume = tk["Volume"] if "Volume" in tk.columns else pd.Series(np.nan, index=tk.index)

        ret        = close.pct_change()
        vol_change = volume.pct_change()
        ma5        = close.rolling(5).mean()
        ma20       = close.rolling(20).mean()
        rsi        = compute_rsi(close, 14)

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


# ── Dataset ───────────────────────────────────────────────────────────────────

class ETFDataset(Dataset):
    def __init__(self, features: np.ndarray, prices: np.ndarray, seq_len: int):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.prices   = torch.tensor(prices,   dtype=torch.float32)
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X      = self.features[idx : idx + self.seq_len]
        p_next = self.prices[idx + self.seq_len]
        p_curr = self.prices[idx + self.seq_len - 1]
        Y      = p_next - p_curr
        return X, Y


# ── Main loader ───────────────────────────────────────────────────────────────

def load_data(cfg, seq_len: int = 96, batch_size: int = 32,
              token: str = None) -> tuple:
    """
    Rolling 80/10/10 split by row count — no hardcoded years.
    Returns: train_loader, val_loader, test_loader, n_features, scaler
    """
    token = token or os.getenv("HF_TOKEN")

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

    features_df, prices_df = build_features(ohlcv_df, cfg.TICKERS)

    # Drop NaN rows from rolling windows / RSI warmup
    valid       = features_df.dropna().index.intersection(prices_df.dropna().index)
    features_df = features_df.loc[valid]
    prices_df   = prices_df.loc[valid]

    n = len(features_df)

    # Rolling 80/10/10 split by row count
    test_size  = max(int(n * cfg.SPLIT_TEST_RATIO), seq_len + 10)
    val_size   = max(int(n * cfg.SPLIT_VAL_RATIO),  seq_len + 10)
    train_size = n - val_size - test_size

    assert train_size > seq_len, f"Not enough training data: {train_size} rows"

    feat_train = features_df.iloc[:train_size].values
    feat_val   = features_df.iloc[train_size : train_size + val_size].values
    feat_test  = features_df.iloc[train_size + val_size :].values

    prc_train  = prices_df.iloc[:train_size].values
    prc_val    = prices_df.iloc[train_size : train_size + val_size].values
    prc_test   = prices_df.iloc[train_size + val_size :].values

    # Log actual date ranges
    print(f"   Train: {features_df.index[0].date()} → {features_df.index[train_size-1].date()} ({train_size} rows)")
    print(f"   Val  : {features_df.index[train_size].date()} → {features_df.index[train_size+val_size-1].date()} ({val_size} rows)")
    print(f"   Test : {features_df.index[train_size+val_size].date()} → {features_df.index[-1].date()} ({len(feat_test)} rows)")

    # Scale — fit on train only
    scaler     = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_val   = scaler.transform(feat_val)
    feat_test  = scaler.transform(feat_test)

    n_features = feat_train.shape[1]

    train_ds = ETFDataset(feat_train, prc_train, seq_len)
    val_ds   = ETFDataset(feat_val,   prc_val,   seq_len)
    test_ds  = ETFDataset(feat_test,  prc_test,  seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"   Train samples: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"   Features per timestep: {n_features}")

    return train_loader, val_loader, test_loader, n_features, scaler
