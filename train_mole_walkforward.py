"""
train_mole_walkforward.py — P2-ETF-DLINEAR-CROSS
=================================================
Walk-forward training for MoLE-DLinear + RET loss only.

Each fold:
  - Train window : 3 years
  - Val window   : 1 year
  - Test window  : 1 year
  - Rolling forward by 1 year per fold

All fold test predictions are concatenated into one continuous
out-of-sample equity curve covering the full data history.

Usage:
  python train_mole_walkforward.py --module A
  python train_mole_walkforward.py --module B
"""

import os
import sys
import json
import pickle
import argparse
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler

from data_loader import build_features, compute_timestamp_features
from model import get_model
from loss_functions import stock_loss_l2
from evaluate import compute_metrics, buy_and_hold, buy_and_hold_single


# ── Constants ────────────────────────────────────────────────────────────────

TRAIN_YEARS = 3
VAL_YEARS   = 1
TEST_YEARS  = 1
STEP_YEARS  = 1    # roll forward 1 year per fold


# ── Data loading ─────────────────────────────────────────────────────────────

def load_raw_data(cfg, token=None):
    """Load full OHLCV parquet from HuggingFace."""
    token = token or os.getenv("HF_TOKEN")
    local = hf_hub_download(
        repo_id=cfg.HF_DATASET_REPO,
        repo_type="dataset",
        filename=f"{cfg.HF_SUBDIR}/{cfg.PARQUET_FILE}",
        token=token,
        force_download=False,
    )
    ohlcv_df = pd.read_parquet(local)
    ohlcv_df.index = pd.to_datetime(ohlcv_df.index).tz_localize(None)
    features_df, prices_df = build_features(ohlcv_df, cfg.TICKERS)
    valid       = features_df.dropna().index.intersection(prices_df.dropna().index)
    features_df = features_df.loc[valid]
    prices_df   = prices_df.loc[valid]
    print(f"   Loaded {len(features_df)} rows "
          f"({features_df.index[0].date()} → {features_df.index[-1].date()})")
    return features_df, prices_df


# ── Walk-forward fold generator ───────────────────────────────────────────────

def generate_folds(index: pd.DatetimeIndex):
    """
    Generate (train_start, train_end, val_end, test_end) date tuples.
    Each fold: 3yr train + 1yr val + 1yr test, rolling by 1yr.
    """
    folds = []
    start = index[0]
    end   = index[-1]

    fold_start = start
    while True:
        train_end = fold_start + relativedelta(years=TRAIN_YEARS)
        val_end   = train_end  + relativedelta(years=VAL_YEARS)
        test_end  = val_end    + relativedelta(years=TEST_YEARS)

        if test_end > end + relativedelta(months=6):
            break

        # Clamp test_end to actual data
        actual_test_end = min(test_end, end)

        folds.append((fold_start, train_end, val_end, actual_test_end))
        fold_start = fold_start + relativedelta(years=STEP_YEARS)

    return folds


# ── Dataset ───────────────────────────────────────────────────────────────────

class ETFWindowDataset(torch.utils.data.Dataset):
    def __init__(self, features, prices, timestamps, seq_len):
        self.features   = torch.tensor(features,   dtype=torch.float32)
        self.prices     = torch.tensor(prices,     dtype=torch.float32)
        self.timestamps = torch.tensor(timestamps, dtype=torch.float32)
        self.seq_len    = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X        = self.features[idx : idx + self.seq_len]
        ts_mark  = self.timestamps[idx]
        p_next   = self.prices[idx + self.seq_len]
        p_curr   = self.prices[idx + self.seq_len - 1]
        ret_diff = torch.where(
            p_curr > 0,
            (p_next - p_curr) / p_curr.clamp(min=1e-8),
            torch.zeros_like(p_curr)
        )
        return X, ts_mark, ret_diff


# ── Train one fold ────────────────────────────────────────────────────────────

def train_fold(fold_idx, features_df, prices_df, fold_dates, cfg, device):
    fold_start, train_end, val_end, test_end = fold_dates

    # Slice windows
    train_mask = (features_df.index >= fold_start) & (features_df.index < train_end)
    val_mask   = (features_df.index >= train_end)  & (features_df.index < val_end)
    test_mask  = (features_df.index >= val_end)    & (features_df.index <= test_end)

    feat_train = features_df[train_mask].values
    feat_val   = features_df[val_mask].values
    feat_test  = features_df[test_mask].values

    prc_train  = prices_df[train_mask].values
    prc_val    = prices_df[val_mask].values
    prc_test   = prices_df[test_mask].values

    ts_train   = compute_timestamp_features(features_df[train_mask].index)
    ts_val     = compute_timestamp_features(features_df[val_mask].index)
    ts_test    = compute_timestamp_features(features_df[test_mask].index)

    if len(feat_train) < cfg.SEQ_LEN + 50:
        print(f"   Fold {fold_idx}: insufficient train data ({len(feat_train)} rows) — skipping")
        return None

    if len(feat_test) < cfg.SEQ_LEN + 10:
        print(f"   Fold {fold_idx}: insufficient test data ({len(feat_test)} rows) — skipping")
        return None

    # Scale
    scaler     = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_val   = scaler.transform(feat_val)
    feat_test  = scaler.transform(feat_test)

    # Datasets
    train_ds = ETFWindowDataset(feat_train, prc_train, ts_train, cfg.SEQ_LEN)
    val_ds   = ETFWindowDataset(feat_val,   prc_val,   ts_val,   cfg.SEQ_LEN)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = torch.utils.data.DataLoader(
        val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False)

    n_features = feat_train.shape[1]
    model = get_model("mole", "RET", cfg, n_features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-6)

    best_val  = float('inf')
    patience  = 15
    no_improve = 0
    best_state = None

    # Walk-forward uses fewer epochs — smaller training window converges faster
    wf_epochs = min(cfg.EPOCHS, 150)
    for epoch in range(1, wf_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X, ts, ret_diff in train_loader:
            X, ts, Y = X.to(device), ts.to(device), ret_diff.to(device)
            optimizer.zero_grad()
            O = model(X, ts)
            loss = stock_loss_l2(O, Y, gamma=cfg.GAMMA,
                                 use_hold=getattr(cfg, 'USE_HOLD', False))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, ts, ret_diff in val_loader:
                X, ts, Y = X.to(device), ts.to(device), ret_diff.to(device)
                O = model(X, ts)
                val_loss += stock_loss_l2(O, Y, gamma=cfg.GAMMA,
                                          use_hold=getattr(cfg, 'USE_HOLD', False)).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"     Epoch {epoch:3d}/{wf_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"     Early stop at epoch {epoch} | Best val: {best_val:.4f}")
                break

    model.load_state_dict(best_state)
    model.eval()

    # Generate test signals
    feat_test_scaled = scaler.transform(features_df[test_mask].values)
    n_samples = len(feat_test_scaled) - cfg.SEQ_LEN
    signals   = []
    ts_test_arr = compute_timestamp_features(features_df[test_mask].index)

    with torch.no_grad():
        for i in range(n_samples):
            X  = torch.tensor(feat_test_scaled[i:i+cfg.SEQ_LEN],
                               dtype=torch.float32).unsqueeze(0).to(device)
            ts = torch.tensor(ts_test_arr[i],
                               dtype=torch.float32).unsqueeze(0).to(device)
            O  = model(X, ts).squeeze(0).cpu().numpy()
            signals.append(O)

    signals      = np.array(signals)
    price_window = prc_test[cfg.SEQ_LEN - 1:]

    return {
        "fold_idx":    fold_idx,
        "train_start": str(fold_start.date()),
        "train_end":   str(train_end.date()),
        "val_end":     str(val_end.date()),
        "test_start":  str(features_df[test_mask].index[cfg.SEQ_LEN - 1].date()),
        "test_end":    str(test_end.date()),
        "best_val":    round(best_val, 6),
        "signals":     signals,
        "prices":      price_window,
        "test_dates":  [str(d.date()) for d in
                        features_df[test_mask].index[cfg.SEQ_LEN - 1:]],
    }


# ── Backtest walk-forward equity curve ────────────────────────────────────────

def build_walkforward_equity(fold_results, initial=10_000.0):
    """
    Concatenate all fold test signals into one continuous equity curve.
    De-duplicate overlapping test dates by taking the most recent fold's signal.
    """
    # Collect all (date, signals, prices) rows
    all_rows = {}
    for fold in fold_results:
        if fold is None:
            continue
        for i, date_str in enumerate(fold["test_dates"]):
            if i < len(fold["signals"]):
                all_rows[date_str] = {
                    "signals": fold["signals"][i],
                    "prices":  fold["prices"][i],
                }

    sorted_dates = sorted(all_rows.keys())
    if not sorted_dates:
        return None, None

    # Build equity curve
    budget    = initial
    portfolio = [budget]
    N         = len(all_rows[sorted_dates[0]]["signals"])

    for i in range(len(sorted_dates) - 1):
        row      = all_rows[sorted_dates[i]]
        O        = row["signals"][:N]
        p_curr   = row["prices"][:N]
        p_next   = all_rows[sorted_dates[i+1]]["prices"][:N]

        abs_sigs = np.abs(O)
        total    = abs_sigs.sum()
        if total < 1e-8:
            portfolio.append(budget)
            continue

        weights  = abs_sigs / total
        signs    = np.sign(O)
        pct_rets = np.where(
            p_curr > 0,
            (p_next - p_curr) / p_curr,
            np.zeros_like(p_curr)
        )
        daily_ret = float(np.sum(weights * signs * pct_rets))
        budget    = max(budget * (1.0 + daily_ret), 0.0)
        portfolio.append(budget)

    return np.array(portfolio), sorted_dates


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["A", "B"], required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    today_str = datetime.utcnow().strftime("%Y%m%d")
    cfg_map   = {"A": "config_equity", "B": "config_fixed_income"}
    cfg       = importlib.import_module(cfg_map[args.module])
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice   : {device}")
    print(f"Module   : {cfg.MODULE} - {cfg.LABEL}")
    print(f"Run date : {today_str}")
    print(f"Strategy : Walk-forward MoLE-DLinear + RET")
    print(f"Windows  : {TRAIN_YEARS}yr train + {VAL_YEARS}yr val + "
          f"{TEST_YEARS}yr test, rolling {STEP_YEARS}yr")

    token = os.getenv("HF_TOKEN")
    print(f"\nLoading data...")
    features_df, prices_df = load_raw_data(cfg, token)

    # Generate folds
    folds = generate_folds(features_df.index)
    print(f"\nGenerated {len(folds)} walk-forward folds:")
    for i, (fs, te, ve, tse) in enumerate(folds):
        print(f"  Fold {i+1:2d}: "
              f"Train {fs.date()}→{te.date()} | "
              f"Val {te.date()}→{ve.date()} | "
              f"Test {ve.date()}→{tse.date()}")

    # Train each fold
    fold_results = []
    for i, fold_dates in enumerate(folds):
        print(f"\n{'='*55}")
        print(f"  FOLD {i+1}/{len(folds)} — "
              f"Test: {fold_dates[2].date()} → {fold_dates[3].date()}")
        print(f"{'='*55}")
        result = train_fold(i+1, features_df, prices_df,
                            fold_dates, cfg, device)
        fold_results.append(result)

    # Build concatenated equity curve
    print(f"\nBuilding walk-forward equity curve...")
    portfolio, dates = build_walkforward_equity(fold_results)

    if portfolio is None:
        print("No valid fold results — exiting")
        return

    metrics    = compute_metrics(portfolio)
    n_folds_ok = sum(1 for f in fold_results if f is not None)

    # Top ETF signal from most recent fold
    last_fold    = next((f for f in reversed(fold_results) if f is not None), None)
    last_signals = last_fold["signals"][-1] if last_fold else np.zeros(cfg.N_ASSETS)
    top_idx      = int(np.argmax(np.abs(last_signals)))
    top_ticker   = cfg.TICKERS[top_idx]
    top_signal   = float(last_signals[top_idx])

    print(f"\nWalk-Forward MoLE-RET Results ({n_folds_ok} folds):")
    print(f"  Total Return : {metrics['total_return_pct']:.2f}%")
    print(f"  CAGR         : {metrics['annual_return_pct']:.2f}%")
    print(f"  Sharpe       : {metrics['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown : {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Top Signal   : {top_ticker} "
          f"{'BUY' if top_signal > 0 else 'SHORT'} "
          f"({abs(top_signal)*100:.1f}%)")

    # Save results
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out = {
        "model":           "mole_ret_walkforward",
        "run_date":        today_str,
        "strategy":        f"walk_forward_{TRAIN_YEARS}yr_train_{VAL_YEARS}yr_val",
        "n_folds":         len(folds),
        "n_folds_ok":      n_folds_ok,
        "metrics":         metrics,
        "portfolio_values": portfolio.tolist(),
        "dates":           dates,
        "fold_summary": [
            {
                "fold":       f["fold_idx"],
                "train":      f"{f['train_start']}→{f['train_end']}",
                "test":       f"{f['test_start']}→{f['test_end']}",
                "best_val":   f["best_val"],
            }
            for f in fold_results if f is not None
        ],
        "last_signal": {
            "ticker":    top_ticker,
            "direction": "BUY" if top_signal > 0 else "SHORT",
            "raw_value": round(top_signal, 4),
            "all_signals": {
                cfg.TICKERS[i]: round(float(last_signals[i]), 4)
                for i in range(cfg.N_ASSETS)
            }
        }
    }

    save_path = os.path.join(
        cfg.RESULTS_DIR, f"mole_ret_walkforward_{today_str}.json")
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    main()
