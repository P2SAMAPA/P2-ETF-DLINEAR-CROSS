"""
evaluate.py — P2-ETF-DLINEAR-CROSS
=====================================
Backtests trained models on the test set and computes:
  - Annual return %
  - Sharpe ratio
  - Max drawdown
  - Per-ETF allocation data
  - Comparison vs Buy & Hold baseline

Usage:
  python evaluate.py --module A
  python evaluate.py --module B

Saves:
  results/{dir}/eval_results_YYYYMMDD.json     ← dated, cleaned next run
  results/{dir}/performance_history.json        ← permanent, never deleted
"""

import os
import sys
import json
import pickle
import glob
import argparse
import importlib
import numpy as np
import pandas as pd
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, build_features
from model import get_model


# ── Find latest dated file ────────────────────────────────────────────────────

def latest_dated_file(directory: str, prefix: str, suffix: str):
    """Find the most recent date-stamped file: prefix_YYYYMMDD.suffix"""
    if not os.path.exists(directory):
        return None
    matches = []
    for fname in os.listdir(directory):
        if fname.startswith(prefix + "_") and fname.endswith(suffix):
            date_part = fname[len(prefix)+1 : -len(suffix)]
            if date_part.isdigit() and len(date_part) == 8:
                matches.append((date_part, os.path.join(directory, fname)))
    if not matches:
        return None
    return sorted(matches)[-1][1]   # most recent


# ── Portfolio simulation ──────────────────────────────────────────────────────

def simulate_portfolio(signals: np.ndarray, prices: np.ndarray,
                       initial_budget: float = 10_000.0) -> np.ndarray:
    """
    Simulate daily portfolio value.

    signals : (T, N+1)  — model tanh outputs, last col = Hold
    prices  : (T, N)    — actual Close prices at each step
                          (NOT differences — we compute % returns here)
    """
    N      = prices.shape[1]
    budget = initial_budget
    values = [budget]

    for t in range(len(signals) - 1):
        O     = signals[t]
        # Handle both N and N+1 output shapes
        O_n   = O[:N]
        abs_O = np.abs(O_n)
        total = abs_O.sum()
        if total < 1e-8:
            values.append(budget)
            continue

        V_n = abs_O / total       # normalised trading weights
        s   = np.sign(O_n)        # buy/short decision

        # Daily % return for each ETF
        p_curr = prices[t]
        p_next = prices[t + 1]
        # Avoid division by zero
        valid  = p_curr > 0
        pct_ret = np.where(valid, (p_next - p_curr) / p_curr, 0.0)

        # Portfolio daily P&L as fraction of budget
        daily_ret = np.sum(V_n * s * pct_ret)
        budget    = budget * (1.0 + daily_ret)
        budget    = max(budget, 0.0)
        values.append(budget)

    return np.array(values)


def buy_and_hold(prices: np.ndarray, initial: float = 10_000.0) -> np.ndarray:
    """Equal-weight buy and hold — uses % returns."""
    N      = prices.shape[1]
    weight = 1.0 / N
    budget = initial
    values = [budget]
    for t in range(len(prices) - 1):
        p_curr  = prices[t]
        p_next  = prices[t + 1]
        valid   = p_curr > 0
        pct_ret = np.where(valid, (p_next - p_curr) / p_curr, 0.0)
        daily_ret = np.sum(weight * pct_ret)
        budget    = budget * (1.0 + daily_ret)
        values.append(max(budget, 0.0))
    return np.array(values)


def buy_and_hold_single(prices: np.ndarray, ticker_idx: int,
                        initial: float = 10_000.0) -> np.ndarray:
    """Buy and hold a single ETF by index."""
    budget = initial
    values = [budget]
    for t in range(len(prices) - 1):
        p_curr = prices[t, ticker_idx]
        p_next = prices[t + 1, ticker_idx]
        if p_curr > 0:
            pct_ret = (p_next - p_curr) / p_curr
        else:
            pct_ret = 0.0
        budget = budget * (1.0 + pct_ret)
        values.append(max(budget, 0.0))
    return np.array(values)


def compute_metrics(portfolio_values: np.ndarray,
                    initial: float = 10_000.0) -> dict:
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    returns = returns[~np.isnan(returns)]

    # n_days = number of trading days in portfolio (excludes initial value)
    n_days       = len(portfolio_values) - 1
    n_years      = n_days / 252.0 if n_days > 0 else 1.0

    # Total return
    total_return = (portfolio_values[-1] / initial - 1.0) * 100.0

    # True annualised return (CAGR)
    if n_years > 0 and portfolio_values[-1] > 0 and initial > 0:
        cagr = ((portfolio_values[-1] / initial) ** (1.0 / n_years) - 1.0) * 100.0
    else:
        cagr = total_return

    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 1e-8:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    roll_max  = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - roll_max) / roll_max.clip(min=1e-8)
    max_dd    = float(drawdowns.min()) * 100.0

    return {
        "total_return_pct":  round(float(total_return), 4),
        "annual_return_pct": round(float(cagr), 4),      # true CAGR
        "sharpe_ratio":      round(float(sharpe), 4),
        "max_drawdown_pct":  round(float(max_dd), 4),
        "final_value":       round(float(portfolio_values[-1]), 2),
        "initial_value":     round(float(initial), 2),
        "n_days":            n_days,
        "n_years":           round(float(n_years), 2),
    }


# ── Get signals and raw prices from test set ──────────────────────────────────

def model_forward(model, X, ts_mark):
    """Forward pass — passes ts_mark for MoLE, ignores for others."""
    import inspect
    sig = inspect.signature(model.forward)
    if 'ts_mark' in sig.parameters:
        return model(X, ts_mark)
    return model(X)


# ── Evaluate one model ────────────────────────────────────────────────────────

def evaluate_model(model_name: str, cfg, test_prices: np.ndarray,
                   test_features: np.ndarray, scaler, device) -> dict:
    """
    test_prices  : (T, N) raw Close prices for test period
    test_features: (T, n_feat) unscaled features for test period
    """
    # Find latest weights — don't require today's date
    weight_path = latest_dated_file(cfg.RESULTS_DIR, f"{model_name}_best", ".pt")
    if not weight_path:
        print(f"  ⚠️  No weights found for {model_name}. Skipping.")
        return {}

    print(f"  ✅ Loading {model_name} weights from {weight_path}")
    model = get_model(model_name, cfg).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    seq_len    = cfg.SEQ_LEN
    N          = cfg.N_ASSETS
    n_samples  = len(test_features) - seq_len

    if n_samples <= 0:
        print(f"  ⚠️  Not enough test data ({len(test_features)} rows, need >{seq_len})")
        return {}

    # Scale features
    feat_scaled = scaler.transform(test_features)

    # Precompute timestamp features for test period
    from data_loader import compute_timestamp_features
    ts_feats = compute_timestamp_features(
        pd.DatetimeIndex(
            pd.date_range(
                start=pd.Timestamp("2000-01-01"),
                periods=len(test_features),
                freq="B"
            )
        )
    )

    # Generate signals day by day
    signals = []
    for i in range(n_samples):
        x  = feat_scaled[i : i + seq_len]
        ts = ts_feats[i]
        X       = torch.tensor(x,  dtype=torch.float32).unsqueeze(0).to(device)
        ts_mark = torch.tensor(ts, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            O = model_forward(model, X, ts_mark).squeeze(0).cpu().numpy()
        signals.append(O)
    signals = np.array(signals)   # (n_samples, N or N+1)

    # Prices aligned with signals: price[i] = close at signal day,
    # price[i+1] = close at next day
    price_window = test_prices[seq_len - 1 :]   # (n_samples+1, N)

    portfolio = simulate_portfolio(signals, price_window)
    bh        = buy_and_hold(price_window)
    metrics   = compute_metrics(portfolio)
    bh_metrics = compute_metrics(bh)

    print(f"     Annual Return : {metrics['annual_return_pct']:.2f}%  "
          f"(B&H: {bh_metrics['annual_return_pct']:.2f}%)")
    print(f"     Sharpe Ratio  : {metrics['sharpe_ratio']:.3f}")
    print(f"     Max Drawdown  : {metrics['max_drawdown_pct']:.2f}%")

    # Per-ETF stats
    abs_sigs  = np.abs(signals[:, :N])
    total     = abs_sigs.sum(axis=1, keepdims=True).clip(min=1e-8)
    weights   = abs_sigs / total
    avg_alloc = {cfg.TICKERS[i]: round(float(weights[:, i].mean() * 100), 2)
                 for i in range(N)}

    signs     = np.sign(signals[:, :N])
    buy_ratio = {cfg.TICKERS[i]: round(float((signs[:, i] > 0).mean() * 100), 2)
                 for i in range(N)}

    # Raw tanh output statistics per ETF
    # High mean + low std = consistent conviction
    # Low mean + high std = inconsistent / fluke
    raw_outputs = signals[:, :N]
    output_stats = {}
    for i, ticker in enumerate(cfg.TICKERS):
        vals = raw_outputs[:, i]
        output_stats[ticker] = {
            "mean":    round(float(vals.mean()), 4),
            "std":     round(float(vals.std()), 4),
            "min":     round(float(vals.min()), 4),
            "max":     round(float(vals.max()), 4),
            "pct_above_02":  round(float((vals > 0.2).mean() * 100), 2),   # % days clear BUY signal
            "pct_below_m02": round(float((vals < -0.2).mean() * 100), 2),  # % days clear SHORT signal
        }

    # Single ETF B&H — use the model's most allocated ETF over the test period
    top_ticker_idx  = int(np.argmax([avg_alloc.get(t, 0) for t in cfg.TICKERS]))
    top_ticker_name = cfg.TICKERS[top_ticker_idx]
    single_bh       = buy_and_hold_single(price_window, top_ticker_idx)
    single_bh_metrics = compute_metrics(single_bh)
    print(f"     Top ETF B&H  : {top_ticker_name} → "
          f"{single_bh_metrics['total_return_pct']:.2f}% total / "
          f"{single_bh_metrics['annual_return_pct']:.2f}% CAGR")

    return {
        "metrics":              metrics,
        "bh_metrics":           bh_metrics,
        "avg_alloc_pct":        avg_alloc,
        "buy_ratio_pct":        buy_ratio,
        "output_stats":         output_stats,
        "portfolio_values":     portfolio.tolist(),
        "bh_values":            bh.tolist(),
        "single_etf_bh": {
            "ticker":           top_ticker_name,
            "metrics":          single_bh_metrics,
            "portfolio_values": single_bh.tolist(),
        },
        "weight_file":          os.path.basename(weight_path),
    }


# ── Update performance history ────────────────────────────────────────────────

def update_performance_history(results_dir: str, today_str: str, results: dict):
    history_path = os.path.join(results_dir, "performance_history.json")
    history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

    entry = {
        "run_date":     today_str,
        "test_period":  results.get("test_period"),
        "buy_and_hold": results.get("buy_and_hold", {}).get("metrics", {}),
        "models": {
            m: results["models"][m].get("metrics", {})
            for m in results.get("models", {})
        }
    }

    history = [e for e in history if e.get("run_date") != today_str]
    history.append(entry)
    history.sort(key=lambda x: x.get("run_date", ""))

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  📈 performance_history.json updated ({len(history)} runs recorded)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ETF models")
    parser.add_argument("--module", choices=["A", "B"], required=True)
    args = parser.parse_args()

    today_str = datetime.utcnow().strftime("%Y%m%d")

    cfg_map = {"A": "config_equity", "B": "config_fixed_income"}
    cfg     = importlib.import_module(cfg_map[args.module])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥  Device    : {device}")
    print(f"📦 Module    : {cfg.MODULE} — {cfg.LABEL}")
    print(f"📅 Run date  : {today_str}")

    token = os.getenv("HF_TOKEN")

    # Load full dataset to extract raw test prices and features
    from huggingface_hub import hf_hub_download
    import pandas as pd

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

    n          = len(features_df)
    test_size  = max(int(n * cfg.SPLIT_TEST_RATIO), cfg.SEQ_LEN + 10)
    val_size   = max(int(n * cfg.SPLIT_VAL_RATIO),  cfg.SEQ_LEN + 10)
    train_size = n - val_size - test_size

    # Test period raw data (unscaled features + raw prices)
    test_features = features_df.iloc[train_size + val_size :].values
    test_prices   = prices_df.iloc[train_size + val_size :].values

    test_start = features_df.index[train_size + val_size].date()
    test_end   = features_df.index[-1].date()
    test_period = f"{test_start} → {test_end} ({len(test_features)} rows)"
    print(f"📅 Test period: {test_period}")

    # Load scaler (fit on train)
    scaler_path = latest_dated_file(cfg.RESULTS_DIR, "scaler", ".pkl")
    if not scaler_path:
        print("❌ No scaler found. Run train.py first.")
        return
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"📐 Loaded scaler from {os.path.basename(scaler_path)}")

    # Align Buy & Hold to the same window as model signals
    # Models start after seq_len rows (need full lookback window first)
    # so Buy & Hold must start from the same point for fair comparison
    aligned_prices = test_prices[cfg.SEQ_LEN - 1:]
    bh_portfolio   = buy_and_hold(aligned_prices)
    bh_metrics     = compute_metrics(bh_portfolio)
    print(f"\n📊 Buy & Hold baseline ({test_period}, aligned to model window):")
    print(f"   Trading days  : {bh_metrics['n_days']}")
    print(f"   Total Return  : {bh_metrics['total_return_pct']:.2f}%")
    print(f"   CAGR          : {bh_metrics['annual_return_pct']:.2f}%")
    print(f"   Sharpe Ratio  : {bh_metrics['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown  : {bh_metrics['max_drawdown_pct']:.2f}%")

    results = {
        "module":       cfg.MODULE,
        "label":        cfg.LABEL,
        "test_period":  test_period,
        "run_date":     today_str,
        "tickers":      cfg.TICKERS,
        "evaluated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "buy_and_hold": {
            "metrics":          bh_metrics,
            "portfolio_values": bh_portfolio.tolist(),
        },
        "models": {},
    }

    variants = getattr(cfg, 'MODEL_VARIANTS',
                       [("dlinear","L2","PRC"),("crossformer","L2","PRC")])
    for arch, loss_variant, loss_type in variants:
        variant_name = f"{arch}_{loss_type.lower()}"
        print(f"\n{'─'*45}")
        print(f"  {variant_name.upper()}")
        print(f"{'─'*45}")
        r = evaluate_model(variant_name, cfg, test_prices,
                           test_features, scaler, device)
        if r:
            results["models"][variant_name] = r

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(cfg.RESULTS_DIR, f"eval_results_{today_str}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved eval results → {out_path}")

    update_performance_history(cfg.RESULTS_DIR, today_str, results)

    # Archive this run's results for paper comparison
    try:
        from train import archive_results
        archive_results(cfg, today_str)
    except Exception as e:
        print(f"  ⚠️  Archive skipped: {e}")

    print(f"\n🎉 Evaluation complete for Module {cfg.MODULE} [{today_str}]")


if __name__ == "__main__":
    main()
