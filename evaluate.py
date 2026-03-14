"""
evaluate.py — P2-ETF-DLINEAR-CROSS
=====================================
Backtests trained models on the test year and computes:
  - Annual return %
  - Sharpe ratio
  - Max drawdown
  - Per-ETF allocation heatmap data
  - Comparison vs Buy & Hold baseline

Usage:
  python evaluate.py --module A
  python evaluate.py --module B

Saves results/{equity|fixed_income}/eval_results.json
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
from datetime import datetime

# Ensure repo root is on path regardless of how the script is invoked
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from model import get_model
from loss_functions import smooth_sign


# ── Portfolio simulation ──────────────────────────────────────────────────────

def simulate_portfolio(signals: np.ndarray, price_diffs: np.ndarray,
                        initial_budget: float = 10_000.0) -> np.ndarray:
    """
    Simulate daily portfolio value given signals and price differences.

    signals     : (T, N+1)  — model outputs after tanh, last col = Hold
    price_diffs : (T, N)    — PRC_{t+1} - PRC_t
    Returns     : (T,) portfolio value series
    """
    N      = price_diffs.shape[1]
    budget = initial_budget
    values = [budget]

    for t in range(len(signals)):
        O   = signals[t]                   # (N+1,)
        O_n = O[:N]
        abs_O = np.abs(O)
        total = abs_O.sum()
        if total < 1e-8:
            values.append(budget)
            continue

        V    = abs_O / total               # normalised weights (N+1,)
        V_n  = V[:N]                       # trading weights
        sign = np.sign(O_n)                # buy/short decision

        # Daily P&L: Σ V_i * sign(O_i) * (PRC_{t+1} - PRC_t) / PRC_t
        # Approximate return using price diff / prev price (not available here)
        # Use raw price diff contribution weighted by allocation
        pnl    = np.sum(V_n * sign * price_diffs[t])
        budget = budget + pnl
        values.append(max(budget, 0.0))    # floor at 0

    return np.array(values)


def compute_metrics(portfolio_values: np.ndarray, initial: float = 10_000.0) -> dict:
    """Compute Sharpe ratio, max drawdown, and annual return."""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    returns = returns[~np.isnan(returns)]

    annual_return = (portfolio_values[-1] / initial - 1.0) * 100.0

    sharpe = 0.0
    if returns.std() > 1e-8:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    roll_max   = np.maximum.accumulate(portfolio_values)
    drawdowns  = (portfolio_values - roll_max) / roll_max
    max_dd     = float(drawdowns.min()) * 100.0

    return {
        "annual_return_pct": round(float(annual_return), 4),
        "sharpe_ratio":      round(float(sharpe), 4),
        "max_drawdown_pct":  round(float(max_dd), 4),
        "final_value":       round(float(portfolio_values[-1]), 2),
        "initial_value":     round(float(initial), 2),
    }


def buy_and_hold(price_diffs: np.ndarray, initial: float = 10_000.0) -> np.ndarray:
    """Equal-weight buy and hold baseline."""
    N      = price_diffs.shape[1]
    weight = 1.0 / N
    budget = initial
    values = [budget]
    for t in range(len(price_diffs)):
        pnl    = np.sum(weight * price_diffs[t])
        budget = budget + pnl
        values.append(max(budget, 0.0))
    return np.array(values)


# ── Get signals from model ────────────────────────────────────────────────────

def get_signals(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run model over test set, return (signals, price_diffs)."""
    all_signals = []
    all_diffs   = []
    model.eval()
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            O = model(X)                   # (batch, N+1)
            all_signals.append(O.cpu().numpy())
            all_diffs.append(Y.numpy())
    return np.vstack(all_signals), np.vstack(all_diffs)


# ── Evaluate one model ────────────────────────────────────────────────────────

def evaluate_model(model_name: str, cfg, test_loader, device) -> dict:
    weight_path = os.path.join(cfg.RESULTS_DIR, f"{model_name}_best.pt")
    if not os.path.exists(weight_path):
        print(f"  ⚠️  No weights found for {model_name} at {weight_path}. Skipping.")
        return {}

    model = get_model(model_name, cfg).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    print(f"  ✅ Loaded {model_name} weights from {weight_path}")

    signals, price_diffs = get_signals(model, test_loader, device)

    portfolio = simulate_portfolio(signals, price_diffs)
    metrics   = compute_metrics(portfolio)

    print(f"     Annual Return : {metrics['annual_return_pct']:.2f}%")
    print(f"     Sharpe Ratio  : {metrics['sharpe_ratio']:.3f}")
    print(f"     Max Drawdown  : {metrics['max_drawdown_pct']:.2f}%")

    # Per-ETF average allocation
    N        = cfg.N_ASSETS
    abs_sigs = np.abs(signals[:, :N])
    total    = abs_sigs.sum(axis=1, keepdims=True).clip(min=1e-8)
    weights  = abs_sigs / total
    avg_alloc = {cfg.TICKERS[i]: round(float(weights[:, i].mean() * 100), 2)
                 for i in range(N)}

    # Buy/short ratio per ETF
    signs     = np.sign(signals[:, :N])
    buy_ratio = {cfg.TICKERS[i]: round(float((signs[:, i] > 0).mean() * 100), 2)
                 for i in range(N)}

    return {
        "metrics":       metrics,
        "avg_alloc_pct": avg_alloc,
        "buy_ratio_pct": buy_ratio,
        "portfolio_values": portfolio.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained ETF models")
    parser.add_argument(
        "--module", choices=["A", "B"], required=True,
        help="A = Equity ETFs, B = Fixed Income/Commodity ETFs"
    )
    args = parser.parse_args()

    cfg_map = {"A": "config_equity", "B": "config_fixed_income"}
    cfg     = importlib.import_module(cfg_map[args.module])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥  Device: {device}")
    print(f"📦 Evaluating Module {cfg.MODULE}: {cfg.LABEL}")

    token = os.getenv("HF_TOKEN")
    _, _, test_loader, _, _ = load_data(
        cfg, seq_len=cfg.SEQ_LEN, batch_size=cfg.BATCH_SIZE, token=token
    )

    # Buy & Hold baseline
    all_diffs = []
    for _, Y in test_loader:
        all_diffs.append(Y.numpy())
    all_diffs = np.vstack(all_diffs)

    bh_portfolio = buy_and_hold(all_diffs)
    bh_metrics   = compute_metrics(bh_portfolio)
    print(f"\n📊 Buy & Hold baseline:")
    print(f"   Annual Return : {bh_metrics['annual_return_pct']:.2f}%")
    print(f"   Sharpe Ratio  : {bh_metrics['sharpe_ratio']:.3f}")
    print(f"   Max Drawdown  : {bh_metrics['max_drawdown_pct']:.2f}%")

    results = {
        "module":          cfg.MODULE,
        "label":           cfg.LABEL,
        "test_year":       cfg.TEST_YEAR,
        "tickers":         cfg.TICKERS,
        "evaluated_at":    datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "buy_and_hold":    {
            "metrics": bh_metrics,
            "portfolio_values": bh_portfolio.tolist(),
        },
        "models": {},
    }

    for model_name in ["dlinear", "crossformer"]:
        print(f"\n{'─'*45}")
        print(f"  {model_name.upper()}")
        print(f"{'─'*45}")
        r = evaluate_model(model_name, cfg, test_loader, device)
        if r:
            results["models"][model_name] = r

    # Save results
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(cfg.RESULTS_DIR, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved evaluation results → {out_path}")
    print(f"🎉 Evaluation complete for Module {cfg.MODULE}")


if __name__ == "__main__":
    main()
