"""
train.py — P2-ETF-DLINEAR-CROSS
==================================
Training loop for Option A (Equity) and Option B (Fixed Income).
Trains both DLinear and Crossformer with StockLoss-L2 PRC.

Usage:
  python train.py --module A
  python train.py --module B

Saves date-stamped model weights to results/{equity|fixed_income}/
Cleans up previous date files (keeps performance_history.json always).
"""

import os
import sys
import json
import argparse
import importlib
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from copy import deepcopy

# Ensure repo root is on path regardless of how the script is invoked
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from model import get_model
from loss_functions import get_loss_fn


# ── Training one epoch ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, cfg, device):
    model.train()
    total_loss = 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        O    = model(X)
        loss = loss_fn(O, Y, gamma=cfg.GAMMA, use_hold=getattr(cfg, 'USE_HOLD', False))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, loss_fn, cfg, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            O    = model(X)
            loss = loss_fn(O, Y, gamma=cfg.GAMMA, use_hold=getattr(cfg, 'USE_HOLD', False))
            total_loss += loss.item()
    return total_loss / len(loader)


# ── Train one model ───────────────────────────────────────────────────────────

def train_model(model_name: str, cfg, train_loader, val_loader, device) -> dict:
    print(f"\n{'='*55}")
    print(f"  Training {model_name.upper()} — Module {cfg.MODULE} ({cfg.LABEL})")
    print(f"{'='*55}")

    model     = get_model(model_name, cfg).to(device)
    loss_fn   = get_loss_fn("L2")
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    early_stop     = 20
    history        = {"train": [], "val": []}

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, cfg, device)
        val_loss   = eval_epoch(model,   val_loader,             loss_fn, cfg, device)
        scheduler.step(val_loss)
        history["train"].append(round(train_loss, 6))
        history["val"].append(round(val_loss, 6))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg.EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= early_stop:
                print(f"  ⏹  Early stopping at epoch {epoch}")
                break

    print(f"  ✅ Best val loss: {best_val_loss:.4f}")
    return {
        "state_dict":     best_state,
        "best_val_loss":  best_val_loss,
        "history":        history,
        "epochs_trained": len(history["train"]),
    }


# ── Cleanup old date-stamped files ────────────────────────────────────────────

def cleanup_old_files(results_dir: str, today_str: str):
    """
    Remove date-stamped files from previous runs.
    Keeps: performance_history.json and today's files.
    Removes: anything with _YYYYMMDD suffix that isn't today.
    """
    keep = {"performance_history.json"}
    deleted = []
    for fname in os.listdir(results_dir):
        if fname in keep:
            continue
        parts = fname.rsplit("_", 1)
        if len(parts) == 2:
            date_part = parts[1].split(".")[0]
            if date_part.isdigit() and len(date_part) == 8 and date_part != today_str:
                fpath = os.path.join(results_dir, fname)
                os.remove(fpath)
                deleted.append(fname)
    if deleted:
        print(f"  🧹 Cleaned {len(deleted)} old file(s): {deleted}")
    else:
        print(f"  🧹 No old files to clean up")


# ── Save model ────────────────────────────────────────────────────────────────

def save_model(model_name: str, result: dict, cfg, today_str: str):
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    weight_path = os.path.join(cfg.RESULTS_DIR, f"{model_name}_best_{today_str}.pt")
    torch.save(result["state_dict"], weight_path)
    print(f"  💾 Saved weights → {weight_path}")

    meta = {
        "model":          model_name,
        "module":         cfg.MODULE,
        "label":          cfg.LABEL,
        "best_val_loss":  result["best_val_loss"],
        "epochs_trained": result["epochs_trained"],
        "trained_at":     datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "run_date":       today_str,
        "test_year":      cfg.TEST_YEAR,
        "val_year":       cfg.VAL_YEAR,
        "tickers":        cfg.TICKERS,
        "loss_fn":        "StockLoss-L2-PRC",
        "seq_len":        cfg.SEQ_LEN,
        "history":        result["history"],
    }
    meta_path = os.path.join(cfg.RESULTS_DIR, f"{model_name}_meta_{today_str}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  📝 Saved metadata → {meta_path}")


# ── Update performance history ────────────────────────────────────────────────

def update_performance_history(results_dir: str, today_str: str, eval_path: str):
    """
    Append today's eval metrics to the permanent performance_history.json.
    This file is never deleted — it accumulates every run's returns.
    """
    history_path = os.path.join(results_dir, "performance_history.json")

    # Load existing history
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = []

    # Load today's eval results
    if not os.path.exists(eval_path):
        print(f"  ⚠️  No eval results found at {eval_path}, skipping history update")
        return

    with open(eval_path) as f:
        eval_data = json.load(f)

    # Build history entry
    entry = {
        "run_date":    today_str,
        "test_year":   eval_data.get("test_year"),
        "buy_and_hold": eval_data.get("buy_and_hold", {}).get("metrics", {}),
        "models":      {
            m: eval_data["models"][m].get("metrics", {})
            for m in eval_data.get("models", {})
        }
    }

    # Replace entry for today if already exists, else append
    history = [e for e in history if e.get("run_date") != today_str]
    history.append(entry)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  📈 Updated performance_history.json ({len(history)} runs recorded)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ETF trading models")
    parser.add_argument(
        "--module", choices=["A", "B"], required=True,
        help="A = Equity ETFs, B = Fixed Income/Commodity ETFs"
    )
    args = parser.parse_args()

    today_str = datetime.utcnow().strftime("%Y%m%d")

    cfg_map = {"A": "config_equity", "B": "config_fixed_income"}
    cfg     = importlib.import_module(cfg_map[args.module])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥  Device     : {device}")
    print(f"📦 Module     : {cfg.MODULE} — {cfg.LABEL}")
    print(f"📅 Run date   : {today_str}")
    print(f"📈 Tickers    : {cfg.TICKERS}")

    token = os.getenv("HF_TOKEN")
    train_loader, val_loader, test_loader, n_features, scaler = load_data(
        cfg,
        seq_len    = cfg.SEQ_LEN,
        batch_size = cfg.BATCH_SIZE,
        token      = token,
    )

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # Clean up old dated files before saving new ones
    cleanup_old_files(cfg.RESULTS_DIR, today_str)

    # Train both models
    for model_name in ["dlinear", "crossformer"]:
        result = train_model(model_name, cfg, train_loader, val_loader, device)
        save_model(model_name, result, cfg, today_str)

    # Save scaler with date stamp
    import pickle
    scaler_path = os.path.join(cfg.RESULTS_DIR, f"scaler_{today_str}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n📐 Saved scaler → {scaler_path}")

    print(f"\n🎉 Training complete for Module {cfg.MODULE} [{today_str}]")


if __name__ == "__main__":
    main()
