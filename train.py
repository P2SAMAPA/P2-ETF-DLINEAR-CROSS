"""
train.py — P2-ETF-DLINEAR-CROSS
==================================
Training loop for Option A (Equity) and Option B (Fixed Income).
Trains both DLinear and Crossformer with StockLoss-L2 PRC.

Usage:
  python train.py --module A
  python train.py --module B

Saves best model weights (by val loss) to results/{equity|fixed_income}/
"""

import os
import json
import argparse
import importlib
import numpy as np
import torch
import torch.optim as optim
from datetime import datetime
from copy import deepcopy

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
        O    = model(X)                         # (batch, N+1)
        loss = loss_fn(O, Y, gamma=cfg.GAMMA)
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
            loss = loss_fn(O, Y, gamma=cfg.GAMMA)
            total_loss += loss.item()
    return total_loss / len(loader)


# ── Train one model ───────────────────────────────────────────────────────────

def train_model(model_name: str, cfg, train_loader, val_loader, device) -> dict:
    """Train a single model and return best weights + metrics."""
    print(f"\n{'='*55}")
    print(f"  Training {model_name.upper()} — Module {cfg.MODULE} ({cfg.LABEL})")
    print(f"{'='*55}")

    model   = get_model(model_name, cfg).to(device)
    loss_fn = get_loss_fn("L2")
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=False
    )

    best_val_loss   = float("inf")
    best_state      = None
    patience_count  = 0
    early_stop      = 20    # stop if no improvement for 20 epochs
    history         = {"train": [], "val": []}

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
                print(f"  ⏹  Early stopping at epoch {epoch} (no improvement for {early_stop} epochs)")
                break

    print(f"  ✅ Best val loss: {best_val_loss:.4f}")
    return {
        "state_dict": best_state,
        "best_val_loss": best_val_loss,
        "history": history,
        "epochs_trained": len(history["train"]),
    }


# ── Save model ────────────────────────────────────────────────────────────────

def save_model(model_name: str, result: dict, cfg):
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    weight_path = os.path.join(cfg.RESULTS_DIR, f"{model_name}_best.pt")
    torch.save(result["state_dict"], weight_path)
    print(f"  💾 Saved weights → {weight_path}")

    meta = {
        "model":          model_name,
        "module":         cfg.MODULE,
        "label":          cfg.LABEL,
        "best_val_loss":  result["best_val_loss"],
        "epochs_trained": result["epochs_trained"],
        "trained_at":     datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "test_year":      cfg.TEST_YEAR,
        "val_year":       cfg.VAL_YEAR,
        "tickers":        cfg.TICKERS,
        "loss_fn":        "StockLoss-L2-PRC",
        "seq_len":        cfg.SEQ_LEN,
        "history":        result["history"],
    }
    meta_path = os.path.join(cfg.RESULTS_DIR, f"{model_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  📝 Saved metadata → {meta_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ETF trading models")
    parser.add_argument(
        "--module", choices=["A", "B"], required=True,
        help="A = Equity ETFs, B = Fixed Income/Commodity ETFs"
    )
    args = parser.parse_args()

    # Dynamically load config
    cfg_map = {"A": "config_equity", "B": "config_fixed_income"}
    cfg     = importlib.import_module(cfg_map[args.module])

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥  Device: {device}")
    print(f"📦 Module {cfg.MODULE}: {cfg.LABEL}")
    print(f"📈 Tickers: {cfg.TICKERS}")

    # Load data
    token = os.getenv("HF_TOKEN")
    train_loader, val_loader, test_loader, n_features, scaler = load_data(
        cfg,
        seq_len    = cfg.SEQ_LEN,
        batch_size = cfg.BATCH_SIZE,
        token      = token,
    )

    # Train both models
    for model_name in ["dlinear", "crossformer"]:
        result = train_model(model_name, cfg, train_loader, val_loader, device)
        save_model(model_name, result, cfg)

    # Save scaler for inference
    import pickle
    scaler_path = os.path.join(cfg.RESULTS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n📐 Saved scaler → {scaler_path}")

    print(f"\n🎉 Training complete for Module {cfg.MODULE}")


if __name__ == "__main__":
    main()
