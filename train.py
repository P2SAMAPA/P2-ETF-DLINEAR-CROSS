"""
train.py — P2-ETF-DLINEAR-CROSS
==================================
Trains all 4 model variants per module:
  1. DLinear     + StockLoss-L2 PRC
  2. Crossformer + StockLoss-L2 PRC
  3. DLinear     + StockLoss-L2 RET
  4. Crossformer + StockLoss-L2 RET

Usage:
  python train.py --module A
  python train.py --module B
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
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from model import get_model
from loss_functions import get_loss_fn


# ── Training one epoch ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, cfg, device, loss_type):
    model.train()
    total_loss = 0.0
    for X, prc_diff, ret_diff in loader:
        X        = X.to(device)
        Y        = prc_diff.to(device) if loss_type == "PRC" else ret_diff.to(device)
        optimizer.zero_grad()
        O    = model(X)
        loss = loss_fn(O, Y, gamma=cfg.GAMMA,
                       use_hold=getattr(cfg, 'USE_HOLD', False))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, loss_fn, cfg, device, loss_type):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, prc_diff, ret_diff in loader:
            X = X.to(device)
            Y = prc_diff.to(device) if loss_type == "PRC" else ret_diff.to(device)
            O    = model(X)
            loss = loss_fn(O, Y, gamma=cfg.GAMMA,
                           use_hold=getattr(cfg, 'USE_HOLD', False))
            total_loss += loss.item()
    return total_loss / len(loader)


# ── Train one variant ─────────────────────────────────────────────────────────

def train_variant(arch: str, loss_type: str, cfg,
                  train_loader, val_loader, device) -> dict:
    variant_name = f"{arch}_{loss_type.lower()}"
    print(f"\n{'='*55}")
    print(f"  Training {variant_name.upper()} — Module {cfg.MODULE}")
    print(f"{'='*55}")

    model     = get_model(arch, cfg).to(device)
    loss_fn   = get_loss_fn("L2")
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0
    early_stop     = 15
    history        = {"train": [], "val": []}

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer,
                                 loss_fn, cfg, device, loss_type)
        val_loss   = eval_epoch(model, val_loader,
                                loss_fn, cfg, device, loss_type)
        scheduler.step(val_loss)
        history["train"].append(round(train_loss, 6))
        history["val"].append(round(val_loss, 6))

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg.EPOCHS} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= early_stop:
                print(f"  ⏹  Early stop at epoch {epoch}")
                break

    print(f"  ✅ Best val loss: {best_val_loss:.4f}")
    return {
        "state_dict":     best_state,
        "best_val_loss":  best_val_loss,
        "history":        history,
        "epochs_trained": len(history["train"]),
    }


# ── Cleanup old dated files ───────────────────────────────────────────────────

def cleanup_old_files(results_dir: str, today_str: str):
    keep = {"performance_history.json"}
    deleted = []
    for fname in os.listdir(results_dir):
        if fname in keep:
            continue
        parts = fname.rsplit("_", 1)
        if len(parts) == 2:
            date_part = parts[1].split(".")[0]
            if date_part.isdigit() and len(date_part) == 8 and date_part != today_str:
                os.remove(os.path.join(results_dir, fname))
                deleted.append(fname)
    if deleted:
        print(f"  🧹 Cleaned {len(deleted)} old file(s)")


# ── Save variant ──────────────────────────────────────────────────────────────

def save_variant(arch: str, loss_type: str, result: dict,
                 cfg, today_str: str):
    variant_name = f"{arch}_{loss_type.lower()}"
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    weight_path = os.path.join(cfg.RESULTS_DIR,
                               f"{variant_name}_best_{today_str}.pt")
    torch.save(result["state_dict"], weight_path)
    print(f"  💾 {weight_path}")

    meta = {
        "model":          arch,
        "loss_type":      loss_type,
        "variant":        variant_name,
        "module":         cfg.MODULE,
        "label":          cfg.LABEL,
        "best_val_loss":  result["best_val_loss"],
        "epochs_trained": result["epochs_trained"],
        "trained_at":     datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "run_date":       today_str,
        "tickers":        cfg.TICKERS,
        "loss_fn":        f"StockLoss-L2-{loss_type}",
        "seq_len":        cfg.SEQ_LEN,
        "history":        result["history"],
    }
    meta_path = os.path.join(cfg.RESULTS_DIR,
                             f"{variant_name}_meta_{today_str}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  📝 {meta_path}")


# ── Archive snapshot ─────────────────────────────────────────────────────────

def get_phase_label(cfg) -> str:
    """
    Auto-generate a phase label from config settings.
    Used to name the archive folder.
    e.g. "phase2a_ep50_test10_withXES" or "phase3_ep300_test15_noXES"
    """
    epochs    = cfg.EPOCHS
    test_pct  = int(cfg.SPLIT_TEST_RATIO * 100)
    has_xes   = "XES" in cfg.TICKERS if hasattr(cfg, 'TICKERS') else True
    xes_label = "withXES" if has_xes else "noXES"
    return f"ep{epochs}_test{test_pct}_{xes_label}"


def archive_results(cfg, today_str: str):
    """
    Copy today's dated result files into results/archive/{phase_label}/
    so they are preserved when the next phase overwrites the main results folder.
    Copies: eval_results, all variant metas, scaler meta (not .pt weights — too large)
    """
    import shutil
    phase_label  = get_phase_label(cfg)
    module_label = f"module_{cfg.MODULE}"
    archive_dir  = os.path.join("results", "archive", phase_label, module_label)
    os.makedirs(archive_dir, exist_ok=True)

    copied = []
    for fname in os.listdir(cfg.RESULTS_DIR):
        # Archive JSON files and pkl (small) — skip .pt weight files (large)
        if fname.endswith(".json") or fname == f"scaler_{today_str}.pkl":
            src = os.path.join(cfg.RESULTS_DIR, fname)
            dst = os.path.join(archive_dir, fname)
            shutil.copy2(src, dst)
            copied.append(fname)

    # Write a phase README
    readme_path = os.path.join(archive_dir, "PHASE_INFO.md")
    with open(readme_path, "w") as f:
        f.write(f"# Archive: {phase_label} — Module {cfg.MODULE}

")
        f.write(f"**Run date**: {today_str}

")
        f.write(f"**Epochs**: {cfg.EPOCHS}

")
        f.write(f"**Test ratio**: {cfg.SPLIT_TEST_RATIO} ({int(cfg.SPLIT_TEST_RATIO*100)}%)

")
        f.write(f"**Val ratio**: {cfg.SPLIT_VAL_RATIO} ({int(cfg.SPLIT_VAL_RATIO*100)}%)

")
        f.write(f"**Tickers**: {cfg.TICKERS}

")
        f.write(f"**Loss function**: StockLoss-L2 (PRC + RET variants)

")
        f.write(f"**Use Hold node**: {getattr(cfg, 'USE_HOLD', False)}

")
        f.write(f"**Files archived**: {copied}

")
        f.write("## Notes

")
        if "XES" in cfg.TICKERS:
            f.write("- XES included in equity universe
")
        else:
            f.write("- XES **excluded** from equity universe (Phase 3)
")

    print(f"  📦 Archived {len(copied)} files → {archive_dir}")
    print(f"  📝 Phase info → {readme_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["A", "B"], required=True)
    args = parser.parse_args()

    today_str = datetime.utcnow().strftime("%Y%m%d")
    cfg_map   = {"A": "config_equity", "B": "config_fixed_income"}
    cfg       = importlib.import_module(cfg_map[args.module])
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🖥  Device   : {device}")
    print(f"📦 Module   : {cfg.MODULE} — {cfg.LABEL}")
    print(f"📅 Run date : {today_str}")
    print(f"🔢 Variants : {cfg.MODEL_VARIANTS}")

    token = os.getenv("HF_TOKEN")
    train_loader, val_loader, test_loader, n_features, scaler = load_data(
        cfg, seq_len=cfg.SEQ_LEN, batch_size=cfg.BATCH_SIZE, token=token
    )

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    cleanup_old_files(cfg.RESULTS_DIR, today_str)

    # Train all variants
    for arch, loss_variant, loss_type in cfg.MODEL_VARIANTS:
        result = train_variant(arch, loss_type, cfg,
                               train_loader, val_loader, device)
        save_variant(arch, loss_type, result, cfg, today_str)

    # Save scaler
    scaler_path = os.path.join(cfg.RESULTS_DIR, f"scaler_{today_str}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"\n📐 Saved scaler → {scaler_path}")
    print(f"\n🎉 All variants trained for Module {cfg.MODULE} [{today_str}]")


if __name__ == "__main__":
    main()
