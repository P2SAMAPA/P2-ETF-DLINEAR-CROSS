"""
archive_current.py - ONE-TIME script
=====================================
Run this ONCE manually before triggering Phase 2b (300-epoch) training
to preserve the current Phase 2a results (50 epochs, test=0.10, with XES).

Usage:
  python archive_current.py
"""

import os
import sys
import shutil
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PHASE_LABEL = "ep50_test10_withXES"

MODULE_CONFIG = {
    "A": {
        "results_dir": "results/equity",
        "tickers": ["SPY","QQQ","XLK","XLF","XLE","XLV","XLI","GDX","IWM","XES"],
        "epochs": 50,
        "test_ratio": 0.10,
    },
    "B": {
        "results_dir": "results/fixed_income",
        "tickers": ["TLT","VNQ","GLD","SLV","LQD","HYG","MBB","PFF"],
        "epochs": 50,
        "test_ratio": 0.10,
    },
}

def archive_module(module: str, cfg: dict):
    src_dir   = cfg["results_dir"]
    arch_dir  = os.path.join("results", "archive", PHASE_LABEL, f"module_{module}")
    os.makedirs(arch_dir, exist_ok=True)

    copied = []
    for fname in os.listdir(src_dir):
        # Copy JSON files only - weights too large for archive
        if fname.endswith(".json"):
            shutil.copy2(os.path.join(src_dir, fname),
                        os.path.join(arch_dir, fname))
            copied.append(fname)

    # Write phase README
    with open(os.path.join(arch_dir, "PHASE_INFO.md"), "w") as f:
        f.write(f"# Archive: {PHASE_LABEL} - Module {module}\n\n")
        f.write(f"**Archived on**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Epochs**: {cfg['epochs']}\n\n")
        f.write(f"**Test ratio**: {cfg['test_ratio']} ({int(cfg['test_ratio']*100)}%)\n\n")
        f.write(f"**Tickers**: {cfg['tickers']}\n\n")
        f.write(f"**Phase**: 2a - initial run with XES, lower epochs\n\n")
        f.write(f"**Files**: {copied}\n\n")
        f.write("## Key Results\n\n")
        f.write("See eval_results_*.json for full backtest metrics.\n\n")
        f.write("### Notes\n")
        f.write("- XES included in equity universe\n")
        f.write("- First run proving RET > PRC for ETFs\n")
        f.write("- Buy & Hold alignment bug present (fixed in Phase 2b)\n")

    print(f"  ✅ Module {module}: {len(copied)} files → {arch_dir}")
    return copied

def main():
    print("=" * 55)
    print(f"Archiving Phase 2a results: {PHASE_LABEL}")
    print("=" * 55)

    for module, cfg in MODULE_CONFIG.items():
        print(f"\n--- Module {module} ---")
        if not os.path.exists(cfg["results_dir"]):
            print(f"  ⚠️  {cfg['results_dir']} not found, skipping")
            continue
        archive_module(module, cfg)

    print(f"\n✅ Archive complete → results/archive/{PHASE_LABEL}/")
    print("Commit this to git before running Phase 2b training!")

if __name__ == "__main__":
    main()
