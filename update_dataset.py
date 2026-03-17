"""
update_dataset.py — P2-ETF-DLINEAR-CROSS — DEBUG VERSION
"""

import os
import sys
import json
import time
import random
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from huggingface_hub import HfApi, hf_hub_download, whoami

print(f"🐍 Python: {sys.version}")
print(f"⏰ Start: {datetime.utcnow()}")
print(f"📦 pandas: {pd.__version__}")

# ── Config ────────────────────────────────────────────────────────────────────
HF_DATASET_REPO = "P2SAMAPA/etf-dlinear-cross-data"
OHLCV_FIELDS = ["Open", "High", "Low", "Close", "Volume"]

MODULE_CONFIG = {
    "A": {
        "tickers": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "GDX", "IWM", "XES"],
        "hf_path_parquet": "equity/ohlcv_equity.parquet",
        "hf_path_metadata": "equity/metadata_equity.json",
        "label": "Equity ETFs",
    },
    "B": {
        "tickers": ["TLT", "VNQ", "GLD", "SLV", "LQD", "HYG", "MBB", "PFF"],
        "hf_path_parquet": "fixed_income/ohlcv_fixed_income.parquet",
        "hf_path_metadata": "fixed_income/metadata_fixed_income.json",
        "label": "Fixed Income / Commodity ETFs",
    },
}


def test_hf_auth(token):
    """Test HF authentication."""
    print(f"\n🔑 Testing HF auth (token len: {len(token) if token else 0})...")
    try:
        user = whoami(token=token)
        print(f"   ✅ Auth OK: {user['name']}")
        return True
    except Exception as e:
        print(f"   ❌ Auth FAILED: {e}")
        return False


def load_from_hf(hf_path, token):
    """Download from HF."""
    print(f"\n📥 Downloading: {hf_path}")
    try:
        local = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=hf_path,
            token=token,
            force_download=True,
        )
        print(f"   ✅ Saved to: {local}")
        df = pd.read_parquet(local)
        print(f"   ✅ Loaded: {df.shape}")
        return df
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        raise


def upload_with_debug(df, repo_path, token, msg):
    """Upload with full debug."""
    print(f"\n📤 Uploading: {repo_path}")
    print(f"   DataFrame: {df.shape}")
    
    temp_file = f"/tmp/debug_{random.randint(1000,9999)}.parquet"
    
    try:
        df.to_parquet(temp_file, index=True)
        print(f"   ✅ Saved temp: {os.path.getsize(temp_file)} bytes")
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
        raise
    
    api = HfApi(token=token)
    
    print(f"   Calling upload_file...")
    try:
        result = api.upload_file(
            path_or_fileobj=temp_file,
            path_in_repo=repo_path,
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=msg,
        )
        print(f"   ✅ SUCCESS: {result}")
        return True
    except Exception as e:
        print(f"   ❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def update_module(module, token):
    """Update with debug."""
    cfg = MODULE_CONFIG[module]
    print(f"\n{'='*60}")
    print(f"MODULE {module}: {cfg['label']}")
    print(f"{'='*60}")
    
    # Test auth
    if not test_hf_auth(token):
        raise RuntimeError("Auth failed")
    
    # Load existing
    existing = load_from_hf(cfg["hf_path_parquet"], token)
    
    # DEBUG: Just re-upload to test
    print(f"\n🧪 TEST: Re-uploading existing data...")
    success = upload_with_debug(
        existing,
        cfg["hf_path_parquet"],
        token,
        f"DEBUG TEST {datetime.utcnow():%Y-%m-%d %H:%M:%S}"
    )
    
    if success:
        print(f"\n✅ MODULE {module} SUCCESS")
    else:
        print(f"\n❌ MODULE {module} FAILED")
    
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["A", "B"], default=None)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    print(f"HF_TOKEN present: {bool(token)}")
    
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    modules = [args.module] if args.module else ["A", "B"]
    
    results = {}
    for mod in modules:
        try:
            results[mod] = update_module(mod, token)
        except Exception as e:
            print(f"\n💥 CRASH: {e}")
            import traceback
            traceback.print_exc()
            results[mod] = False
        time.sleep(2)

    print(f"\n{'='*60}")
    print(f"RESULTS: {results}")
    print(f"{'='*60}")
    
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
