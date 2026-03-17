"""
update_dataset.py — P2-ETF-DLINEAR-CROSS
Working version with proper yfinance fetching and change detection.
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
from huggingface_hub import HfApi, hf_hub_download

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


def fetch_yfinance_data(tickers, target_date_str, max_retries=4):
    """
    Fetch data from yfinance with proper date handling.
    CRITICAL: Use period="1d" with specific end date for reliability.
    """
    print(f"\n📡 Fetching yfinance data for {target_date_str}...")
    print(f"   Tickers: {tickers}")
    
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    
    # Try multiple approaches
    for attempt in range(max_retries):
        try:
            # APPROACH 1: Try downloading a range and filtering
            start_dt = target_dt - timedelta(days=5)  # Buffer for weekends
            end_dt = target_dt + timedelta(days=2)
            
            print(f"   Attempt {attempt+1}: Download {start_dt.date()} to {end_dt.date()}")
            
            # CRITICAL FIX: Use auto_adjust=False to get raw prices, then handle adjustment
            raw = yf.download(
                tickers,
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=False,  # Changed: get raw data
                threads=True,
                group_by='ticker' if len(tickers) > 1 else 'column',
            )
            
            print(f"   Raw download shape: {raw.shape}")
            print(f"   Raw columns: {list(raw.columns)[:5]}...")
            print(f"   Raw index: {raw.index[:3]}...")
            
            if raw.empty:
                print(f"   ⚠️ Empty download")
                time.sleep(2)
                continue
            
            # Filter to target date
            target_ts = pd.Timestamp(target_date_str)
            mask = raw.index.date == target_ts.date()
            filtered = raw[mask]
            
            print(f"   Filtered to {target_date_str}: {len(filtered)} rows")
            
            if len(filtered) == 0:
                print(f"   ⚠️ No data for exact date {target_date_str}")
                # Check what dates we have
                print(f"   Available dates: {set(raw.index.date)}")
                time.sleep(2)
                continue
            
            # Handle MultiIndex columns properly
            if len(tickers) == 1:
                # Single ticker: wrap in MultiIndex
                ticker = tickers[0]
                df = filtered.copy()
                df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
            else:
                # Multiple tickers: should already be MultiIndex
                df = filtered.copy()
                if not isinstance(df.columns, pd.MultiIndex):
                    print(f"   ⚠️ Converting to MultiIndex")
                    # Assume columns are like ['Open', 'High', ...] for each ticker
                    # This shouldn't happen with group_by='ticker'
                    pass
            
            # Keep only OHLCV fields
            df = df.loc[:, df.columns.get_level_values(1).isin(OHLCV_FIELDS)]
            
            print(f"   ✅ Success: {df.shape}")
            return df
            
        except Exception as e:
            print(f"   ❌ Attempt {attempt+1} failed: {e}")
            import traceback
            traceback.print_exc()
            
            if attempt < max_retries - 1:
                wait = 10 * (2 ** attempt)
                print(f"   ⏳ Waiting {wait}s...")
                time.sleep(wait)
    
    return None


def update_module(module, token):
    """Update module with proper change detection."""
    cfg = MODULE_CONFIG[module]
    print(f"\n{'='*60}")
    print(f"MODULE {module}: {cfg['label']}")
    print(f"{'='*60}")
    
    api = HfApi(token=token)
    
    # 1. Load existing
    print(f"\n📥 Loading existing data...")
    try:
        local_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=cfg["hf_path_parquet"],
            token=token,
            force_download=True,
        )
        existing_df = pd.read_parquet(local_path)
        print(f"   Loaded: {existing_df.shape}")
        print(f"   Date range: {existing_df.index[0]} to {existing_df.index[-1]}")
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        return False
    
    # 2. Determine target date
    last_date = pd.to_datetime(existing_df.index[-1]).date()
    target_date = last_date + timedelta(days=1)
    
    # Skip weekends
    while target_date.weekday() >= 5:  # 5=Sat, 6=Sun
        target_date += timedelta(days=1)
    
    target_str = target_date.strftime("%Y-%m-%d")
    today = datetime.utcnow().date()
    
    print(f"\n📅 Last data: {last_date}")
    print(f"📅 Target: {target_str}")
    print(f"📅 Today: {today}")
    
    # Don't try to fetch future dates
    if target_date > today:
        print(f"   ⏭️ Target date is in future. Nothing to do.")
        return True
    
    # Check if already exists
    if target_date in [d.date() for d in pd.to_datetime(existing_df.index)]:
        print(f"   ⏭️ Date already exists in dataset.")
        return True
    
    # 3. Fetch new data
    new_df = fetch_yfinance_data(cfg["tickers"], target_str)
    
    if new_df is None or new_df.empty:
        print(f"   ⚠️ No data fetched for {target_str}. Market closed?")
        return True  # Not a failure, just no data
    
    # 4. Process and merge
    print(f"\n🔧 Processing new data...")
    
    # Ensure column alignment
    new_df = new_df.reindex(columns=existing_df.columns)
    
    # Append
    combined = pd.concat([existing_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    
    # Forward fill only (no backfill to avoid future leakage)
    combined = combined.ffill()
    
    new_rows = len(combined) - len(existing_df)
    print(f"   Added {new_rows} rows")
    print(f"   New shape: {combined.shape}")
    
    if new_rows == 0:
        print(f"   ⏭️ No new data added.")
        return True
    
    # 5. Save and upload
    print(f"\n📤 Uploading...")
    temp_file = f"/tmp/update_{module}_{random.randint(1000,9999)}.parquet"
    combined.to_parquet(temp_file, index=True)
    
    try:
        result = api.upload_file(
            path_or_fileobj=temp_file,
            path_in_repo=cfg["hf_path_parquet"],
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Add {target_str} to {module}",
        )
        print(f"   ✅ Uploaded: {result}")
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        # Check if it's "no changes" error
        if "No files have been modified" in str(e):
            print(f"   ⚠️ Data unchanged (possibly duplicate). Continuing.")
            return True
        raise
    
    # 6. Update metadata
    try:
        meta_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=cfg["hf_path_metadata"],
            token=token,
        )
        with open(meta_path) as f:
            metadata = json.load(f)
    except:
        metadata = {"module": module, "label": cfg["label"]}
    
    metadata.update({
        "last_data_update": str(combined.index[-1]),
        "rows": len(combined),
        "last_updated_utc": datetime.utcnow().isoformat(),
    })
    
    meta_temp = f"/tmp/meta_{module}.json"
    with open(meta_temp, "w") as f:
        json.dump(metadata, f, indent=2)
    
    api.upload_file(
        path_or_fileobj=meta_temp,
        path_in_repo=cfg["hf_path_metadata"],
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message=f"Update metadata for {module}",
    )
    
    print(f"\n✅ MODULE {module} COMPLETE")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["A", "B"], default=None)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    modules = [args.module] if args.module else ["A", "B"]
    
    results = {}
    for mod in modules:
        try:
            results[mod] = update_module(mod, token)
        except Exception as e:
            print(f"\n💥 MODULE {mod} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[mod] = False
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {results}")
    print(f"{'='*60}")
    
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
