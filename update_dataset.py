"""
update_dataset.py — P2-ETF-DLINEAR-CROSS
Working version with proper yfinance data handling.
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


def fetch_yfinance_data(tickers, target_date_str):
    """
    Fetch data from yfinance with proper formatting.
    CRITICAL: Returns DataFrame with MultiIndex columns (ticker, field).
    """
    print(f"\n📡 Fetching yfinance for {target_date_str}")
    print(f"   Tickers: {tickers}")
    
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    
    # Download 5 days of data to ensure we get the target date
    start_dt = target_dt - timedelta(days=7)
    end_dt = target_dt + timedelta(days=1)
    
    try:
        # Download with group_by='ticker' for proper MultiIndex
        data = yf.download(
            tickers,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
            group_by='ticker',
            threads=True,
        )
        
        print(f"   Downloaded shape: {data.shape}")
        print(f"   Columns type: {type(data.columns)}")
        print(f"   Index: {data.index[:3]}...")
        
        if data.empty:
            print(f"   ⚠️ Empty download")
            return None
        
        # Filter to exact target date
        target_ts = pd.Timestamp(target_date_str)
        mask = data.index.date == target_ts.date()
        day_data = data[mask]
        
        print(f"   Rows for {target_date_str}: {len(day_data)}")
        
        if len(day_data) == 0:
            print(f"   ⚠️ No data for target date")
            print(f"   Available dates: {list(set(data.index.date))[:5]}")
            return None
        
        # Handle single vs multiple tickers
        if len(tickers) == 1:
            # yfinance returns flat columns for single ticker
            ticker = tickers[0]
            df = day_data.copy()
            # Create MultiIndex columns: (ticker, field)
            df.columns = pd.MultiIndex.from_tuples(
                [(ticker, col) for col in df.columns]
            )
        else:
            # Multiple tickers: should already have MultiIndex from group_by='ticker'
            df = day_data.copy()
            if not isinstance(df.columns, pd.MultiIndex):
                print(f"   ⚠️ Converting to MultiIndex")
                # Try to infer structure
                df.columns = pd.MultiIndex.from_tuples(
                    [tuple(col) if isinstance(col, tuple) else (col, 'Unknown') 
                     for col in df.columns]
                )
        
        # Keep only OHLCV fields
        valid_cols = [col for col in df.columns if col[1] in OHLCV_FIELDS]
        df = df[valid_cols]
        
        print(f"   Final shape: {df.shape}")
        print(f"   Columns: {list(df.columns)[:5]}...")
        
        return df
        
    except Exception as e:
        print(f"   ❌ Fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_module(module, token):
    """Update module with proper data handling."""
    cfg = MODULE_CONFIG[module]
    print(f"\n{'='*60}")
    print(f"MODULE {module}: {cfg['label']}")
    print(f"{'='*60}")
    
    api = HfApi(token=token)
    
    # 1. Load existing
    print(f"\n📥 Loading existing...")
    try:
        local_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=cfg["hf_path_parquet"],
            token=token,
            force_download=True,
        )
        existing_df = pd.read_parquet(local_path)
        
        # Ensure index is datetime
        existing_df.index = pd.to_datetime(existing_df.index)
        
        print(f"   Loaded: {existing_df.shape}")
        print(f"   Date range: {existing_df.index[0]} to {existing_df.index[-1]}")
        print(f"   Columns: {len(existing_df.columns)}")
        
    except Exception as e:
        print(f"   ❌ Load failed: {e}")
        return False
    
    # 2. Determine target date
    last_date = existing_df.index[-1].date()
    target_date = last_date + timedelta(days=1)
    
    # Skip to next weekday
    while target_date.weekday() >= 5:
        target_date += timedelta(days=1)
    
    target_str = target_date.strftime("%Y-%m-%d")
    today = datetime.utcnow().date()
    
    print(f"\n📅 Last: {last_date} | Target: {target_str} | Today: {today}")
    
    if target_date > today:
        print(f"   ⏭️ Target is in future")
        return True
    
    # Check if exists
    existing_dates = set(pd.to_datetime(existing_df.index).date)
    if target_date in existing_dates:
        print(f"   ⏭️ Already exists")
        return True
    
    # 3. Fetch new data
    new_df = fetch_yfinance_data(cfg["tickers"], target_str)
    
    if new_df is None or new_df.empty:
        print(f"   ⚠️ No data (market closed?)")
        return True
    
    # 4. Prepare for merge
    print(f"\n🔧 Preparing merge...")
    
    # Ensure both have same column structure
    print(f"   Existing columns: {existing_df.columns[:3]}...")
    print(f"   New columns: {new_df.columns[:3]}...")
    
    # Reindex new to match existing
    new_df = new_df.reindex(columns=existing_df.columns)
    
    # Append
    combined = pd.concat([existing_df, new_df])
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.sort_index()
    
    # Check if actually changed
    if len(combined) == len(existing_df):
        print(f"   ⚠️ No new rows added")
        return True
    
    print(f"   Added {len(combined) - len(existing_df)} rows")
    print(f"   New total: {len(combined)}")
    print(f"   New date range: {combined.index[0]} to {combined.index[-1]}")
    
    # 5. Upload
    print(f"\n📤 Uploading...")
    temp_file = f"/tmp/update_{module}_{random.randint(1000,9999)}.parquet"
    combined.to_parquet(temp_file, index=True)
    
    file_size = os.path.getsize(temp_file)
    print(f"   File size: {file_size:,} bytes")
    
    try:
        result = api.upload_file(
            path_or_fileobj=temp_file,
            path_in_repo=cfg["hf_path_parquet"],
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Add {target_str} to {cfg['label']}",
        )
        print(f"   ✅ Success")
        
    except Exception as e:
        err_str = str(e)
        if "No files have been modified" in err_str:
            print(f"   ⚠️ HF reports no changes (data may be identical)")
            return True
        print(f"   ❌ Upload failed: {e}")
        return False
    
    # 6. Update metadata
    print(f"\n📝 Updating metadata...")
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
        metadata = {}
    
    metadata.update({
        "module": module,
        "label": cfg["label"],
        "last_data_update": str(combined.index[-1]),
        "rows": len(combined),
        "tickers": cfg["tickers"],
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
        commit_message=f"Metadata update for {target_str}",
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
        time.sleep(3)
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {results}")
    print(f"{'='*60}")
    
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
