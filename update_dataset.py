"""
update_dataset.py — P2-ETF-DLINEAR-CROSS
==========================================
DAILY script to append the latest trading day's OHLCV data
to the HuggingFace dataset for both modules.

- Downloads latest data via yfinance
- Checks if the new date already exists (no duplicates)
- Skips gracefully if market was closed (weekend / holiday)
- Appends new row to existing parquet and re-uploads to HF

Run:
  python update_dataset.py --module A
  python update_dataset.py --module B
  python update_dataset.py              # both (default)
"""

import os
import io
import json
import time
import random
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

# ── Config ────────────────────────────────────────────────────────────────────
HF_DATASET_REPO = "P2SAMAPA/etf-dlinear-cross-data"
OHLCV_FIELDS    = ["Open", "High", "Low", "Close", "Volume"]

MODULE_CONFIG = {
    "A": {
        "tickers":          ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "GDX", "IWM", "XES"],
        "parquet":          "ohlcv_equity.parquet",
        "metadata":         "metadata_equity.json",
        "hf_path_parquet":  "equity/ohlcv_equity.parquet",
        "hf_path_metadata": "equity/metadata_equity.json",
        "label":            "Equity ETFs",
    },
    "B": {
        "tickers":          ["TLT", "VNQ", "GLD", "SLV", "LQD", "HYG", "MBB", "PFF"],
        "parquet":          "ohlcv_fixed_income.parquet",
        "metadata":         "metadata_fixed_income.json",
        "hf_path_parquet":  "fixed_income/ohlcv_fixed_income.parquet",
        "hf_path_metadata": "fixed_income/metadata_fixed_income.json",
        "label":            "Fixed Income / Commodity ETFs",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_market_open_day(date: datetime) -> bool:
    """Rough check — skip weekends. Holidays handled by empty yfinance response."""
    return date.weekday() < 5   # Mon=0 … Fri=4


def fetch_latest_yf(tickers: list, date_str: str) -> pd.DataFrame | None:
    """
    Fetch OHLCV for a single trading day.
    yfinance end date is exclusive, so we request date+1.
    """
    end_str = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    for attempt in range(4):
        try:
            raw = yf.download(
                tickers,
                start=date_str,
                end=end_str,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if raw.empty:
                return None

            if isinstance(raw.columns, pd.MultiIndex):
                # Shape: (dates, (field, ticker)) → keep as MultiIndex (ticker, field)
                raw = raw.swaplevel(axis=1).sort_index(axis=1)
            else:
                # Single ticker — wrap in MultiIndex
                ticker = tickers[0]
                raw.columns = pd.MultiIndex.from_tuples([(ticker, f) for f in raw.columns])

            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            # Keep only requested fields
            raw = raw.loc[:, raw.columns.get_level_values(1).isin(OHLCV_FIELDS)]
            print(f"  ✅ Fetched {len(raw)} row(s) for {len(tickers)} tickers")
            return raw

        except Exception as e:
            err = str(e).lower()
            is_rate = any(k in err for k in ["rate limit", "too many", "429"])
            if is_rate and attempt < 3:
                wait = 20 * (2 ** attempt) + random.randint(3, 10)
                print(f"  ⚠️  Rate limited (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ YF fetch failed (attempt {attempt+1}): {e}")
                return None
    return None


def load_from_hf(hf_path: str, token: str) -> pd.DataFrame:
    """Download existing parquet from HF and load into DataFrame."""
    local = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=hf_path,
        token=token,
        force_download=True,
    )
    return pd.read_parquet(local)


def upload_to_hf(df: pd.DataFrame, local_file: str, repo_path: str,
                 token: str, commit_msg: str):
    """Save DataFrame to parquet and upload to HF."""
    df.to_parquet(local_file)
    api = HfApi(token=token)
    with open(local_file, "rb") as f:
        content = f.read()
    api.create_commit(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=token,
        commit_message=commit_msg,
        operations=[CommitOperationAdd(
            path_in_repo=repo_path,
            path_or_fileobj=content,
        )],
    )
    print(f"  ✅ Uploaded → {repo_path}")


def upload_metadata(metadata: dict, local_file: str, repo_path: str,
                    token: str, commit_msg: str):
    """Save metadata JSON and upload to HF."""
    with open(local_file, "w") as f:
        json.dump(metadata, f, indent=2)
    api = HfApi(token=token)
    with open(local_file, "rb") as f:
        content = f.read()
    api.create_commit(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=token,
        commit_message=commit_msg,
        operations=[CommitOperationAdd(
            path_in_repo=repo_path,
            path_or_fileobj=content,
        )],
    )
    print(f"  ✅ Uploaded → {repo_path}")


# ── Update one module ─────────────────────────────────────────────────────────

def update_module(module: str, token: str):
    cfg     = MODULE_CONFIG[module]
    tickers = cfg["tickers"]
    label   = cfg["label"]

    print("\n" + "=" * 60)
    print(f"MODULE {module} — {label}")
    print("=" * 60)

    # Determine target date = yesterday (most recent completed trading day)
    today     = datetime.utcnow()
    target_dt = today - timedelta(days=1)

    # Walk back to find last weekday
    while not is_market_open_day(target_dt):
        target_dt -= timedelta(days=1)
    target_str = target_dt.strftime("%Y-%m-%d")

    print(f"Target date : {target_str}")

    # Load existing dataset from HF
    print(f"📥 Loading existing dataset from HF...")
    try:
        existing_df = load_from_hf(cfg["hf_path_parquet"], token)
    except Exception as e:
        print(f"  ❌ Could not load existing parquet: {e}")
        return

    existing_df.index = pd.to_datetime(existing_df.index).tz_localize(None)
    last_date = existing_df.index[-1].date()
    print(f"   Existing dataset last date : {last_date}")
    print(f"   Existing dataset rows      : {len(existing_df)}")

    # Check if target date already exists
    target_date = pd.to_datetime(target_str).date()
    if target_date <= last_date:
        print(f"  ℹ️  Target date {target_str} already in dataset. Skipping.")
        return

    # Fetch new data
    print(f"\n📡 Fetching new data for {target_str}...")
    new_df = fetch_latest_yf(tickers, target_str)

    if new_df is None or new_df.empty:
        print(f"  ⚠️  No data returned for {target_str} — market may have been closed. Skipping.")
        return

    # Filter to only the target date row
    new_df = new_df[new_df.index.date == target_date]
    if new_df.empty:
        print(f"  ⚠️  No row for exact date {target_str} in fetched data. Skipping.")
        return

    # Align columns with existing dataset
    new_df = new_df.reindex(columns=existing_df.columns)

    # Append and deduplicate
    combined = pd.concat([existing_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined = combined.ffill()

    print(f"   New dataset rows : {len(combined)} (+{len(combined) - len(existing_df)})")

    # Upload updated parquet
    print(f"\n📤 Uploading updated dataset to HF...")
    upload_to_hf(
        combined,
        cfg["parquet"],
        cfg["hf_path_parquet"],
        token,
        f"Daily update Module {module}: added {target_str}",
    )

    # Update metadata
    try:
        meta_local = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=cfg["hf_path_metadata"],
            token=token,
            force_download=True,
        )
        with open(meta_local) as f:
            metadata = json.load(f)
    except Exception:
        metadata = {"module": module, "label": label, "dataset_version": 1}

    metadata["last_data_update"] = str(combined.index[-1].date())
    metadata["rows"]             = len(combined)
    metadata["last_updated_utc"] = str(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))

    upload_metadata(
        metadata,
        cfg["metadata"],
        cfg["hf_path_metadata"],
        token,
        f"Daily update Module {module}: metadata — {target_str}",
    )

    print(f"\n🎉 MODULE {module} UPDATE COMPLETE — added {target_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Daily incremental ETF data update")
    parser.add_argument(
        "--module",
        choices=["A", "B"],
        default=None,
        help="Module to update: A (equity) or B (fixed income). Omit for both.",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    modules = [args.module] if args.module else ["A", "B"]

    for module in modules:
        update_module(module, token)
        time.sleep(random.uniform(2.0, 4.0))   # polite delay between modules

    print("\n" + "=" * 60)
    print("✅ ALL REQUESTED MODULES UPDATED")
    print("=" * 60)


if __name__ == "__main__":
    main()
