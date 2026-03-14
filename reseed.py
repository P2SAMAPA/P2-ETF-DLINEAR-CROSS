"""
reseed.py — P2-ETF-DLINEAR-CROSS
==================================
ONE-TIME script to build complete OHLCV dataset from:
  Option A (Equity ETFs)             : from 2006-01-01
  Option B (Fixed Income/Commodity)  : from 2008-01-01

Uses Yahoo Finance first, falls back to Stooq if YF fails.

Output:
  ohlcv_equity.parquet        → HF dataset split: equity
  ohlcv_fixed_income.parquet  → HF dataset split: fixed_income
  metadata_equity.json
  metadata_fixed_income.json

Run manually:
  python reseed.py --module A    # equity only
  python reseed.py --module B    # fixed income/commodity only
  python reseed.py               # both (default)
"""

import os
import json
import time
import random
import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime
from huggingface_hub import HfApi, CommitOperationAdd

# ── Configuration ─────────────────────────────────────────────────────────────
HF_DATASET_REPO = "P2SAMAPA/etf-dlinear-cross-data"

# Option A — Equity ETFs
EQUITY_TICKERS  = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "GDX", "IWM", "XES"]
EQUITY_START    = "2006-01-01"

# Option B — Fixed Income / Commodity ETFs
FIXED_TICKERS   = ["TLT", "VNQ", "GLD", "SLV", "LQD", "HYG", "MBB", "PFF"]
FIXED_START     = "2008-01-01"

# Shared
END_DATE        = datetime.today().strftime("%Y-%m-%d")
OHLCV_FIELDS    = ["Open", "High", "Low", "Close", "Volume"]

MODULE_CONFIG = {
    "A": {
        "tickers":   EQUITY_TICKERS,
        "start":     EQUITY_START,
        "parquet":   "ohlcv_equity.parquet",
        "metadata":  "metadata_equity.json",
        "hf_path_parquet":  "equity/ohlcv_equity.parquet",
        "hf_path_metadata": "equity/metadata_equity.json",
        "label":     "Equity ETFs",
    },
    "B": {
        "tickers":   FIXED_TICKERS,
        "start":     FIXED_START,
        "parquet":   "ohlcv_fixed_income.parquet",
        "metadata":  "metadata_fixed_income.json",
        "hf_path_parquet":  "fixed_income/ohlcv_fixed_income.parquet",
        "hf_path_metadata": "fixed_income/metadata_fixed_income.json",
        "label":     "Fixed Income / Commodity ETFs",
    },
}


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def fetch_ohlcv_yf(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch full OHLCV from Yahoo Finance with exponential backoff."""
    for attempt in range(6):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if raw.empty:
                raise ValueError(f"Empty response for {ticker}")

            # Flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0] for col in raw.columns]

            available = [f for f in OHLCV_FIELDS if f in raw.columns]
            if not available:
                raise ValueError(f"No OHLCV columns found for {ticker}")

            df = raw[available].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = pd.MultiIndex.from_tuples(
                [(ticker, f) for f in df.columns]
            )
            print(f"  ✅ {ticker} (YF): {len(df)} rows")
            return df

        except Exception as e:
            err = str(e).lower()
            is_rate = any(k in err for k in ["rate limit", "too many", "429", "ratelimit"])
            if is_rate and attempt < 5:
                wait = 30 * (2 ** attempt) + random.randint(5, 15)
                print(f"  ⚠️  YF rate limited on {ticker} (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ YF failed for {ticker} after {attempt+1} attempts: {e}")
                return None
    return None


def fetch_ohlcv_stooq(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch full OHLCV from Stooq as fallback."""
    stooq_symbol = ticker.lower() + ".us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"

    for attempt in range(3):
        try:
            raw = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
            if raw.empty:
                raise ValueError(f"Empty Stooq response for {ticker}")

            raw = raw.sort_index()
            mask = (raw.index >= start) & (raw.index <= end)
            raw = raw.loc[mask]
            if raw.empty:
                raise ValueError(f"No data in range for {ticker} from Stooq")

            available = [f for f in OHLCV_FIELDS if f in raw.columns]
            df = raw[available].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = pd.MultiIndex.from_tuples(
                [(ticker, f) for f in df.columns]
            )
            print(f"  ✅ {ticker} (Stooq): {len(df)} rows")
            return df

        except Exception as e:
            if attempt < 2:
                wait = 5 * (2 ** attempt) + random.randint(1, 5)
                print(f"  ⚠️  Stooq attempt {attempt+1} failed for {ticker}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Stooq failed for {ticker} after 3 attempts.")
                return None
    return None


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Try YF first, fall back to Stooq."""
    df = fetch_ohlcv_yf(ticker, start, end)
    if df is None:
        print(f"  🔄 Trying Stooq fallback for {ticker}...")
        df = fetch_ohlcv_stooq(ticker, start, end)
    return df


# ── Upload to HuggingFace ─────────────────────────────────────────────────────

def upload_to_hf(local_file: str, repo_path: str, token: str, commit_msg: str):
    """Upload a single file to HuggingFace dataset repo."""
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


# ── Seed one module ───────────────────────────────────────────────────────────

def seed_module(module: str, token: str):
    cfg = MODULE_CONFIG[module]
    tickers = cfg["tickers"]
    start   = cfg["start"]
    label   = cfg["label"]

    print("\n" + "=" * 60)
    print(f"MODULE {module} — {label}")
    print(f"Tickers : {tickers}")
    print(f"Range   : {start} → {END_DATE}")
    print("=" * 60)

    frames = []
    failed = []

    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        df = fetch_ticker(ticker, start, END_DATE)
        if df is not None:
            frames.append(df)
        else:
            failed.append(ticker)
        time.sleep(random.uniform(1.0, 2.5))

    if not frames:
        raise RuntimeError(f"No data fetched for module {module}. Aborting.")

    if failed:
        print(f"\n⚠️  Failed tickers: {failed} — continuing with {len(frames)} tickers.")

    # Combine into MultiIndex DataFrame: (ticker, field)
    ohlcv_df = pd.concat(frames, axis=1)
    ohlcv_df = ohlcv_df.sort_index()
    ohlcv_df = ohlcv_df[~ohlcv_df.index.duplicated(keep="last")]
    ohlcv_df = ohlcv_df.ffill()

    print(f"\n📊 Shape  : {ohlcv_df.shape}")
    print(f"   Columns: {list(ohlcv_df.columns[:6])} ...")
    print(f"   Range  : {ohlcv_df.index[0].date()} → {ohlcv_df.index[-1].date()}")

    # Save parquet locally
    ohlcv_df.to_parquet(cfg["parquet"])
    print(f"💾 Saved {cfg['parquet']} ({os.path.getsize(cfg['parquet']):,} bytes)")

    # Save metadata
    fetched_tickers = list(set(col[0] for col in ohlcv_df.columns))
    metadata = {
        "module":             module,
        "label":              label,
        "last_data_update":   str(ohlcv_df.index[-1].date()),
        "last_model_fit":     None,
        "dataset_version":    1,
        "seed_date":          str(datetime.today().date()),
        "rows":               len(ohlcv_df),
        "tickers":            fetched_tickers,
        "failed_tickers":     failed,
        "fields":             OHLCV_FIELDS,
        "start_date":         start,
        "end_date":           END_DATE,
    }
    with open(cfg["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Saved {cfg['metadata']}")

    # Upload to HuggingFace
    print(f"\n📤 Uploading to HuggingFace: {HF_DATASET_REPO}")
    last_date = metadata["last_data_update"]

    upload_to_hf(
        cfg["parquet"],
        cfg["hf_path_parquet"],
        token,
        f"Reseed Module {module}: {cfg['parquet']} — {last_date}",
    )
    upload_to_hf(
        cfg["metadata"],
        cfg["hf_path_metadata"],
        token,
        f"Reseed Module {module}: {cfg['metadata']} — {last_date}",
    )

    print(f"\n🎉 MODULE {module} RESEED COMPLETE — {len(ohlcv_df)} rows, {len(fetched_tickers)} tickers")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reseed ETF OHLCV dataset to HuggingFace")
    parser.add_argument(
        "--module",
        choices=["A", "B"],
        default=None,
        help="Module to seed: A (equity) or B (fixed income). Omit to seed both.",
    )
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    modules = [args.module] if args.module else ["A", "B"]

    for module in modules:
        seed_module(module, token)

    print("\n" + "=" * 60)
    print("✅ ALL REQUESTED MODULES RESEEDED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
