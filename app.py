"""
app.py — P2-ETF-DLINEAR-CROSS
================================
Streamlit app showing:
  Tab 1 — Next-day ETF signals (Buy / Short / Hold + allocation %)
  Tab 2 — Backtest performance (return, Sharpe, drawdown vs Buy & Hold)
  Tab 3 — Model info (last trained, architecture, loss function)

Data sources:
  - HuggingFace dataset : P2SAMAPA/etf-dlinear-cross-data
  - GitHub results/     : model weights + eval_results.json
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import hf_hub_download
from datetime import datetime

# Ensure repo root is on path regardless of how the script is invoked
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Next US trading day helper ────────────────────────────────────────────────

US_HOLIDAYS = {
    # Fixed holidays (month, day)
    (1, 1), (7, 4), (12, 25),
}

def is_us_holiday(d):
    """Very rough US holiday check — covers major fixed holidays."""
    # New Year's, Independence Day, Christmas
    if (d.month, d.day) in US_HOLIDAYS:
        return True
    # Thanksgiving: 4th Thursday of November
    if d.month == 11 and d.weekday() == 3:
        count = (d.day - 1) // 7 + 1
        if count == 4:
            return True
    # MLK Day: 3rd Monday of January
    if d.month == 1 and d.weekday() == 0:
        count = (d.day - 1) // 7 + 1
        if count == 3:
            return True
    # Memorial Day: last Monday of May
    if d.month == 5 and d.weekday() == 0 and d.day > 24:
        return True
    # Labor Day: 1st Monday of September
    if d.month == 9 and d.weekday() == 0:
        count = (d.day - 1) // 7 + 1
        if count == 1:
            return True
    return False


def next_trading_day():
    """Return the next US market trading day from today (UTC)."""
    from datetime import date, timedelta
    d = date.today() + timedelta(days=1)
    while d.weekday() >= 5 or is_us_holiday(d):
        d += timedelta(days=1)
    return d.strftime("%A, %B %d, %Y")


def last_trading_day():
    """Return the most recent completed US trading day."""
    from datetime import date, timedelta
    d = date.today()
    # If today is weekend, go back to Friday
    while d.weekday() >= 5 or is_us_holiday(d):
        d -= timedelta(days=1)
    return d.strftime("%A, %B %d, %Y")

from data_loader import build_features
from model import get_model
import config_equity      as cfg_a
import config_fixed_income as cfg_b

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETF DLinear-Cross Trader",
    page_icon="📈",
    layout="wide",
)

HF_DATASET_REPO = "P2SAMAPA/etf-dlinear-cross-data"
HF_TOKEN        = os.getenv("HF_TOKEN", None)

MODULE_MAP = {
    "Option A — Equity ETFs":              ("A", cfg_a),
    "Option B — Fixed Income / Commodities": ("B", cfg_b),
}

RESULTS_MAP = {
    "A": "results/equity",
    "B": "results/fixed_income",
}

HF_PARQUET_MAP = {
    "A": "equity/ohlcv_equity.parquet",
    "B": "fixed_income/ohlcv_fixed_income.parquet",
}

HF_META_MAP = {
    "A": "equity/metadata_equity.json",
    "B": "fixed_income/metadata_fixed_income.json",
}


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_ohlcv(module: str) -> pd.DataFrame:
    local = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=HF_PARQUET_MAP[module],
        token=HF_TOKEN,
        force_download=False,
    )
    df = pd.read_parquet(local)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_hf_metadata(module: str) -> dict:
    try:
        local = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=HF_META_MAP[module],
            token=HF_TOKEN,
            force_download=False,
        )
        with open(local) as f:
            return json.load(f)
    except Exception:
        return {}


def latest_dated_file(directory: str, prefix: str, suffix: str):
    """Find the most recent date-stamped file: prefix_YYYYMMDD.suffix"""
    if not os.path.exists(directory):
        return None
    matches = []
    for fname in os.listdir(directory):
        if fname.startswith(prefix + "_") and fname.endswith(suffix):
            date_part = fname[len(prefix)+1 : -len(suffix)]
            if date_part.isdigit() and len(date_part) == 8:
                matches.append((date_part, os.path.join(directory, fname)))
    if not matches:
        return None
    return sorted(matches)[-1][1]


@st.cache_data(ttl=3600, show_spinner=False)
def load_eval_results(module: str) -> dict:
    path = latest_dated_file(RESULTS_MAP[module], "eval_results", ".json")
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600, show_spinner=False)
def load_model_meta(module: str, model_name: str) -> dict:
    # model_name is variant e.g. "dlinear_prc"
    path = latest_dated_file(RESULTS_MAP[module], f"{model_name}_meta", ".json")
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600, show_spinner=False)
def load_performance_history(module: str) -> list:
    path = os.path.join(RESULTS_MAP[module], "performance_history.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


# ── Signal generation ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def generate_signals(module: str, model_name: str,
                     ohlcv_cache_key: str = "") -> pd.DataFrame | None:
    """
    Run trained model on most recent seq_len days → next-day signals.
    ohlcv_cache_key is passed to bust cache when data updates.
    """
    cfg = cfg_a if module == "A" else cfg_b

    # model_name is now a variant like "dlinear_prc" or "crossformer_ret"
    weight_path = latest_dated_file(RESULTS_MAP[module], f"{model_name}_best", ".pt")
    scaler_path = latest_dated_file(RESULTS_MAP[module], "scaler", ".pkl")

    if not weight_path or not scaler_path:
        return None

    # Load scaler first — cheap local file
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Use already-cached OHLCV — avoids re-downloading from HF
    ohlcv_df = load_ohlcv(module)

    # Build features — only need last seq_len + 30 rows for speed
    tail_df = ohlcv_df.iloc[-(cfg.SEQ_LEN + 50):]
    features_df, prices_df = build_features(tail_df, cfg.TICKERS)
    valid = features_df.dropna().index.intersection(prices_df.dropna().index)
    features_df = features_df.loc[valid]

    if len(features_df) < cfg.SEQ_LEN:
        return None

    recent        = features_df.iloc[-cfg.SEQ_LEN:].values
    recent_scaled = scaler.transform(recent)

    # Model forward pass — CPU only
    device = torch.device("cpu")
    model  = get_model(model_name, cfg).to(device)
    try:
        model.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )
    except RuntimeError:
        # Weight mismatch — old weights incompatible with new architecture
        # Happens when USE_HOLD changed. Re-train to fix.
        return None
    model.eval()

    # Build timestamp mark for most recent window start
    from data_loader import compute_timestamp_features
    recent_idx   = ohlcv_df.iloc[-(cfg.SEQ_LEN + 50):].index
    valid_idx    = features_df.index
    window_start = valid_idx[-cfg.SEQ_LEN]
    ts_arr       = compute_timestamp_features(pd.DatetimeIndex([window_start]))
    ts_mark      = torch.tensor(ts_arr, dtype=torch.float32)   # (1, 4)

    import inspect as _inspect
    X = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        sig = _inspect.signature(model.forward)
        if 'ts_mark' in sig.parameters:
            O = model(X, ts_mark).squeeze(0).numpy()
        else:
            O = model(X).squeeze(0).numpy()    # (N+1,)

    use_hold = getattr(cfg, 'USE_HOLD', False)
    N        = cfg.N_ASSETS
    O_n      = O[:N]
    abs_O_n  = np.abs(O_n)
    total    = abs_O_n.sum()
    alloc_n  = (abs_O_n / total) * 100 if total > 1e-8 else np.zeros(N)

    rows = []
    for i, ticker in enumerate(cfg.TICKERS):
        signal = "BUY" if O_n[i] > 0.2 else ("SHORT" if O_n[i] < -0.2 else "HOLD")
        rows.append({
            "Ticker":      ticker,
            "Raw Output":  round(float(O_n[i]), 4),
            "Signal":      signal,
            "Allocation%": round(float(alloc_n[i]), 2),
        })
    if use_hold and len(O) > N:
        hold_alloc = abs(float(O[N])) / (abs_O_n.sum() + abs(float(O[N])) + 1e-8) * 100
        rows.append({
            "Ticker":      "HOLD (cash)",
            "Raw Output":  round(float(O[N]), 4),
            "Signal":      "HOLD",
            "Allocation%": round(hold_alloc, 2),
        })

    return pd.DataFrame(rows)


# ── UI helpers ────────────────────────────────────────────────────────────────

def signal_color(signal: str) -> str:
    return {"BUY": "🟢", "SHORT": "🔴", "HOLD": "🟡"}.get(signal, "⚪")


def render_signals_table(df: pd.DataFrame):
    display = df.copy()
    display["Signal"] = display["Signal"].apply(lambda s: f"{signal_color(s)} {s}")
    st.dataframe(
        display[["Ticker", "Signal", "Allocation%"]],
        use_container_width=True,
        hide_index=True,
    )


def render_allocation_chart(df: pd.DataFrame):
    fig = px.bar(
        df[df["Ticker"] != "HOLD (cash)"],
        x="Ticker", y="Allocation%",
        color="Signal",
        color_discrete_map={"BUY": "#00cc44", "SHORT": "#ff4444", "HOLD": "#ffcc00"},
        title="Portfolio Allocation by ETF",
    )
    fig.update_layout(showlegend=True, height=350)
    st.plotly_chart(fig, use_container_width=True)


def render_portfolio_chart(eval_results: dict, model_name: str):
    bh  = eval_results.get("buy_and_hold", {}).get("portfolio_values", [])
    model_data_chart = eval_results.get("models", {}).get(model_name, {})
    mdl = model_data_chart.get("portfolio_values", [])
    # Also try bh_values from model entry if top-level bh is missing
    if not bh:
        bh = model_data_chart.get("bh_values", [])

    if not bh or not mdl:
        st.info("Portfolio value data not available.")
        return

    n = min(len(bh), len(mdl))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=bh[:n],  name="Buy & Hold", line=dict(color="#aaaaaa", dash="dash")))
    fig.add_trace(go.Scatter(y=mdl[:n], name=model_name.upper(), line=dict(color="#0066ff")))
    fig.update_layout(
        title=f"{model_name.upper()} vs Buy & Hold — {eval_results.get('test_year', '')}",
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value ($)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_metrics_cards(metrics: dict, label: str):
    c1, c2, c3, c4 = st.columns(4)
    total  = metrics.get('total_return_pct', metrics.get('annual_return_pct', 'N/A'))
    n_days = metrics.get('n_days', 0)
    if isinstance(total, (int, float)) and n_days > 0:
        n_years  = n_days / 252.0
        cagr     = round(((1 + total / 100.0) ** (1.0 / n_years) - 1.0) * 100.0, 4)
        cagr_str = f"{cagr}%"
    else:
        cagr_str = "N/A"
    c1.metric("Total Return",      f"{total}%" if isinstance(total, (int,float)) else "N/A")
    c2.metric("CAGR (Annualised)", cagr_str)
    c3.metric("Sharpe Ratio",      f"{metrics.get('sharpe_ratio', 'N/A')}")
    c4.metric("Max Drawdown",      f"{metrics.get('max_drawdown_pct', 'N/A')}%")


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.title("📈 ETF DLinear-Cross Trader")
    st.caption("Next-day ETF signals via profit-guided neural networks | Kar et al. (2025)")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        module_label = st.selectbox("Module", list(MODULE_MAP.keys()))
        module, cfg  = MODULE_MAP[module_label]

        model_name = st.selectbox("Model", [
            "dlinear_prc",
            "crossformer_prc",
            "dlinear_ret",
            "crossformer_ret",
            "mole_prc",
            "mole_ret",
        ], format_func=lambda x: {
            "dlinear_prc":     "DLinear + Price Loss (PRC)",
            "crossformer_prc": "Crossformer + Price Loss (PRC)",
            "dlinear_ret":     "DLinear + Return Loss (RET)",
            "crossformer_ret": "Crossformer + Return Loss (RET)",
            "mole_prc":        "MoLE-DLinear + Price Loss (PRC) 🆕",
            "mole_ret":        "MoLE-DLinear + Return Loss (RET) 🆕",
        }.get(x, x))

        st.divider()
        hf_meta = load_hf_metadata(module)
        if hf_meta:
            st.caption(f"**Dataset last updated:** {hf_meta.get('last_data_update', 'N/A')}")
            st.caption(f"**Dataset rows:** {hf_meta.get('rows', 'N/A'):,}")
        model_meta = load_model_meta(module, model_name)
        if model_meta:
            st.caption(f"**Model last trained:** {model_meta.get('trained_at', 'N/A')}")
            st.caption(f"**Epochs trained:** {model_meta.get('epochs_trained', 'N/A')}")

        st.divider()
        st.markdown(
            "**Paper:** [Kar et al., arXiv:2507.19639](https://arxiv.org/abs/2507.19639)\n\n"
            "**GitHub:** [P2-ETF-DLINEAR-CROSS](https://github.com/P2SAMAPA/P2-ETF-DLINEAR-CROSS)\n\n"
            "**HF Dataset:** [etf-dlinear-cross-data](https://huggingface.co/datasets/P2SAMAPA/etf-dlinear-cross-data)"
        )
        st.warning("⚠️ Research only. Not financial advice.")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Next-Day Signals",
        "📊 Backtest Performance",
        "🔧 Model Info",
        "📋 Equity Summary",
        "📋 Fixed Income Summary",
    ])

    # ── Tab 1: Next-Day Signals ───────────────────────────────────────────────
    with tab1:
        st.subheader(f"Next Trading Day Signals — {module_label}")
        data_thru = hf_meta.get("last_data_update", "N/A")
        st.caption(f"Model trained on data through: {data_thru} | Using last {cfg.SEQ_LEN} trading days for inference")

        hf_meta    = load_hf_metadata(module)
        cache_key  = hf_meta.get("last_data_update", "")

        next_day  = next_trading_day()
        last_day  = last_trading_day()
        st.info(f"📅 **Prediction for:** {next_day}  |  Based on data through: {last_day}")

        with st.spinner("Loading model and computing signals (first load may take ~30s)..."):
            signals_df = generate_signals(module, model_name, cache_key)

        if signals_df is None:
            st.warning(
                "Model weights not found or incompatible. Please run training first via GitHub Actions "
                "(`train_equity.yml` or `train_fixed_income.yml`)."
            )
        else:
            render_signals_table(signals_df)
            render_allocation_chart(signals_df)

    # ── Tab 2: Backtest Performance ───────────────────────────────────────────
    with tab2:
        st.subheader(f"Backtest Performance — {module_label}")
        eval_results = load_eval_results(module)

        if not eval_results:
            st.warning("Evaluation results not found. Run `evaluate.py` first.")
        else:
            test_period = eval_results.get("test_period", eval_results.get("test_year", ""))
            st.caption(f"Test period: {test_period}")

            # Buy & Hold metrics
            bh_metrics = eval_results.get("buy_and_hold", {}).get("metrics", {})
            if bh_metrics:
                st.caption(
                    "📊 **Equal-Weight Buy & Hold** — equal allocation across all "
                    f"{len(cfg.TICKERS)} ETFs in this module, held for the full test period."
                )
                render_metrics_cards(bh_metrics, "Buy & Hold")

            st.divider()

            # Model metrics
            model_data = eval_results.get("models", {}).get(model_name, {})
            if model_data:
                render_metrics_cards(model_data.get("metrics", {}), model_name.upper())
                render_portfolio_chart(eval_results, model_name)

                # Allocation heatmap
                avg_alloc = model_data.get("avg_alloc_pct", {})
                if avg_alloc:
                    alloc_df = pd.DataFrame(
                        avg_alloc.items(), columns=["Ticker", "Avg Allocation%"]
                    ).sort_values("Avg Allocation%", ascending=False)
                    fig = px.bar(
                        alloc_df, x="Ticker", y="Avg Allocation%",
                        title="Average Portfolio Allocation over Test Period",
                        color="Avg Allocation%",
                        color_continuous_scale="Blues",
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No evaluation results for {model_name.upper()} yet.")

    # ── Tab 3: Model Info ─────────────────────────────────────────────────────
    with tab3:
        st.subheader("Model Architecture & Training Info")

        col1, col2 = st.columns(2)

        arch      = model_name.split("_")[0]
        loss_type = model_name.split("_")[1].upper() if "_" in model_name else "PRC"

        with col1:
            st.markdown("**Architecture Details**")
            if arch == "mole":
                n_heads = getattr(cfg, 'MOLE_N_HEADS', 4)
                st.markdown(f"""
- **Model**: MoLE-DLinear ({loss_type} loss) 🆕
- **Experts (heads)**: {n_heads} DLinear models in parallel
- **Router**: 2-layer MLP on timestamp embedding
- **Timestamp features**: day_of_week, day_of_month, month, quarter
- **Mixing**: channel-wise softmax weighted sum
- **Seq Length**: {cfg.SEQ_LEN} trading days
- **Output**: {cfg.N_ASSETS} neurons (tanh, no Hold)
- **Loss**: StockLoss-L2 {loss_type}
- **γ (smooth sign)**: {cfg.GAMMA}
- **Paper**: Ni et al. (AISTATS 2024)
                """)
            elif arch == "dlinear":
                st.markdown(f"""
- **Model**: DLinear ({loss_type} loss)
- **Decomposition**: Seasonal-Trend (kernel=25)
- **Seq Length**: {cfg.SEQ_LEN} trading days
- **Output**: {cfg.N_ASSETS} neurons (tanh, no Hold)
- **Loss**: StockLoss-L2 {loss_type}
- **γ (smooth sign)**: {cfg.GAMMA}
                """)
            else:
                st.markdown(f"""
- **Model**: Crossformer ({loss_type} loss)
- **d_model**: {cfg.CROSS_D_MODEL}
- **Heads**: {cfg.CROSS_N_HEADS}
- **Encoder layers**: {cfg.CROSS_E_LAYERS}
- **Segment length**: {cfg.CROSS_SEG_LEN}
- **Seq Length**: {cfg.SEQ_LEN} trading days
- **Output**: {cfg.N_ASSETS} neurons (tanh, no Hold)
- **Loss**: StockLoss-L2 {loss_type}
- **γ (smooth sign)**: {cfg.GAMMA}
                """)

        with col2:
            st.markdown("**Training Config**")
            st.markdown(f"""
- **Module**: {cfg.MODULE} — {cfg.LABEL}
- **Tickers**: {', '.join(cfg.TICKERS)}
- **Train data**: from {cfg.START_DATE} to end of {cfg.VAL_YEAR - 1}
- **Val year**: {cfg.VAL_YEAR}
- **Test year**: {cfg.TEST_YEAR}
- **Epochs**: {cfg.EPOCHS} (early stop patience=20)
- **Batch size**: {cfg.BATCH_SIZE}
- **Learning rate**: {cfg.LR}
- **Optimiser**: Adam + ReduceLROnPlateau
            """)

        # Training history chart
        model_meta = load_model_meta(module, model_name)
        history    = model_meta.get("history", {})
        if history.get("train") and history.get("val"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history["train"], name="Train Loss", line=dict(color="#0066ff")))
            fig.add_trace(go.Scatter(y=history["val"],   name="Val Loss",   line=dict(color="#ff6600")))
            fig.update_layout(
                title="Training Loss History",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Raw price chart from HF dataset
        st.divider()
        st.markdown("**Raw Price History (Close)**")
        with st.spinner("Loading price data..."):
            try:
                ohlcv_df = load_ohlcv(module)
                close_df = pd.DataFrame({
                    t: ohlcv_df[t]["Close"]
                    for t in cfg.TICKERS
                    if t in ohlcv_df.columns.get_level_values(0)
                })
                # Normalise to 100 at start for comparison
                norm_df = close_df / close_df.iloc[0] * 100
                fig = px.line(norm_df, title="Normalised Close Prices (base=100)", height=400)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load price chart: {e}")


    # ── Helper: render one module summary ────────────────────────────────────
    def render_summary_tab(sum_module: str, sum_cfg, sum_label: str):
        """Render the full 4-variant summary for a given module."""
        next_day = next_trading_day()
        last_day = last_trading_day()
        st.info(f"📅 **Prediction for:** {next_day}  |  Data through: {last_day}")

        VARIANTS = [
            ("dlinear_prc",     "DLinear + PRC"),
            ("crossformer_prc", "Crossformer + PRC"),
            ("dlinear_ret",     "DLinear + RET"),
            ("crossformer_ret", "Crossformer + RET"),
            ("mole_prc",        "MoLE-DLinear + PRC 🆕"),
            ("mole_ret",        "MoLE-DLinear + RET 🆕"),
        ]

        hf_meta_sum  = load_hf_metadata(sum_module)
        cache_key    = hf_meta_sum.get("last_data_update", "")
        eval_results = load_eval_results(sum_module)

        # ── Section 1: Next-Day Top 2 Signals for all 4 variants ─────────────
        st.subheader("🎯 Next-Day Top 2 Signals — All Models")
        st.caption("Top 2 ETFs by allocation % for each model variant")

        sig_cols = st.columns(len(VARIANTS))
        for col_idx, (variant, variant_label) in enumerate(VARIANTS):
            with sig_cols[col_idx]:
                st.markdown(f"**{variant_label}**")
                signals_df = generate_signals(sum_module, variant, cache_key)
                if signals_df is None:
                    st.warning("No weights")
                else:
                    # Filter out HOLD (cash) row, sort by allocation
                    trading = signals_df[signals_df["Ticker"] != "HOLD (cash)"].copy()
                    trading = trading.sort_values("Allocation%", ascending=False).head(2)
                    for _, row in trading.iterrows():
                        icon = "🟢" if row["Signal"] == "BUY" else (
                               "🔴" if row["Signal"] == "SHORT" else "🟡")
                        st.metric(
                            label=f"{icon} {row['Ticker']}",
                            value=f"{row['Signal']}",
                            delta=f"{row['Allocation%']:.1f}% allocated"
                        )

        # ── Section 2: Backtest Metrics Comparison ────────────────────────────
        st.divider()
        st.subheader("📊 Backtest Performance — All Models vs Buy & Hold")
        with st.expander("📖 About the benchmarks", expanded=False):
            st.markdown("""
**Equal-Weight Buy & Hold** — splits the portfolio equally across all ETFs in the module
and holds them for the entire test period. For Option A this is 1/10th in each of the
10 equity ETFs; for Option B it is 1/8th in each of the 8 fixed income/commodity ETFs.
This is the standard portfolio benchmark used in the paper (Kar et al., 2025).

**Single ETF Buy & Hold** — buys and holds only the single ETF that the model allocated
the most capital to on average across the test period (e.g. GLD or SLV). This is the most
relevant benchmark for live trading since the model is effectively recommending one ETF.
If the model cannot beat simply buying and holding its own top pick, the active trading
strategy adds no value.
            """)

        if not eval_results:
            st.warning("No evaluation results found. Run training first.")
            return

        test_period = eval_results.get("test_period",
                      eval_results.get("test_year", "N/A"))
        st.caption(f"Test period: {test_period}")

        # Buy & Hold baseline row
        bh = eval_results.get("buy_and_hold", {}).get("metrics", {})

        # Build comparison table
        rows = []
        def compute_cagr(total_pct, n_days):
            """Compute true CAGR from total return % and number of trading days."""
            if not isinstance(total_pct, (int, float)) or n_days <= 0:
                return "N/A"
            n_years = n_days / 252.0
            if n_years <= 0:
                return "N/A"
            cagr = ((1 + total_pct / 100.0) ** (1.0 / n_years) - 1.0) * 100.0
            return round(cagr, 4)

        def fmt_row(label, m, bh_total=None):
            total  = m.get('total_return_pct', m.get('annual_return_pct', 'N/A'))
            n_days = m.get('n_days', 0)
            n_yrs  = m.get('n_years', round(n_days / 252.0, 2) if n_days else 0)
            # Recompute CAGR properly — don't trust stored annual_return_pct
            # since old files stored total return in that field
            cagr   = compute_cagr(total, n_days) if n_days else m.get('annual_return_pct', 'N/A')
            beats  = isinstance(total, (int,float)) and isinstance(bh_total, (int,float)) and total > bh_total
            prefix = "✅" if beats else "⚙️"
            return {
                "Model":          label if "baseline" in label else f"{prefix} {label}",
                "Total Return":   f"{total}%" if isinstance(total, (int,float)) else "—",
                "CAGR (Ann.)":    f"{cagr}%" if isinstance(cagr, (int,float)) else "—",
                "Test Period":    f"{n_yrs:.1f} yrs" if isinstance(n_yrs, float) and n_yrs > 0 else "—",
                "Sharpe Ratio":   m.get('sharpe_ratio', 'N/A'),
                "Max Drawdown":   f"{m.get('max_drawdown_pct', 'N/A')}%",
                "Final $10k →":   f"${m.get('final_value', 0):,.2f}" if isinstance(m.get('final_value'), float) else "N/A",
            }

        models_data = eval_results.get("models", {})

        # Use model n_days for B&H period — ensures fair comparison
        # (B&H in old JSON files may have different n_days due to alignment bug)
        model_n_days = 0
        for variant, _ in VARIANTS:
            m_check = models_data.get(variant, {}).get("metrics", {})
            if m_check.get("n_days", 0) > 0:
                model_n_days = m_check["n_days"]
                break
        if model_n_days > 0:
            bh["n_days"] = model_n_days   # align B&H period to model period

        bh_total = bh.get('total_return_pct', bh.get('annual_return_pct', 0))
        rows.append(fmt_row("📈 Buy & Hold (baseline)", bh))
        for variant, variant_label in VARIANTS:
            m = models_data.get(variant, {}).get("metrics", {})
            if not m:
                rows.append({
                    "Model":          f"⚙️ {variant_label}",
                    "Total Return":   "—",
                    "CAGR (Ann.)":    "—",
                    "Test Period":    "—",
                    "Sharpe Ratio":   "—",
                    "Max Drawdown":   "—",
                    "Final $10k →":   "—",
                })
            else:
                rows.append(fmt_row(variant_label, m, bh_total))

            # Add single ETF B&H row directly below each model
            single = models_data.get(variant, {}).get("single_etf_bh", {})
            if single:
                ticker   = single.get("ticker", "")
                sm       = single.get("metrics", {})
                rows.append(fmt_row(
                    f"  └ {ticker} B&H (model top pick)",
                    sm,
                    bh_total
                ))

        metrics_df = pd.DataFrame(rows)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # ── Section 2b: Signal stability ─────────────────────────────────────
        st.divider()
        st.subheader("🎯 Signal Stability — How Often Each ETF Was Chosen")
        st.caption(
            "% of test days each ETF had BUY signal + avg allocation. "
            "🎯 = strong conviction (>60% of days, >30% avg alloc) | "
            "📊 = moderate | ❓ = inconsistent"
        )

        with st.expander("📖 How to read Signal Stability", expanded=False):
            st.markdown("""
**What is Signal Stability?**
Signal Stability measures whether a model's top ETF pick is a genuine recurring conviction or a one-off fluke.
It looks at three things across all test days:

| Metric | What it means |
|---|---|
| **Avg alloc %** | Average portfolio weight given to this ETF across all test days. High = the model consistently puts most of its money here. |
| **BUY X% of days** | How often this ETF had a clear BUY signal (tanh output > 0.2) during the test period. High = consistent direction. |
| **raw μ (mean)** | Average raw tanh output value. Range is -1 to +1. A mean of 0.87 means the model outputs a strong positive signal almost every day. |
| **raw σ (std dev)** | Variability of the raw output. Low σ = the model is consistent. High σ = the signal fluctuates a lot and today's reading may not be representative. |

**Conviction scoring:**
- 🎯 **Strong conviction** — mean output > 0.3 AND std dev < 0.3 AND BUY > 60% of days. The model consistently picked this ETF with a strong signal. Trust this signal.
- 📊 **Moderate** — mean output > 0.1 AND BUY > 40% of days. The model leans this way but with less certainty.
- ❓ **Inconsistent** — low average output despite occasional high allocation. The 100% allocation you see today may be an extreme reading of a noisy signal — treat with caution.

**Example interpretation:**
- `🎯 GLD: 94% avg alloc | BUY 88% of days | raw μ=0.87 σ=0.09` → Genuine conviction. The model outputs a strong, stable signal for GLD on almost every test day. High confidence.
- `❓ XES: 48% avg alloc | BUY 55% of days | raw μ=0.21 σ=0.38` → Likely a fluke. The model is barely above the BUY threshold on average and very inconsistent. Today's 100% reading is an outlier.
            """)

        stab_cols = st.columns(len(VARIANTS))
        for col_idx, (variant, variant_label) in enumerate(VARIANTS):
            with stab_cols[col_idx]:
                st.markdown(f"**{variant_label}**")
                model_metrics = models_data.get(variant, {})
                alloc_pct     = model_metrics.get("avg_alloc_pct", {})
                buy_ratio     = model_metrics.get("buy_ratio_pct", {})

                if not alloc_pct:
                    st.caption("No data yet")
                    continue

                output_stats  = model_metrics.get("output_stats", {})

                # Sort by avg allocation, show top 3
                top3 = sorted(alloc_pct.items(),
                              key=lambda x: x[1], reverse=True)[:3]
                for ticker, alloc in top3:
                    buy_pct   = buy_ratio.get(ticker, 0)
                    stats     = output_stats.get(ticker, {})
                    mean_out  = stats.get("mean", None)
                    std_out   = stats.get("std", None)
                    pct_clear = stats.get("pct_above_02", buy_pct)

                    # Conviction: high mean output + low std = consistent
                    if mean_out is not None:
                        is_conviction = mean_out > 0.3 and std_out < 0.3 and pct_clear > 60
                        is_moderate   = mean_out > 0.1 and pct_clear > 40
                    else:
                        is_conviction = alloc > 30 and buy_pct > 60
                        is_moderate   = alloc > 10 and buy_pct > 40

                    icon = "🎯" if is_conviction else ("📊" if is_moderate else "❓")

                    line = f"{icon} **{ticker}**: {alloc:.1f}% avg alloc | BUY {pct_clear:.0f}% of days"
                    if mean_out is not None:
                        line += f" | raw μ={mean_out:.2f} σ={std_out:.2f}"
                    st.markdown(line)

        # ── Section 3: Portfolio value chart — all 4 variants + B&H ─────────
        st.divider()
        st.subheader("📈 Portfolio Value — All Models vs Buy & Hold")

        fig = go.Figure()
        bh_vals = eval_results.get("buy_and_hold", {}).get("portfolio_values", [])
        if bh_vals:
            fig.add_trace(go.Scatter(
                y=bh_vals, name="Buy & Hold",
                line=dict(color="#aaaaaa", dash="dash", width=2)
            ))

        colors = ["#0066ff", "#ff6600", "#00cc44", "#cc00cc"]
        for (variant, variant_label), color in zip(VARIANTS, colors):
            vals = models_data.get(variant, {}).get("portfolio_values", [])
            if vals:
                fig.add_trace(go.Scatter(
                    y=vals, name=variant_label,
                    line=dict(color=color, width=1.5)
                ))

        fig.update_layout(
            xaxis_title="Trading Days",
            yaxis_title="Portfolio Value ($)",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Equity Summary ─────────────────────────────────────────────────
    with tab4:
        st.subheader("📋 Equity ETFs — Full Model Comparison")
        render_summary_tab("A", cfg_a, "Option A — Equity ETFs")

    # ── Tab 5: Fixed Income Summary ───────────────────────────────────────────
    with tab5:
        st.subheader("📋 Fixed Income / Commodities — Full Model Comparison")
        render_summary_tab("B", cfg_b, "Option B — Fixed Income / Commodities")


if __name__ == "__main__":
    main()
