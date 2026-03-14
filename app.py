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


@st.cache_data(ttl=3600, show_spinner=False)
def load_eval_results(module: str) -> dict:
    path = os.path.join(RESULTS_MAP[module], "eval_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=3600, show_spinner=False)
def load_model_meta(module: str, model_name: str) -> dict:
    path = os.path.join(RESULTS_MAP[module], f"{model_name}_meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ── Signal generation ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def generate_signals(module: str, model_name: str) -> pd.DataFrame | None:
    """
    Run trained model on most recent seq_len days → next-day signals.
    Returns DataFrame with columns: Ticker, Signal, Allocation%, Direction
    """
    cfg = cfg_a if module == "A" else cfg_b

    weight_path = os.path.join(RESULTS_MAP[module], f"{model_name}_best.pt")
    scaler_path = os.path.join(RESULTS_MAP[module], "scaler.pkl")

    if not os.path.exists(weight_path) or not os.path.exists(scaler_path):
        return None

    # Load OHLCV
    ohlcv_df = load_ohlcv(module)

    # Build features
    features_df, prices_df = build_features(ohlcv_df, cfg.TICKERS)
    valid = features_df.dropna().index.intersection(prices_df.dropna().index)
    features_df = features_df.loc[valid]

    # Take last seq_len rows
    if len(features_df) < cfg.SEQ_LEN:
        return None
    recent = features_df.iloc[-cfg.SEQ_LEN:].values

    # Scale
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    recent_scaled = scaler.transform(recent)

    # Model forward pass
    device = torch.device("cpu")
    model  = get_model(model_name, cfg).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    X = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, n_feat)
    with torch.no_grad():
        O = model(X).squeeze(0).numpy()    # (N+1,)

    N      = cfg.N_ASSETS
    O_n    = O[:N]
    abs_O  = np.abs(O)
    total  = abs_O.sum()
    alloc  = (abs_O / total) * 100 if total > 1e-8 else np.zeros(N + 1)

    rows = []
    for i, ticker in enumerate(cfg.TICKERS):
        signal    = "BUY" if O_n[i] > 0.1 else ("SHORT" if O_n[i] < -0.1 else "HOLD")
        rows.append({
            "Ticker":      ticker,
            "Raw Output":  round(float(O_n[i]), 4),
            "Signal":      signal,
            "Allocation%": round(float(alloc[i]), 2),
        })

    # Hold node
    rows.append({
        "Ticker":      "HOLD (cash)",
        "Raw Output":  round(float(O[N]), 4),
        "Signal":      "HOLD",
        "Allocation%": round(float(alloc[N]), 2),
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
    mdl = eval_results.get("models", {}).get(model_name, {}).get("portfolio_values", [])

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
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{label} — Annual Return", f"{metrics.get('annual_return_pct', 'N/A')}%")
    c2.metric(f"{label} — Sharpe Ratio",  f"{metrics.get('sharpe_ratio', 'N/A')}")
    c3.metric(f"{label} — Max Drawdown",  f"{metrics.get('max_drawdown_pct', 'N/A')}%")


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    st.title("📈 ETF DLinear-Cross Trader")
    st.caption("Next-day ETF signals via profit-guided neural networks | Kar et al. (2025)")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        module_label = st.selectbox("Module", list(MODULE_MAP.keys()))
        module, cfg  = MODULE_MAP[module_label]

        model_name = st.selectbox("Model", ["dlinear", "crossformer"])

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
    tab1, tab2, tab3 = st.tabs([
        "🎯 Next-Day Signals",
        "📊 Backtest Performance",
        "🔧 Model Info",
    ])

    # ── Tab 1: Next-Day Signals ───────────────────────────────────────────────
    with tab1:
        st.subheader(f"Next Trading Day Signals — {module_label}")
        st.caption(f"Based on last {cfg.SEQ_LEN} trading days of data")

        with st.spinner("Generating signals..."):
            signals_df = generate_signals(module, model_name)

        if signals_df is None:
            st.warning(
                "Model weights not found. Please run training first via GitHub Actions "
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
            test_year = eval_results.get("test_year", "")
            st.caption(f"Test year: {test_year}")

            # Buy & Hold metrics
            bh_metrics = eval_results.get("buy_and_hold", {}).get("metrics", {})
            if bh_metrics:
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

        with col1:
            st.markdown("**Architecture Details**")
            if model_name == "dlinear":
                st.markdown(f"""
- **Model**: DLinear
- **Decomposition**: Seasonal-Trend (kernel=25)
- **Seq Length**: {cfg.SEQ_LEN} trading days
- **Output**: {cfg.N_ASSETS}+1 neurons (tanh)
- **Loss**: StockLoss-L2 PRC
- **γ (smooth sign)**: {cfg.GAMMA}
                """)
            else:
                st.markdown(f"""
- **Model**: Crossformer
- **d_model**: {cfg.CROSS_D_MODEL}
- **Heads**: {cfg.CROSS_N_HEADS}
- **Encoder layers**: {cfg.CROSS_E_LAYERS}
- **Segment length**: {cfg.CROSS_SEG_LEN}
- **Seq Length**: {cfg.SEQ_LEN} trading days
- **Output**: {cfg.N_ASSETS}+1 neurons (tanh)
- **Loss**: StockLoss-L2 PRC
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


if __name__ == "__main__":
    main()
