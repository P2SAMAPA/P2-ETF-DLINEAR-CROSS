# P2-ETF-DLINEAR-CROSS

> **Next-day ETF trading signals using profit-guided loss functions with DLinear, Crossformer and MoLE-DLinear neural networks.**

---

## 📄 Research Foundation

This project implements and extends two papers:

### Primary Paper
**"Directly Learning Stock Trading Strategies Through Profit Guided Loss Functions"**
Devroop Kar, Zimeng Lyu, Sheeraja Rajakrishnan, Alex Ororbia, Hao Zhang, Travis Desell, Daniel Krutz
*Rochester Institute of Technology — arXiv:2507.19639v1 [cs.LG], July 2025*
🔗 [https://arxiv.org/abs/2507.19639](https://arxiv.org/abs/2507.19639)

The core idea: rather than predicting prices and *then* making trading decisions (the traditional two-step pipeline), a neural network directly learns a trading strategy by training on a custom **profit-guided loss function** that reflects actual P&L.

Key innovations adopted:
- Replace the final layer of any neural network with **N output neurons** (N ETFs, no Hold node)
- Apply `tanh` activation to bound outputs to (-1, 1) → buy/short decisions
- Train using **StockLoss-L2** loss function — best performing variant in the paper
- Smooth the discontinuous `sign()` function using `tanh(γx)` with γ=10

### Phase 4 Extension Paper
**"Mixture-of-Linear-Experts for Long-term Time Series Forecasting"**
Ronghao Ni, Zinan Lin, Shuaiqi Wang, Giulia Fanti
*Carnegie Mellon University & Microsoft Research — AISTATS 2024*
🔗 [https://github.com/RogerNi/MoLE](https://github.com/RogerNi/MoLE)

MoLE trains multiple DLinear experts in parallel and uses a timestamp-aware MLP router to weight their outputs. The router learns calendar effects (day of week, month, quarter) — relevant for ETFs which exhibit strong seasonal patterns around Fed meetings, earnings seasons, and quarter-end rebalancing.

---

## 🏗️ Project Structure

```
P2-ETF-DLINEAR-CROSS/
│
├── .github/workflows/
│   ├── reseed.yml              # ONE-TIME: seed full OHLCV history to HF dataset
│   ├── update_dataset.yml      # DAILY: append new trading day to HF dataset
│   ├── train_equity.yml        # DAILY + manual: train Option A models
│   ├── train_fixed_income.yml  # DAILY + manual: train Option B models
│   ├── evaluate_only.yml       # Manual: re-evaluate without retraining
│   └── archive_phase2a.yml     # ONE-TIME: archived Phase 2a results
│
├── README.md
├── requirements.txt
│
├── config_equity.py            # Option A: 9 equity ETFs, hyperparams, model variants
├── config_fixed_income.py      # Option B: 8 fixed income/commodity ETFs
│
├── reseed.py                   # ONE-TIME full history seeding script
├── update_dataset.py           # DAILY incremental data update
├── archive_current.py          # ONE-TIME Phase 2a archive helper
│
├── data_loader.py              # HF parquet → features + timestamp embeddings
├── loss_functions.py           # StockLoss-L2 PRC + RET variants
├── model.py                    # DLinear, Crossformer, MoLE-DLinear
├── train.py                    # Training loop — all 6 model variants
├── evaluate.py                 # Backtest: returns, CAGR, Sharpe, drawdown
│
├── app.py                      # Streamlit app — signals + performance
│
└── results/
    ├── equity/                 # Latest model weights + eval for Option A
    ├── fixed_income/           # Latest model weights + eval for Option B
    └── archive/                # Permanent phase snapshots for paper
        ├── ep50_test10_withXES/    # Phase 2a
        ├── ep300_test15_withXES/   # Phase 2b (auto-archived)
        └── ep300_test15_noXES/     # Phase 3 (auto-archived)
```

---

## 📦 Two Modules

### Option A — Equity ETFs (from 2006)
| Ticker | Description | Phase |
|--------|-------------|-------|
| SPY | S&P 500 | 2-4 |
| QQQ | NASDAQ 100 | 2-4 |
| XLK | Technology | 2-4 |
| XLF | Financials | 2-4 |
| XLE | Energy | 2-4 |
| XLV | Health Care | 2-4 |
| XLI | Industrials | 2-4 |
| GDX | Gold Miners | 2-4 |
| IWM | Russell 2000 Small Cap | 2-4 |
| ~~XES~~ | ~~Oil & Gas Equipment~~ | Phase 2 only — removed in Phase 3 due to price distortion |

### Option B — Fixed Income / Commodities ETFs (from 2008)
| Ticker | Description |
|--------|-------------|
| TLT | 20+ Year Treasury Bond |
| VNQ | Real Estate (REITs) |
| GLD | Gold |
| SLV | Silver |
| LQD | Investment Grade Corporate Bonds |
| HYG | High Yield Corporate Bonds |
| MBB | Mortgage-Backed Securities |
| PFF | Preferred Stock |

---

## 🧠 Models (6 variants per module)

### Phase 2/3 Models

#### DLinear
Simple but effective MLP with seasonal-trend decomposition. Fast to train, strong baseline. Full 300 epochs typically required — equity patterns are harder to learn than fixed income.

#### Crossformer
Transformer with Dimension-Segment-Wise (DSW) embeddings and Two-Stage Attention (TSA) to model inter-temporal and inter-asset dependencies. Note: Crossformer + PRC stops very early on fixed income (epoch 22) suggesting it cannot learn the dominant rate factor under PRC loss.

### Phase 4 Model — MoLE-DLinear
Mixture-of-Linear-Experts (Ni et al., AISTATS 2024) applied to ETF trading:
- **4 DLinear experts** trained in parallel
- **Timestamp-aware MLP router** takes start-of-window date features (day_of_week, day_of_month, month, quarter) and outputs channel-specific weights for each expert
- Learns calendar effects: different experts specialise in different market regimes (e.g. Fed week vs normal week, quarter-end vs mid-quarter)
- Added head dropout option for regularisation

All models use the same output layer:
- N output neurons with `tanh` activation (no Hold node)
- Loss function: StockLoss-L2 (PRC or RET variant)

---

## 📉 Loss Functions

### StockLoss-L2 (active)

```
L = 1 - sqrt( Σ (V̂ᵢ · norm_i · sign_i)² )
```

Two variants:
- **PRC** — `norm_i = (PRCᵢ,t+1 - PRCᵢ,t) / max_j(...)` — absolute price differences
- **RET** — `norm_i = (RETᵢ,t+1 - RETᵢ,t) / max_j(...)` — percentage return differences

`sign(Oᵢ)` is approximated by `tanh(γ · Oᵢ)` with γ=10 for smooth gradients.

### Key Finding: RET > PRC for ETFs
This project finds that **RET variants consistently outperform PRC variants** for both equity and fixed income ETFs — contradicting Kar et al. (2025) who found PRC > RET for individual S&P 500 stocks. The explanation: ETFs have significant price dispersion (GLD ~$460 vs MBB ~$95), causing PRC loss to be dominated by absolute dollar moves rather than meaningful return signals. RET loss normalises this out and produces genuinely superior trading strategies.

---

## 📊 Key Experimental Results

### Phase 2b (300 epochs, test=15%, with XES) — Equity Module A
| Model | Total Return | CAGR | Sharpe | vs Single ETF B&H |
|---|---|---|---|---|
| Buy & Hold (equal weight) | ~51.3% | ~19.5% | 1.56 | — |
| DLinear + PRC | 155.7% | 49.7% | 2.08 | XES B&H beats it ❌ |
| Crossformer + PRC | 66.7% | 24.5% | 1.90 | XES B&H beats it ❌ |
| **DLinear + RET** | **268.9%** | **75.1%** | 1.55 | +3.9% alpha over GDX ✅ |
| **Crossformer + RET** | **260.8%** | **73.5%** | 1.55 | +2.3% alpha over GDX ✅ |

### Phase 2b — Fixed Income Module B
| Model | Total Return | CAGR | Sharpe | vs Single ETF B&H |
|---|---|---|---|---|
| Buy & Hold (equal weight) | 51.3% | 20.6% | 1.90 | — |
| DLinear + PRC | 155.7% | 49.6% | 2.08 | ~+0.9% alpha over GLD ✅ |
| Crossformer + PRC | 66.7% | 24.5% | 1.90 | GLD B&H beats it ❌ |
| **DLinear + RET** | **268.9%** | **75.3%** | 1.55 | +3.95% alpha over SLV ✅ |
| **Crossformer + RET** | **260.8%** | **73.6%** | 1.55 | +2.28% alpha over SLV ✅ |

*Test period: 2023-06-23 → 2026-03-13 (2.3 years)*

### Phase 3 (XES removed from equity) — in progress

---

## 🔬 Research Phases

| Phase | Config | Key Question |
|-------|--------|-------------|
| **2a** | 50 epochs, test=10%, with XES | Baseline — does profit-guided loss work for ETFs? |
| **2b** | 300 epochs, test=15%, with XES | Does more training help? PRC vs RET comparison |
| **3** | 300 epochs, test=15%, no XES | Does removing low-price XES fix PRC equity performance? |
| **4** | 300 epochs, test=15%, no XES + MoLE | Does timestamp-aware mixture of experts improve results? |

Archived snapshots at: `results/archive/` and GitHub releases.

---

## 🔄 Data Pipeline

```
ONE-TIME:  reseed.py        → HF Dataset (full OHLCV history)
DAILY:     update_dataset.py → HF Dataset (append new trading day)
DAILY:     train.py          → results/ (retrain 6 model variants)
ALWAYS:    app.py            → Streamlit (live next-day signals)
```

**HuggingFace Dataset**: [`P2SAMAPA/etf-dlinear-cross-data`](https://huggingface.co/datasets/P2SAMAPA/etf-dlinear-cross-data)

Two splits:
- `equity` — Option A, from 2006-01-01
- `fixed_income` — Option B, from 2008-01-01

---

## ⚙️ GitHub Actions Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `reseed.yml` | Manual only | One-time full history seed |
| `update_dataset.yml` | Daily 21:30 UTC + manual | Append new trading day |
| `train_equity.yml` | Daily 22:00 UTC + manual | Retrain Option A (6 variants) |
| `train_fixed_income.yml` | Daily 22:30 UTC + manual | Retrain Option B (6 variants) |
| `evaluate_only.yml` | Manual | Re-evaluate without retraining |

Training times (CPU, GitHub Actions free tier):
- Fixed Income: ~1.5 hours (6 variants, 300 epochs)
- Equity: ~3.5-4 hours (6 variants, 300 epochs, 60 features vs 48)

---

## 📱 Streamlit App

Live app: [P2-ETF-DLINEAR-CROSS on Streamlit](https://p2-etf-dlinear-cross.streamlit.app)

**Tabs:**
- 🎯 **Next-Day Signals** — BUY/SHORT signal + allocation % for selected model
- 📊 **Backtest Performance** — Total return, CAGR, Sharpe, Max Drawdown vs B&H
- 🔧 **Model Info** — Architecture details, training config, loss history chart
- 📋 **Equity Summary** — All 6 models side-by-side for Option A
- 📋 **Fixed Income Summary** — All 6 models side-by-side for Option B

Summary tabs show:
- Next-day top 2 ETF signals per model
- Signal stability scores (conviction vs fluke analysis)
- Backtest metrics table including Single ETF B&H comparison
- Portfolio value chart — all models vs Buy & Hold

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/P2SAMAPA/P2-ETF-DLINEAR-CROSS.git
cd P2-ETF-DLINEAR-CROSS
pip install -r requirements.txt
```

### 2. Seed the dataset (run once)
```bash
export HF_TOKEN=your_token_here
python reseed.py          # seeds both modules
```

### 3. Train locally
```bash
python train.py --module A    # train all 6 equity variants
python train.py --module B    # train all 6 fixed income variants
```

### 4. Evaluate
```bash
python evaluate.py --module A
python evaluate.py --module B
```

### 5. Run Streamlit app locally
```bash
streamlit run app.py
```

---

## 🙏 Citations

If you use this project, please cite both papers:

```bibtex
@article{kar2025directly,
  title={Directly Learning Stock Trading Strategies Through Profit Guided Loss Functions},
  author={Kar, Devroop and Lyu, Zimeng and Rajakrishnan, Sheeraja and Ororbia, Alex
          and Zhang, Hao and Desell, Travis and Krutz, Daniel},
  journal={arXiv preprint arXiv:2507.19639},
  year={2025}
}

@inproceedings{ni2024mixture,
  title={Mixture-of-Linear-Experts for Long-term Time Series Forecasting},
  author={Ni, Ronghao and Lin, Zinan and Wang, Shuaiqi and Fanti, Giulia},
  booktitle={Proceedings of the 27th International Conference on Artificial Intelligence
             and Statistics (AISTATS)},
  year={2024}
}
```

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. It is not financial advice. Past model performance does not guarantee future returns. Always consult a qualified financial advisor before making investment decisions.
