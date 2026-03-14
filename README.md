# P2-ETF-DLINEAR-CROSS

> **Next-day ETF trading signals using profit-guided loss functions with DLinear and Crossformer neural networks.**

---

## 📄 Research Foundation

This project is a direct implementation and adaptation of:

**"Directly Learning Stock Trading Strategies Through Profit Guided Loss Functions"**
Devroop Kar, Zimeng Lyu, Sheeraja Rajakrishnan, Alex Ororbia, Hao Zhang, Travis Desell, Daniel Krutz
*Rochester Institute of Technology — arXiv:2507.19639v1 [cs.LG], July 2025*
🔗 [https://arxiv.org/abs/2507.19639](https://arxiv.org/abs/2507.19639)

### Core Idea from the Paper
Rather than predicting prices and *then* making trading decisions (the traditional two-step pipeline), this approach collapses it into **one step**: a neural network directly learns a trading strategy by training on a custom loss function that reflects actual profit and loss.

Key innovations adopted from the paper:
- Replace the final layer of any neural network with **N+1 output neurons** (N assets + 1 "hold" node)
- Apply `tanh` activation to bound outputs between -1 (short) and +1 (buy)
- Train using **StockLoss-L2 with price values (PRC)** — the best-performing variant in the paper
- Smooth the discontinuous `sign()` function using `tanh(γx)` with γ=10 for stable training

The paper demonstrated this approach outperformed PPO, DDPG, SAC, TD3 and A2C reinforcement learning baselines, achieving returns of ~51% across 2021, 2022 and 2023 test periods on S&P 500 stocks.

---

## 🏗️ Project Structure

```
P2-ETF-DLINEAR-CROSS/
│
├── .github/workflows/
│   ├── reseed.yml              # ONE-TIME: seed full history to HF dataset
│   ├── update_dataset.yml      # DAILY: append new trading day to HF dataset
│   ├── train_equity.yml        # DAILY + manual: train Option A models
│   └── train_fixed_income.yml  # DAILY + manual: train Option B models
│
├── README.md
├── requirements.txt
│
├── config_equity.py            # Option A ETF list, start date, hyperparams
├── config_fixed_income.py      # Option B ETF list, start date, hyperparams
│
├── reseed.py                   # ONE-TIME full history seeding script
├── update_dataset.py           # DAILY incremental data update script
│
├── loss_functions.py           # StockLoss-L2 PRC (+ other variants stubbed)
├── model.py                    # DLinear + Crossformer with N+1 output layer
├── train.py                    # Training loop (--module A or B)
├── evaluate.py                 # Backtest: returns%, Sharpe ratio, drawdown
│
├── app.py                      # Streamlit app — predictions + performance
│
└── results/
    ├── equity/                 # Saved weights + metrics for Option A
    └── fixed_income/           # Saved weights + metrics for Option B
```

---

## 📦 Two Modules

### Option A — Equity ETFs (from 2006)
| Ticker | Description |
|--------|-------------|
| SPY | S&P 500 |
| QQQ | NASDAQ 100 |
| XLK | Technology |
| XLF | Financials |
| XLE | Energy |
| XLV | Health Care |
| XLI | Industrials |
| GDX | Gold Miners |
| IWM | Russell 2000 Small Cap |
| XES | Oil & Gas Equipment & Services |

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

## 🧠 Models

### DLinear
A simple but surprisingly effective MLP-based model with seasonal-trend decomposition along the temporal dimension. Fast to train, strong baseline.

### Crossformer
A transformer architecture that uses Dimension-Segment-Wise (DSW) embeddings and a Two-Stage Attention (TSA) layer to model both inter-temporal and inter-asset dependencies — ideal for capturing relationships between ETFs in the same module.

Both models are adapted with:
- N+1 output layer (N ETFs + 1 Hold node)
- `tanh` activation on final layer
- **StockLoss-L2 with PRC** loss function

---

## 📉 Loss Function: StockLoss-L2 (PRC variant)

```
L = 1 - sqrt( Σ V̂ᵢ · ( (PRCᵢ,t+1 - PRCᵢ,t) / max_j(PRCⱼ,t+1 - PRCⱼ,t) )² + H(V)² )
```

Where:
- `V̂ᵢ` = proportion of portfolio allocated to asset i
- `PRCᵢ,t` = price of ETF i at time t
- `sign(Oᵢ)` approximated by `tanh(γ · Oᵢ)` with γ=10 for smooth gradients
- `H(V)` = hold node output (risk mitigation)

---

## 🔄 Data Pipeline

```
ONE-TIME:  reseed.py        → HF Dataset (full history)
DAILY:     update_dataset.py → HF Dataset (append new row)
DAILY:     train.py          → results/ (retrain models)
ALWAYS:    app.py            → Streamlit (live predictions)
```

**HuggingFace Dataset**: [`P2SAMAPA/etf-dlinear-cross-data`](https://huggingface.co/datasets/P2SAMAPA/etf-dlinear-cross-data)

Two dataset splits:
- `equity` — Option A, from 2006
- `fixed_income` — Option B, from 2008

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
python reseed.py --module A   # seeds equity ETFs
python reseed.py --module B   # seeds fixed income/commodity ETFs
```

### 3. Train locally
```bash
python train.py --module A    # train equity models
python train.py --module B    # train fixed income models
```

### 4. Run Streamlit app locally
```bash
streamlit run app.py
```

---

## ⚙️ GitHub Actions

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `reseed.yml` | Manual only | One-time full history seed |
| `update_dataset.yml` | Daily 10:00 UTC + manual | Append new trading day |
| `train_equity.yml` | Daily 11:00 UTC + manual | Retrain Option A |
| `train_fixed_income.yml` | Daily 11:30 UTC + manual | Retrain Option B |

> Times are set after US market close (4pm ET = ~21:00 UTC). Adjust as needed.

---

## 📊 Streamlit App

Live app: [etf-dlinear-cross.streamlit.app](https://etf-dlinear-cross.streamlit.app)

Features:
- **Next-day signals**: Buy / Short / Hold per ETF with allocation %
- **Backtest performance**: Annual return vs Buy & Hold vs RL baseline
- **Model info**: Last trained, model type, loss function used

---

## 🙏 Citation

If you use this project, please cite the original paper:

```bibtex
@article{kar2025directly,
  title={Directly Learning Stock Trading Strategies Through Profit Guided Loss Functions},
  author={Kar, Devroop and Lyu, Zimeng and Rajakrishnan, Sheeraja and Ororbia, Alex and Zhang, Hao and Desell, Travis and Krutz, Daniel},
  journal={arXiv preprint arXiv:2507.19639},
  year={2025}
}
```

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**. It is not financial advice. Past model performance does not guarantee future returns. Always consult a qualified financial advisor before making investment decisions.
