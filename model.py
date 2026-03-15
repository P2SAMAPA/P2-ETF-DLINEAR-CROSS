"""
model.py — P2-ETF-DLINEAR-CROSS
==================================
Two model architectures adapted from the paper:
  1. DLinear  — simple MLP with seasonal-trend decomposition
  2. Crossformer — transformer with cross-dimension attention

Both share the same output layer design from the paper:
  - Final layer: N+1 neurons (N ETFs + 1 Hold node)
  - Activation : tanh() bounds outputs to (-1, 1)
  - The loss function (StockLoss-L2) interprets these outputs as
    buy (+), short (-), hold (last neuron)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Shared output head
# ══════════════════════════════════════════════════════════════════════════════

class TradingHead(nn.Module):
    """
    Final output layer with tanh activation.
    use_hold=True  → N+1 outputs (N ETFs + 1 Hold node)
    use_hold=False → N outputs only, forces model to always allocate
    Bias initialised away from zero to prevent hold-collapse.
    """
    def __init__(self, in_features: int, n_assets: int,
                 bias_init: float = 1.0, use_hold: bool = False):
        super().__init__()
        n_out   = n_assets + 1 if use_hold else n_assets
        self.fc = nn.Linear(in_features, n_out)
        nn.init.constant_(self.fc.bias, bias_init)
        if use_hold:
            # Hold node should start neutral
            with torch.no_grad():
                self.fc.bias[-1] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.fc(x))


# ══════════════════════════════════════════════════════════════════════════════
# Model 1: DLinear
# ══════════════════════════════════════════════════════════════════════════════

class MovingAvgDecomposition(nn.Module):
    """Seasonal-trend decomposition via moving average."""
    def __init__(self, kernel_size: int):
        super().__init__()
        # Pad to keep same length
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, n_feat)
        pad_size = self.kernel_size - 1
        pad_l    = pad_size // 2
        pad_r    = pad_size - pad_l

        x_t = x.permute(0, 2, 1)                       # (batch, n_feat, seq_len)
        x_t = F.pad(x_t, (pad_l, pad_r), mode="replicate")
        trend    = self.avg(x_t).permute(0, 2, 1)      # (batch, seq_len, n_feat)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear adapted for ETF trading.
    Input  : (batch, seq_len, n_features)
    Output : (batch, N+1)  — tanh-bounded trading decisions
    """
    def __init__(self, seq_len: int, n_features: int, n_assets: int,
                 individual: bool = False, kernel_size: int = 25, bias_init: float = 1.0, use_hold: bool = False):
        super().__init__()
        self.seq_len    = seq_len
        self.n_features = n_features
        self.individual = individual

        self.decomp = MovingAvgDecomposition(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList([
                nn.Linear(seq_len, 1) for _ in range(n_features)
            ])
            self.linear_trend = nn.ModuleList([
                nn.Linear(seq_len, 1) for _ in range(n_features)
            ])
        else:
            self.linear_seasonal = nn.Linear(seq_len, 1)
            self.linear_trend    = nn.Linear(seq_len, 1)

        # Flatten (1, n_features) → n_features → trading head
        self.head = TradingHead(n_features, n_assets, bias_init, use_hold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_feat)
        seasonal, trend = self.decomp(x)

        if self.individual:
            s_out = torch.stack([
                self.linear_seasonal[i](seasonal[:, :, i])   # (batch, 1)
                for i in range(self.n_features)
            ], dim=-1)                                         # (batch, 1, n_feat)
            t_out = torch.stack([
                self.linear_trend[i](trend[:, :, i])
                for i in range(self.n_features)
            ], dim=-1)
        else:
            # linear_seasonal expects (batch, n_feat, seq_len)
            s_out = self.linear_seasonal(seasonal.permute(0, 2, 1))   # (batch, n_feat, 1)
            t_out = self.linear_trend(trend.permute(0, 2, 1))
            s_out = s_out.permute(0, 2, 1)   # (batch, 1, n_feat)
            t_out = t_out.permute(0, 2, 1)

        out = (s_out + t_out).squeeze(1)     # (batch, n_feat)
        return self.head(out)                # (batch, N+1)


# ══════════════════════════════════════════════════════════════════════════════
# Model 2: Crossformer
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """Dimension-Segment-Wise (DSW) patch embedding."""
    def __init__(self, seg_len: int, d_model: int, n_features: int, dropout: float):
        super().__init__()
        self.seg_len  = seg_len
        self.proj     = nn.Linear(seg_len, d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(dropout)
        # Learnable per-feature embedding
        self.feat_emb = nn.Embedding(n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_feat)
        B, T, F = x.shape
        # Pad seq_len to be divisible by seg_len
        n_segs = math.ceil(T / self.seg_len)
        pad    = n_segs * self.seg_len - T
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))

        # Reshape to segments: (batch, n_segs, n_feat, seg_len)
        x = x.reshape(B, n_segs, self.seg_len, F).permute(0, 1, 3, 2)
        # x: (batch, n_segs, n_feat, seg_len)

        # Project segments → d_model
        x = self.proj(x)        # (batch, n_segs, n_feat, d_model)

        # Add feature embeddings
        feat_idx = torch.arange(F, device=x.device)
        feat_e   = self.feat_emb(feat_idx)              # (n_feat, d_model)
        x = x + feat_e.unsqueeze(0).unsqueeze(0)

        x = self.norm(x)
        x = self.dropout(x)
        return x   # (batch, n_segs, n_feat, d_model)


class TwoStageAttention(nn.Module):
    """
    Two-Stage Attention (TSA):
      Stage 1 — temporal attention across segments (per feature)
      Stage 2 — cross-dimension attention across features (per segment)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.temp_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_segs, n_feat, d_model)
        B, S, F, D = x.shape

        # Stage 1: temporal attention — for each feature, attend across segments
        x_t = x.permute(0, 2, 1, 3).reshape(B * F, S, D)
        attn_out, _ = self.temp_attn(x_t, x_t, x_t)
        x_t = self.norm1(x_t + attn_out)
        x_t = x_t.reshape(B, F, S, D).permute(0, 2, 1, 3)  # (B, S, F, D)

        # Stage 2: cross-dimension attention — for each segment, attend across features
        x_c = x_t.reshape(B * S, F, D)
        attn_out, _ = self.cross_attn(x_c, x_c, x_c)
        x_c = self.norm2(x_c + attn_out)
        x_c = x_c.reshape(B, S, F, D)

        # Feed-forward
        out = self.norm3(x_c + self.ff(x_c))
        return out   # (B, S, F, D)


class Crossformer(nn.Module):
    """
    Crossformer adapted for ETF trading.
    Input  : (batch, seq_len, n_features)
    Output : (batch, N+1)  — tanh-bounded trading decisions
    """
    def __init__(self, seq_len: int, n_features: int, n_assets: int,
                 d_model: int = 64, n_heads: int = 2, e_layers: int = 2,
                 d_ff: int = 128, seg_len: int = 12,
                 dropout: float = 0.2, bias_init: float = 1.0, use_hold: bool = False):
        super().__init__()
        self.patch_emb = PatchEmbedding(seg_len, d_model, n_features, dropout)
        self.encoder   = nn.ModuleList([
            TwoStageAttention(d_model, n_heads, dropout)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Pool over segments and features → flat vector → trading head
        self.head = TradingHead(d_model, n_assets, bias_init, use_hold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_feat)
        out = self.patch_emb(x)              # (B, n_segs, n_feat, d_model)
        for layer in self.encoder:
            out = layer(out)
        out = self.norm(out)
        # Average pool over segments and features
        out = out.mean(dim=[1, 2])           # (batch, d_model)
        return self.head(out)                # (batch, N+1)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_model(model_name: str, cfg) -> nn.Module:
    """
    Instantiate a model from config.
    model_name: 'dlinear', 'crossformer',
                or variant like 'dlinear_prc', 'crossformer_ret'
    cfg       : config_equity or config_fixed_income module
    """
    n_feat   = cfg.N_ASSETS * 6     # 6 features per ETF (close, return, vol_change, ma5, ma20, rsi)
    n_assets = cfg.N_ASSETS

    bias_init = getattr(cfg, 'OUTPUT_BIAS_INIT', 1.0)
    use_hold  = getattr(cfg, 'USE_HOLD', False)

    # Strip loss suffix if present (e.g. "dlinear_prc" → "dlinear")
    arch = model_name.lower().split("_")[0]

    if arch == "dlinear":
        return DLinear(
            seq_len    = cfg.SEQ_LEN,
            n_features = n_feat,
            n_assets   = n_assets,
            individual = cfg.DLINEAR_INDIVIDUAL,
            bias_init  = bias_init,
            use_hold   = use_hold,
        )
    elif arch == "crossformer":
        return Crossformer(
            seq_len    = cfg.SEQ_LEN,
            n_features = n_feat,
            n_assets   = n_assets,
            d_model    = cfg.CROSS_D_MODEL,
            n_heads    = cfg.CROSS_N_HEADS,
            e_layers   = cfg.CROSS_E_LAYERS,
            d_ff       = cfg.CROSS_D_FF,
            seg_len    = cfg.CROSS_SEG_LEN,
            dropout    = cfg.CROSS_DROPOUT,
            bias_init  = bias_init,
            use_hold   = use_hold,
        )
    elif arch == "mole":
        n_heads      = getattr(cfg, 'MOLE_N_HEADS', 4)
        head_dropout = getattr(cfg, 'MOLE_HEAD_DROPOUT', 0.0)
        return MoLEDLinear(
            seq_len      = cfg.SEQ_LEN,
            n_features   = n_feat,
            n_assets     = n_assets,
            n_heads      = n_heads,
            ts_dim       = 4,
            individual   = cfg.DLINEAR_INDIVIDUAL,
            bias_init    = bias_init,
            use_hold     = use_hold,
            head_dropout = head_dropout,
        )
    else:
        raise ValueError(f"Unknown model arch: '{arch}' (from '{model_name}'). Choose 'dlinear', 'crossformer', or 'mole'.")


# ══════════════════════════════════════════════════════════════════════════════
# Model 3: MoLE-DLinear
# ══════════════════════════════════════════════════════════════════════════════

class MoLEDLinear(nn.Module):
    """
    Mixture-of-Linear-Experts applied to DLinear for ETF trading.
    Based on: Ni et al. (2024) "Mixture-of-Linear-Experts for Long-term
    Time Series Forecasting" (AISTATS 2024)

    Architecture:
      - n_heads DLinear experts trained in parallel
      - A 2-layer MLP router that takes the start-of-window timestamp
        embedding (4 features: day_of_week, day_of_month, month, quarter)
        and outputs per-channel weights for each expert
      - Final output = weighted sum of expert outputs

    Input  : X (batch, seq_len, n_features), ts_mark (batch, 4)
    Output : (batch, N+1) — tanh-bounded trading decisions
    """
    def __init__(self, seq_len: int, n_features: int, n_assets: int,
                 n_heads: int = 4, ts_dim: int = 4,
                 individual: bool = False, kernel_size: int = 25,
                 bias_init: float = 1.0, use_hold: bool = False,
                 head_dropout: float = 0.0):
        super().__init__()
        self.n_heads      = n_heads
        self.n_assets     = n_assets
        self.head_dropout = head_dropout

        # n_heads DLinear experts (without their own TradingHead)
        self.experts = nn.ModuleList([
            _DLinearBackbone(seq_len, n_features, individual, kernel_size)
            for _ in range(n_heads)
        ])

        # Shared trading head over mixed output
        self.head = TradingHead(n_features, n_assets, bias_init, use_hold)

        # Router MLP: ts_dim → hidden → (n_assets+1) * n_heads
        # Channel-specific weights per head as in MoLE paper
        n_out_neurons = (n_assets + 1 if use_hold else n_assets)
        self.router = nn.Sequential(
            nn.Linear(ts_dim, ts_dim * 4),
            nn.ReLU(),
            nn.Linear(ts_dim * 4, n_out_neurons * n_heads),
        )
        self.n_out_neurons = n_out_neurons

    def forward(self, x: torch.Tensor,
                ts_mark: torch.Tensor | None = None) -> torch.Tensor:
        """
        x       : (batch, seq_len, n_features)
        ts_mark : (batch, 4) timestamp embedding — if None, equal weights
        """
        B = x.shape[0]

        # Get outputs from all experts: each is (batch, n_features)
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )   # (batch, n_heads, n_features)

        # Router weights
        if ts_mark is not None:
            # (batch, n_out_neurons * n_heads) → (batch, n_heads, n_out_neurons)
            raw_weights = self.router(ts_mark).view(B, self.n_heads, self.n_out_neurons)

            # Head dropout during training (regularisation from MoLE paper)
            if self.training and self.head_dropout > 0:
                mask = torch.bernoulli(
                    torch.ones(B, self.n_heads, 1, device=x.device) * (1 - self.head_dropout)
                )
                raw_weights = raw_weights * mask

            # Softmax over heads per output neuron
            weights = torch.softmax(raw_weights, dim=1)   # (batch, n_heads, n_out_neurons)
        else:
            # Equal weights fallback
            weights = torch.ones(B, self.n_heads, self.n_out_neurons,
                                 device=x.device) / self.n_heads

        # Mix expert outputs: weighted sum over heads
        # expert_outs: (batch, n_heads, n_features)
        # weights    : (batch, n_heads, n_out_neurons) — channel-specific
        # We apply weights to the final trading head outputs per expert

        # Get per-expert trading decisions first
        expert_decisions = torch.stack(
            [self.head(expert_outs[:, i, :]) for i in range(self.n_heads)], dim=1
        )   # (batch, n_heads, n_out_neurons)

        # Channel-wise weighted sum
        mixed = (expert_decisions * weights).sum(dim=1)   # (batch, n_out_neurons)

        # Apply tanh to final mixed output
        return torch.tanh(mixed)


class _DLinearBackbone(nn.Module):
    """DLinear without the TradingHead — outputs raw feature vector."""
    def __init__(self, seq_len: int, n_features: int,
                 individual: bool = False, kernel_size: int = 25):
        super().__init__()
        self.seq_len    = seq_len
        self.n_features = n_features
        self.individual = individual
        self.decomp     = MovingAvgDecomposition(kernel_size)

        if individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(seq_len, 1) for _ in range(n_features)])
            self.linear_trend    = nn.ModuleList([nn.Linear(seq_len, 1) for _ in range(n_features)])
        else:
            self.linear_seasonal = nn.Linear(seq_len, 1)
            self.linear_trend    = nn.Linear(seq_len, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(x)
        if self.individual:
            s_out = torch.stack([self.linear_seasonal[i](seasonal[:, :, i]) for i in range(self.n_features)], dim=-1)
            t_out = torch.stack([self.linear_trend[i](trend[:, :, i])       for i in range(self.n_features)], dim=-1)
        else:
            s_out = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            t_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        return (s_out + t_out).squeeze(1)   # (batch, n_features)
