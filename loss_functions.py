"""
loss_functions.py — P2-ETF-DLINEAR-CROSS
==========================================
Profit-guided loss functions adapted from:
  "Directly Learning Stock Trading Strategies Through Profit Guided Loss Functions"
  Kar et al., RIT, arXiv:2507.19639

Active loss  : StockLoss-L2 with PRC (Loss III, price variant) — best in paper
Others       : Stubbed and available for experimentation

All loss functions:
  - Accept raw model outputs O of shape (batch, N+1)
    where N = number of ETFs and the last neuron is the Hold node
  - Use tanh(gamma * O) as a smooth approximation of sign(O)
  - Return a scalar loss value (mean over batch)
"""

import torch
import torch.nn as nn


# ── Smooth sign approximation ─────────────────────────────────────────────────

def smooth_sign(x: torch.Tensor, gamma: float = 10.0) -> torch.Tensor:
    """
    Smooth approximation of sign(x) using tanh(gamma * x).
    Resolves the gradient discontinuity at 0.
    gamma=10 closely follows sign(x) as used in the paper.
    """
    return torch.tanh(gamma * x)


# ── Portfolio allocation from outputs ─────────────────────────────────────────

def portfolio_weights(O: torch.Tensor) -> torch.Tensor:
    """
    Compute normalised portfolio weights V̂_i = |O_i| / Σ|O_j|
    O shape: (batch, N+1) — last column is Hold node
    Returns V̂ shape: (batch, N+1)
    """
    abs_O = torch.abs(O)
    total = abs_O.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return abs_O / total


# ── Loss I: StockLoss ─────────────────────────────────────────────────────────

def stockloss(O: torch.Tensor, prc: torch.Tensor,
              gamma: float = 10.0, use_hold: bool = True) -> torch.Tensor:
    """
    Loss I — StockLoss (return-based, unnormalised).
    Volatile in practice; superseded by StockLoss-L2.

    Args:
        O   : model outputs (batch, N+1); last col = Hold node
        prc : price differences (batch, N)  i.e. PRC_{t+1} - PRC_t
        gamma     : smoothing coefficient
        use_hold  : whether to include Hold node in loss
    """
    N   = prc.shape[1]
    O_n = O[:, :N]                          # (batch, N) — trading outputs
    O_h = O[:, N:]                          # (batch, 1) — hold output

    V   = portfolio_weights(O)[:, :N]       # (batch, N)
    s   = smooth_sign(O_n, gamma)           # (batch, N)

    profit = (V * prc * s).sum(dim=1)       # (batch,)

    hold_penalty = torch.zeros_like(profit)
    if use_hold:
        V_h = portfolio_weights(O)[:, N:]   # (batch, 1)
        hold_penalty = V_h.squeeze(1)

    loss = -(profit + hold_penalty)
    return loss.mean()


# ── Loss II: StockLoss-Max ────────────────────────────────────────────────────

def stockloss_max(O: torch.Tensor, prc: torch.Tensor,
                  gamma: float = 10.0, use_hold: bool = True) -> torch.Tensor:
    """
    Loss II — StockLoss-Max (normalised by max return in window).
    More stable than Loss I.
    """
    N   = prc.shape[1]
    O_n = O[:, :N]
    O_h = O[:, N:]

    V   = portfolio_weights(O)[:, :N]
    s   = smooth_sign(O_n, gamma)

    max_prc = prc.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    norm_prc = prc / max_prc                # (batch, N)

    profit = (V * norm_prc * s).sum(dim=1)

    hold_penalty = torch.zeros_like(profit)
    if use_hold:
        V_h = portfolio_weights(O)[:, N:]
        hold_penalty = V_h.squeeze(1)

    loss = 1.0 - (profit + hold_penalty)
    return loss.mean()


# ── Loss III: StockLoss-L2 (ACTIVE) ──────────────────────────────────────────

def stockloss_l2(O: torch.Tensor, prc: torch.Tensor,
                 gamma: float = 10.0, use_hold: bool = False) -> torch.Tensor:
    """
    Loss III — StockLoss-L2 with PRC (price variant).
    *** ACTIVE LOSS FUNCTION FOR THIS PROJECT ***

    use_hold=False (default): O shape is (batch, N) — no Hold node
    use_hold=True:            O shape is (batch, N+1) — last col is Hold

    L = 1 - sqrt( Σ (V̂_i · norm_prc_i · sign_i)² [+ H²] )
    """
    N   = prc.shape[1]

    if use_hold:
        O_n = O[:, :N]
        V   = portfolio_weights(O)[:, :N]
        V_h = portfolio_weights(O)[:, N:]
    else:
        O_n = O                              # (batch, N)
        # When no hold: weights = |O_i| / Σ|O_j|
        abs_O = O_n.abs()
        total = abs_O.sum(dim=1, keepdim=True).clamp(min=1e-8)
        V     = abs_O / total

    s = smooth_sign(O_n, gamma)              # (batch, N)

    # Normalise price diff by max in batch
    max_prc  = prc.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    norm_prc = prc / max_prc

    terms  = (V * norm_prc * s) ** 2        # (batch, N)
    inside = terms.sum(dim=1)

    if use_hold:
        inside = inside + V_h.squeeze(1) ** 2

    loss = 1.0 - torch.sqrt(inside.clamp(min=1e-8))
    return loss.mean()


# ── Loss IV: StockLoss-Norm ───────────────────────────────────────────────────

def stockloss_norm(O: torch.Tensor, prc: torch.Tensor,
                   gamma: float = 10.0, use_hold: bool = True) -> torch.Tensor:
    """
    Loss IV — StockLoss-Norm (normalises both weights and price diff together).
    """
    N   = prc.shape[1]
    O_n = O[:, :N]
    O_h = O[:, N:]

    V   = portfolio_weights(O)[:, :N]
    s   = smooth_sign(O_n, gamma)

    numerator   = (V * prc * s)             # (batch, N)
    denominator = (V * prc).abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
    norm_terms  = numerator / denominator   # (batch, N)

    profit = norm_terms.sum(dim=1)

    hold_penalty = torch.zeros_like(profit)
    if use_hold:
        V_h = portfolio_weights(O)[:, N:]
        hold_penalty = V_h.squeeze(1)

    loss = 1.0 - (profit + hold_penalty)
    return loss.mean()


# ── Unified interface ─────────────────────────────────────────────────────────

LOSS_REGISTRY = {
    "L1": stockloss,
    "L2": stockloss_l2,       # ← active
    "L3": stockloss_max,
    "L4": stockloss_norm,
}


def get_loss_fn(name: str = "L2"):
    """
    Return the loss function by name.
    Default is L2 (StockLoss-L2 with PRC) — best in paper.
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]
