"""
For each target BA, aggregates time series features from its
first-order neighbors (WECC adjacency), weighted by pre-computed
Pearson similarity.  Isolated nodes (e.g. LDWP) are passed through
unchanged via an identity shortcut.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialNeighborAggregation(nn.Module):
    """
    Weighted aggregation of neighbor time series, injected as a
    residual correction to the target BA's input before patch embedding.

    Args:
        n_vars   (int): number of input features per timestep (enc_in)
        dropout  (float): dropout rate
    """

    def __init__(self, n_vars: int, dropout: float = 0.1):
        super().__init__()
        # Projects aggregated neighbor features before residual addition
        self.proj = nn.Linear(n_vars, n_vars)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(n_vars)

    def forward(
        self,
        x_target: torch.Tensor,
        neighbor_x_encs: list,
        sim_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_target      : (B, L, n_vars)  target BA time series
            neighbor_x_encs: list of (B, L, n_vars), one per neighbor
                             Empty list → identity (isolated node)
            sim_weights   : (N_neighbors,) pre-computed Pearson similarities,
                             already masked by 0/1 adjacency (no negatives kept)

        Returns:
            (B, L, n_vars)  spatially enriched time series
        """
        if len(neighbor_x_encs) == 0:
            return x_target

        # Stack neighbors: (N, B, L, n_vars)
        neighbors = torch.stack(neighbor_x_encs, dim=0)

        # Normalise similarity weights; clamp negatives to 0 before softmax
        w = sim_weights.clamp(min=0.0)
        if w.sum() < 1e-8:
            # All similarities ≤ 0: fall back to uniform weights
            w = torch.ones_like(w)
        w = F.softmax(w, dim=0)          # (N,)
        w = w.view(-1, 1, 1, 1)          # broadcast over (B, L, n_vars)

        # Weighted average of neighbor features
        aggregated = (neighbors * w).sum(dim=0)   # (B, L, n_vars)

        # Project + residual + layer norm
        out = x_target + self.dropout(self.proj(aggregated))
        return self.norm(out)
