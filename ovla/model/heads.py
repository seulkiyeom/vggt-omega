"""Action heads for the ladder.

SimpleActionHead (L0): read the action-register rows of the LAST cached layer, average the frames, MLP -> (n_act, 7).
No patch rows, no cross-attention: the trunk (policy tokens + LoRA) does the planning.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class SimpleActionHead(nn.Module):
    def __init__(self, dim_in: int = 2048, hidden: int = 1024, action_dim: int = 7) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.mlp = nn.Sequential(nn.Linear(dim_in, hidden), nn.GELU(), nn.Linear(hidden, action_dim))

    def forward(self, act_rows: Tensor) -> Tensor:
        """act_rows (B, S, n_act, dim_in) -> (B, n_act, action_dim); frames averaged (view fusion = mean)."""
        return self.mlp(self.norm(act_rows.float().mean(dim=1)))
