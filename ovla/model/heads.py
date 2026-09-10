"""Action heads for the ladder.

SimpleActionHead (L0)    read the action-register rows of the LAST cached layer, average frames, MLP -> (n_act, 7).
                         No patch rows, no cross-attention: the trunk (policy tokens + LoRA) does the planning.
RegisterActionHead (L2)  the hierarchical readout: for each cached depth take [act | cam | reg | lang | proprio]
                         (optionally + one pooled patch row), mix the depths with a learned softmax weight, run
                         self-attention blocks over the flattened frame x token set, then read the action rows and
                         fuse frames. This is the register-only analogue of VGA's PVM: many depths, no 513-row
                         patch cross-attention (B-2 showed that path carries no action information).
Capacity note: RegisterActionHead is larger than SimpleActionHead, so an L2-vs-L0 difference mixes "hierarchy" with
"more head parameters". `SimpleActionHead(hidden=...)` gives a capacity-matched control.
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


class RegisterActionHead(nn.Module):
    """Multi-depth register readout. Input = list of per-depth row dicts (from trunk.rows(cached))."""

    def __init__(self, dim_in: int = 2048, dim: int = 1024, n_act: int = 8, n_depths: int = 4,
                 n_blocks: int = 4, num_heads: int = 16, action_dim: int = 7,
                 use_scene: bool = True, use_pooled_patch: bool = False, hidden: int = 1024) -> None:
        super().__init__()
        from vggt_omega.models.layers import SelfAttentionBlock

        self.n_act, self.n_depths = n_act, n_depths
        self.use_scene, self.use_pooled_patch = use_scene, use_pooled_patch
        self.depth_norm = nn.ModuleList([nn.LayerNorm(dim_in) for _ in range(n_depths)])
        self.depth_proj = nn.ModuleList([nn.Linear(dim_in, dim) for _ in range(n_depths)])
        self.depth_logit = nn.Parameter(torch.zeros(n_depths))          # uniform mix at init
        self.role_embed = nn.Parameter(torch.zeros(1, 1, 5, dim))       # act / cam / reg / lang / prop
        nn.init.normal_(self.role_embed, std=1e-3)
        self.token_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(dim=dim, num_heads=num_heads, ffn_ratio=4.0, qkv_bias=True, proj_bias=True,
                               ffn_bias=True, init_values=1e-5, use_qk_norm=False, mask_k_bias=True)
            for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, action_dim))

    def _assemble(self, rows: dict) -> tuple[Tensor, Tensor]:
        """rows of one depth -> (B, S, K, dim_in) token block and the matching role index per token."""
        parts, roles = [rows["act"]], [torch.zeros(rows["act"].shape[2], dtype=torch.long)]
        if self.use_scene:
            parts += [rows["cam"], rows["reg"]]
            roles += [torch.ones(rows["cam"].shape[2], dtype=torch.long),
                      torch.full((rows["reg"].shape[2],), 2, dtype=torch.long)]
        parts += [rows["lang"], rows["proprio"]]
        roles += [torch.full((1,), 3, dtype=torch.long), torch.full((1,), 4, dtype=torch.long)]
        if self.use_pooled_patch:
            parts.append(rows["patch"].mean(dim=2, keepdim=True))
            roles.append(torch.full((1,), 2, dtype=torch.long))  # patch pool shares the scene role
        return torch.cat(parts, dim=2), torch.cat(roles).to(parts[0].device)

    def forward(self, rows_per_depth: list[dict]) -> Tensor:
        if len(rows_per_depth) != self.n_depths:
            raise ValueError(f"expected {self.n_depths} depths, got {len(rows_per_depth)}")
        w = torch.softmax(self.depth_logit.float(), dim=0)
        mixed = None
        roles = None
        for d, rows in enumerate(rows_per_depth):
            block, roles = self._assemble(rows)
            x = self.depth_proj[d](self.depth_norm[d](block.float()))
            mixed = x * w[d] if mixed is None else mixed + x * w[d]
        B, S, K, D = mixed.shape
        x = mixed + self.role_embed[:, :, roles, :].squeeze(0).unsqueeze(0)  # (1,1,K,D) broadcast
        x = self.token_norm(x).reshape(B, S * K, D)
        for blk in self.blocks:
            x = blk(x, None)
        x = x.reshape(B, S, K, D)[:, :, : self.n_act]          # action rows
        return self.mlp(self.out_norm(x.mean(dim=1)))          # frames averaged -> (B, n_act, action_dim)
