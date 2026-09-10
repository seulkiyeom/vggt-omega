"""Assemble trunk + projectors + head; load Ω weights; apply LoRA; count parameters."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .heads import SimpleActionHead
from .lora import apply_lora, lora_parameters
from .trunk import PolicyAggregator


@dataclass(frozen=True)
class PolicyConfig:
    omega_ckpt: str
    n_act: int = 8
    action_dim: int = 7
    lang_dim: int = 1536
    proprio_dim: int = 8
    lora_rank: int = 64
    lora_alpha: float = 128.0
    lora_targets: tuple[str, ...] = ("attn.qkv", "attn.proj")
    policy_in_register_attn: bool = True
    head: str = "simple"


class OmegaPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = PolicyAggregator(n_act=cfg.n_act, policy_in_register_attn=cfg.policy_in_register_attn)
        sd = torch.load(cfg.omega_ckpt, map_location="cpu")
        agg = {k[len("aggregator."):]: v for k, v in sd.items() if k.startswith("aggregator.")}
        missing, unexpected = self.trunk.load_state_dict(agg, strict=False)
        assert not unexpected, unexpected[:5]
        assert set(missing) == {"action_query", "policy_slot_bias"}, missing[:8]
        self.trunk.init_policy_tokens()
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        for p in (self.trunk.action_query, self.trunk.policy_slot_bias):
            p.requires_grad_(True)
        self.n_lora = apply_lora(self.trunk, cfg.lora_targets, cfg.lora_rank, cfg.lora_alpha)
        d = self.trunk.camera_token.shape[-1]
        self.proj_lang = nn.Sequential(nn.LayerNorm(cfg.lang_dim), nn.Linear(cfg.lang_dim, d))
        self.proj_prop = nn.Sequential(nn.Linear(cfg.proprio_dim, d), nn.GELU(), nn.Linear(d, d))
        if cfg.head != "simple":
            raise NotImplementedError(cfg.head)
        self.head = SimpleActionHead(dim_in=2 * d, action_dim=cfg.action_dim)

    def forward(self, images: Tensor, lang: Tensor, proprio: Tensor) -> dict:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cached, start = self.trunk(images, self.proj_lang(lang), self.proj_prop(proprio))
        last = [c for c in cached if c is not None][-1]
        rows = self.trunk.rows(last)
        action = self.head(rows["act"])
        return {"action": action, "cached": cached, "patch_token_start": start}

    def param_report(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for p in lora_parameters(self.trunk))
        return dict(total_M=tot / 1e6, trainable_M=train / 1e6, lora_M=lora / 1e6, n_lora_modules=self.n_lora,
                    head_M=sum(p.numel() for p in self.head.parameters()) / 1e6,
                    proj_M=sum(p.numel() for p in list(self.proj_lang.parameters()) + list(self.proj_prop.parameters())) / 1e6)
