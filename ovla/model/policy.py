"""Assemble trunk + projectors + head; load pretrained weights; apply LoRA; report parameters.

Trunks:  omega = VGGT-Ω aggregator (DINOv3 ViT-L/16, 16 registers, 5 register-attention layers), 256² input
         v1    = VGGT-1B aggregator (DINOv2 ViT-L/14, 4 registers, all-global inter-frame), 224² input
Heads:   simple   = last cached depth, action rows, frame-mean, MLP
         register = all cached depths, [act|cam|reg|lang|prop] tokens, self-attention blocks, frame-mean, MLP
Both trunks expose the same contract: forward(images, lang, proprio) -> (cached list of (B,S,N,2*D) or None,
patch_token_start) and .rows(cached) -> dict of row slices. So a head is trunk-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .heads import RegisterActionHead, SimpleActionHead
from .lora import apply_lora, lora_parameters

OMEGA_CKPT = "/NHNHOME/nota/skyeom/models/vggt-omega/vggt_omega_1b_512.pt"
V1_REPO = "facebook/VGGT-1B"


@dataclass(frozen=True)
class PolicyConfig:
    omega_ckpt: str = OMEGA_CKPT
    trunk: str = "omega"            # omega | v1
    head: str = "simple"            # simple | register
    image_size: int = 256           # 256 for omega (patch 16), 224 for v1 (patch 14): both give 256 patches
    n_act: int = 8
    action_dim: int = 7
    lang_dim: int = 1536
    proprio_dim: int = 8
    lora_rank: int = 64
    lora_alpha: float = 128.0
    lora_targets: tuple[str, ...] = ("attn.qkv", "attn.proj")
    policy_in_register_attn: bool = True
    head_dim: int = 1024
    head_blocks: int = 4
    head_hidden: int = 1024
    head_use_scene: bool = True
    head_use_pooled_patch: bool = False
    head_depth_mode: str = "multi"   # multi = the 4 cached depths | last = the last depth repeated 4x (param-identical control)


def _load_trunk_state(cfg: PolicyConfig) -> dict:
    if cfg.trunk == "omega":
        sd = torch.load(cfg.omega_ckpt, map_location="cpu")
    elif cfg.trunk == "v1":
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        sd = load_file(hf_hub_download(repo_id=V1_REPO, filename="model.safetensors"))
    else:
        raise ValueError(f"unknown trunk {cfg.trunk!r}")
    return {k[len("aggregator."):]: v for k, v in sd.items() if k.startswith("aggregator.")}


class OmegaPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.trunk == "omega":
            from .trunk import PolicyAggregator
            self.trunk = PolicyAggregator(n_act=cfg.n_act, policy_in_register_attn=cfg.policy_in_register_attn)
            expected_new = {"action_query", "policy_slot_bias"}
        elif cfg.trunk == "v1":
            from .trunk_v1 import PolicyAggregatorV1
            self.trunk = PolicyAggregatorV1(n_act=cfg.n_act)
            expected_new = {"action_query", "policy_slot_bias"}
        else:
            raise ValueError(f"unknown trunk {cfg.trunk!r}")

        missing, unexpected = self.trunk.load_state_dict(_load_trunk_state(cfg), strict=False)
        assert not unexpected, unexpected[:5]
        assert set(missing) == expected_new, sorted(missing)[:8]
        self.trunk.init_policy_tokens()
        for p in self.trunk.parameters():
            p.requires_grad_(False)
        for p in (self.trunk.action_query, self.trunk.policy_slot_bias):
            p.requires_grad_(True)
        self.n_lora = apply_lora(self.trunk, cfg.lora_targets, cfg.lora_rank, cfg.lora_alpha)

        d = self.trunk.camera_token.shape[-1]
        self.proj_lang = nn.Sequential(nn.LayerNorm(cfg.lang_dim), nn.Linear(cfg.lang_dim, d))
        self.proj_prop = nn.Sequential(nn.Linear(cfg.proprio_dim, d), nn.GELU(), nn.Linear(d, d))
        if cfg.head == "simple":
            self.head = SimpleActionHead(dim_in=2 * d, hidden=cfg.head_hidden, action_dim=cfg.action_dim)
        elif cfg.head == "register":
            self.head = RegisterActionHead(dim_in=2 * d, dim=cfg.head_dim, n_act=cfg.n_act,
                                           n_depths=len(self.trunk.cached_layer_indices), n_blocks=cfg.head_blocks,
                                           action_dim=cfg.action_dim, use_scene=cfg.head_use_scene,
                                           use_pooled_patch=cfg.head_use_pooled_patch, hidden=cfg.head_hidden)
        else:
            raise ValueError(f"unknown head {cfg.head!r}")

    def forward(self, images: Tensor, lang: Tensor, proprio: Tensor) -> dict:
        if images.shape[-1] != self.cfg.image_size:
            b, s = images.shape[:2]
            images = F.interpolate(images.flatten(0, 1), size=(self.cfg.image_size, self.cfg.image_size),
                                   mode="bilinear", align_corners=False,
                                   antialias=self.cfg.image_size < images.shape[-1]).view(b, s, 3, self.cfg.image_size, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cached, start = self.trunk(images, self.proj_lang(lang), self.proj_prop(proprio))
        kept = [c for c in cached if c is not None]
        if self.cfg.head == "simple":
            action = self.head(self.trunk.rows(kept[-1])["act"])
        else:
            if self.cfg.head_depth_mode == "last":
                kept = [kept[-1]] * len(kept)      # same parameters, no hierarchy
            elif self.cfg.head_depth_mode != "multi":
                raise ValueError(f"unknown head_depth_mode {self.cfg.head_depth_mode!r}")
            action = self.head([self.trunk.rows(c) for c in kept])
        return {"action": action, "cached": cached, "patch_token_start": start}

    def param_report(self) -> dict:
        tot = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for p in lora_parameters(self.trunk))
        return dict(total_M=round(tot / 1e6, 3), trainable_M=round(train / 1e6, 3), lora_M=round(lora / 1e6, 3),
                    n_lora_modules=self.n_lora,
                    head_M=round(sum(p.numel() for p in self.head.parameters()) / 1e6, 3),
                    proj_M=round(sum(p.numel() for p in list(self.proj_lang.parameters())
                                     + list(self.proj_prop.parameters())) / 1e6, 3),
                    tokens_K=round((self.trunk.action_query.numel() + self.trunk.policy_slot_bias.numel()) / 1e3, 1))
