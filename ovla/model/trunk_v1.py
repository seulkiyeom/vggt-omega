"""VGGT-1B (v1) aggregator with the same per-frame policy tokens — the matched control for the Ω trunk.

Identical policy-token treatment to `PolicyAggregator`, on the v1 trunk instead:
  per frame  [cam(1) | reg(4) | lang(1) | proprio(1) | act(n_act) | patch(P)]  -> patch_start_idx = 6 + n_act
v1 differences that are intrinsic to the model (not choices of ours):
  DINOv2 ViT-L/14 tokenizer (so the input must be a multiple of 14: we use 224 -> 16x16 = 256 patches,
  the same patch count as Ω at 256), 4 registers instead of 16, every inter-frame block is global
  (no register-attention layers), RoPE supplied as per-token positions with 0 for the special prefix.
Cached outputs are concat[frame, global] = 2048-d at layers (4, 11, 17, 23), same contract as Ω,
so the same action heads run on either trunk unchanged.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch
from torch import Tensor, nn

# The v1 model code is not part of this fork; point at a VGGT (v1) checkout. Override with VGGT_V1_PATH.
V1_PATH = os.environ.get("VGGT_V1_PATH", "/NHNHOME/nota/skyeom/projects/vggt_ref")
if importlib.util.find_spec("vggt") is None:
    if not os.path.isdir(os.path.join(V1_PATH, "vggt")):
        raise ImportError(f"VGGT v1 code not found at {V1_PATH!r}; set VGGT_V1_PATH to a vggt checkout")
    sys.path.insert(0, V1_PATH)

from vggt.models.aggregator import Aggregator as V1Aggregator  # noqa: E402
from vggt.models.aggregator import slice_expand_and_flatten  # noqa: E402

N_CAM_V1, N_REG_V1 = 1, 4


class PolicyAggregatorV1(V1Aggregator):
    def __init__(self, n_act: int = 8, **kw) -> None:
        super().__init__(**kw)
        self.n_act = n_act
        self.n_policy = 2 + n_act
        d = self.camera_token.shape[-1]
        self.action_query = nn.Parameter(torch.zeros(1, n_act, d))
        self.policy_slot_bias = nn.Parameter(torch.zeros(1, 2, self.n_policy, d))
        self.prefix_geo = N_CAM_V1 + N_REG_V1                     # 5
        self.patch_start_idx = self.prefix_geo + self.n_policy     # 15 for n_act = 8
        # kept under the Ω name too, so heads/probes can use one attribute for both trunks
        self.patch_token_start = self.patch_start_idx
        self.cached_layer_indices = set(self.cached_layer_indices)

    @torch.no_grad()
    def init_policy_tokens(self, std: float = 1e-3) -> None:
        mean = self.register_token.mean(dim=(0, 1, 2), keepdim=True)
        self.action_query.copy_(mean[0] + std * torch.randn_like(self.action_query))
        self.policy_slot_bias.normal_(std=std)

    def _policy_tokens(self, lang: Tensor, proprio: Tensor, B: int, S: int) -> Tensor:
        pol = torch.cat([lang.unsqueeze(1), proprio.unsqueeze(1), self.action_query.expand(B, -1, -1)], dim=1)
        pol = pol.unsqueeze(1).expand(B, S, -1, -1)
        slot = self.policy_slot_bias[:, :1] if S == 1 else torch.cat(
            [self.policy_slot_bias[:, :1], self.policy_slot_bias[:, 1:].expand(-1, S - 1, -1, -1)], dim=1)
        return (pol + slot).reshape(B * S, self.n_policy, -1)

    def forward(self, images: Tensor, lang: Tensor, proprio: Tensor):  # type: ignore[override]
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"expected 3 channels, got {C_in}")
        images = ((images - self._resnet_mean) / self._resnet_std).view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        pol = self._policy_tokens(lang, proprio, B, S)
        tokens = torch.cat([camera_token, register_token, pol, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
            pos = pos + 1  # v1 convention: patch positions are 1-indexed, the special prefix gets 0
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, device=images.device, dtype=pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape
        frame_idx = global_idx = 0
        outputs: list[Tensor | None] = []
        for _ in range(self.aa_block_num):
            frame_inter = global_inter = None
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_inter = self._process_frame_attention(tokens, B, S, P, C, frame_idx, pos=pos)
                elif attn_type == "global":
                    tokens, global_idx, global_inter = self._process_global_attention(tokens, B, S, P, C, global_idx, pos=pos)
                else:
                    raise ValueError(f"unknown attention type {attn_type}")
            for i in range(len(frame_inter)):
                layer_idx = len(outputs)
                outputs.append(torch.cat([frame_inter[i], global_inter[i]], dim=-1)
                               if layer_idx in self.cached_layer_indices else None)
        return outputs, self.patch_start_idx

    def rows(self, cached: Tensor):
        g = self.prefix_geo
        return dict(cam=cached[:, :, :N_CAM_V1], reg=cached[:, :, N_CAM_V1:g], lang=cached[:, :, g:g + 1],
                    proprio=cached[:, :, g + 1:g + 2], act=cached[:, :, g + 2:g + 2 + self.n_act],
                    patch=cached[:, :, self.patch_start_idx:])
