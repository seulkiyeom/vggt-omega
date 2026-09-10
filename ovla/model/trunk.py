"""VGGT-Ω aggregator with per-frame policy tokens (lang, proprio, action registers) as register-class tokens.

Per frame the token order becomes  [cam(1) | scene reg(16) | lang(1) | proprio(1) | act(n_act) | patch(H/16*W/16)].
Everything before the patches is the "special prefix" (patch_token_start = 18 + n_act): it receives no RoPE
(Ω applies RoPE only to the trailing H*W tokens) and, at register-attention layers, it is exactly the set of
tokens that exchange information across frames. The policy tokens therefore ride the pretrained register
bus for free. Frame slot semantics follow Ω: slot 0 = reference frame (we put the wrist view there by default),
slot 1 = all other frames.

Switches (for the ablation ladder):
  policy_in_register_attn  True  = policy tokens join the R-layer pool (L0/L1 default)
                           False = only cam+scene registers exchange; policy tokens pass through R layers (L1 ablation)
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from vggt_omega.models.aggregator import Aggregator, slice_expand_and_flatten

N_CAM, N_REG = 1, 16


class PolicyAggregator(Aggregator):
    def __init__(self, n_act: int = 8, policy_in_register_attn: bool = True, **kw) -> None:
        super().__init__(**kw)
        self.n_act = n_act
        self.n_policy = 2 + n_act  # lang + proprio + action registers
        self.policy_in_register_attn = policy_in_register_attn
        d = self.camera_token.shape[-1]
        # action registers: shared content, initialised later from the scene-register mean (see init_policy_tokens)
        self.action_query = nn.Parameter(torch.zeros(1, n_act, d))
        # per-slot (reference / other) additive bias so wrist and agentview action registers are distinguishable
        self.policy_slot_bias = nn.Parameter(torch.zeros(1, 2, self.n_policy, d))
        self.prefix_geo = N_CAM + N_REG            # 17: cam + scene registers
        self.patch_token_start = self.prefix_geo + self.n_policy  # 27 for n_act = 8

    @torch.no_grad()
    def init_policy_tokens(self, std: float = 1e-3) -> None:
        """Small non-zero init around the pretrained register mean (lesson from PVM/GQ-entry: not zero)."""
        mean = self.register_token.mean(dim=(0, 1, 2), keepdim=True)  # (1,1,1,D)
        self.action_query.copy_(mean[0] + std * torch.randn_like(self.action_query))
        self.policy_slot_bias.normal_(std=std)

    def forward(self, images: Tensor, lang: Tensor, proprio: Tensor):  # type: ignore[override]
        """images (B,S,3,H,W) in [0,1]; lang (B,D); proprio (B,D). Returns (cached outputs list, patch_token_start)."""
        B, S, C, H, W = images.shape
        images = ((images - self._resnet_mean) / self._resnet_std).view(B * S, C, H, W)
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)          # (B*S,1,D)
        register_token = slice_expand_and_flatten(self.register_token, B, S)      # (B*S,16,D)
        # policy tokens: same content for every frame, plus a slot bias (slot 0 = reference frame)
        pol = torch.cat([lang.unsqueeze(1), proprio.unsqueeze(1), self.action_query.expand(B, -1, -1)], dim=1)  # (B,10,D)
        pol = pol.unsqueeze(1).expand(B, S, -1, -1)                                                            # (B,S,10,D)
        slot = torch.cat([self.policy_slot_bias[:, :1], self.policy_slot_bias[:, 1:].expand(-1, S - 1, -1, -1)], dim=1)
        pol = (pol + slot).reshape(B * S, self.n_policy, -1)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        tokens = torch.cat([camera_token, register_token, pol, patch_tokens], dim=1)
        _, N, D = tokens.shape
        grid = (H // self.patch_size, W // self.patch_size)
        with torch.no_grad():
            sin, cos = self.rope_embed(H=grid[0], W=grid[1])
            rope = (sin.to(patch_tokens.device, torch.float32), cos.to(patch_tokens.device, torch.float32))
        outputs = []
        for i in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(tokens, B, S, N, D, i, rope)
            tokens = self._inter_frame(tokens, B, S, N, D, i)
            outputs.append(torch.cat([frame_tokens, tokens], dim=-1) if i in self.cached_layer_indices else None)
        return outputs, self.patch_token_start

    def _inter_frame(self, tokens, B, S, N, D, i):
        kind = self.inter_frame_attention_types[i]
        if kind == "global" or self.policy_in_register_attn:
            return self._run_inter_frame_attention_block(tokens, B, S, N, D, i, kind)
        # register attention WITHOUT policy tokens: only cam + scene registers exchange across frames
        t = tokens.view(B, S, N, D)
        geo = t[:, :, : self.prefix_geo].reshape(B, S * self.prefix_geo, D)
        geo = self._run_block(self.inter_frame_blocks[i], geo, None).view(B, S, self.prefix_geo, D)
        return torch.cat([geo, t[:, :, self.prefix_geo:]], dim=2)

    # convenience slices on the cached (B,S,N,2D) tensors
    def rows(self, cached: Tensor):
        g = self.prefix_geo
        return dict(cam=cached[:, :, :N_CAM], reg=cached[:, :, N_CAM:g], lang=cached[:, :, g:g + 1],
                    proprio=cached[:, :, g + 1:g + 2], act=cached[:, :, g + 2:g + 2 + self.n_act],
                    patch=cached[:, :, self.patch_token_start:])
