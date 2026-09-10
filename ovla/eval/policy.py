"""Inference wrapper: LIBERO obs dict -> (chunk, 7) env actions, with the training-time preprocessing mirrored.

Preprocessing parity with the dataset (RLDS-orientation JPEG frames):
  live render -> rotate 180° -> JPEG round-trip (q=95) -> (already 256²) -> uint8 (S,3,H,W), views in `view_order`.
Proprio: [eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)] -> quantile-normalised with the run's stats.
"""
from __future__ import annotations

import json
import os
import re
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from ..data.libero_hdf5 import LangTable, Normalizer, Stats
from ..model.policy import OmegaPolicy, PolicyConfig

VIEW_KEYS = {0: "agentview_image", 1: "robot0_eye_in_hand_image"}
JPEG_Q = 95


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    """robosuite convention, quat = (x, y, z, w)."""
    w = float(np.clip(q[3], -1.0, 1.0))
    den = np.sqrt(max(1.0 - w * w, 0.0))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * np.arccos(w) / den).astype(np.float32)


def normalize_instruction(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


class OmegaLiberoPolicy:
    def __init__(self, run_dir: str, ckpt_name: str | None = None, device: str = "cuda") -> None:
        meta = json.load(open(os.path.join(run_dir, "run_meta.json")))
        pc = dict(meta["policy"])
        pc["lora_targets"] = tuple(pc["lora_targets"])
        self.cfg = PolicyConfig(**pc)
        self.model = OmegaPolicy(self.cfg).to(device).eval()
        if ckpt_name is None:
            ckpt_name = sorted(f for f in os.listdir(run_dir) if f.startswith("step_") and f.endswith(".pt"))[-1]
        payload = torch.load(os.path.join(run_dir, ckpt_name), map_location="cpu")
        missing, unexpected = self.model.load_state_dict(payload["trainable"], strict=False)
        assert not unexpected, unexpected[:5]
        trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        assert not (trainable & set(missing)), sorted(trainable & set(missing))[:5]
        self.step = int(payload["step"])
        self.norm = Normalizer(Stats.from_dict(meta["stats"]))
        self.view_order = tuple(meta["view_order"])
        self.device = device
        table = LangTable()
        self._lang_by_norm = {normalize_instruction(k): v for k, v in table.idx.items()}
        self._emb = table.emb
        self._task_emb = None

    def set_task(self, instruction: str) -> None:
        key = normalize_instruction(instruction)
        if key not in self._lang_by_norm:
            raise KeyError(f"instruction not in language cache: {instruction!r}")
        self._task_emb = torch.from_numpy(self._emb[self._lang_by_norm[key]]).unsqueeze(0).to(self.device)

    @staticmethod
    def _prep_view(render: np.ndarray) -> np.ndarray:
        rot = np.ascontiguousarray(render[::-1, ::-1])
        buf = BytesIO()
        Image.fromarray(rot).save(buf, "JPEG", quality=JPEG_Q)
        img = Image.open(BytesIO(buf.getvalue())).convert("RGB")
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
        return np.array(img)

    def proprio(self, obs) -> np.ndarray:
        return np.concatenate([np.asarray(obs["robot0_eef_pos"], np.float32), quat2axisangle(np.asarray(obs["robot0_eef_quat"], np.float32)),
                               np.asarray(obs["robot0_gripper_qpos"], np.float32)]).astype(np.float32)

    @torch.no_grad()
    def __call__(self, obs) -> np.ndarray:
        assert self._task_emb is not None, "call set_task first"
        views = np.stack([self._prep_view(np.asarray(obs[VIEW_KEYS[v]])) for v in self.view_order])
        images = torch.from_numpy(views).permute(0, 3, 1, 2).unsqueeze(0).float().div(255.0).to(self.device)
        prop = torch.from_numpy(self.norm.proprio(self.proprio(obs))).unsqueeze(0).to(self.device)
        out = self.model(images, self._task_emb, prop)
        chunk = out["action"].squeeze(0).float().cpu().numpy()
        return self.norm.unaction(chunk)  # (chunk, 7) in env units
