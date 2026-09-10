"""LIBERO_for_VGA hdf5 → (images, proprio, lang, action chunk) samples, with quantile normalisation.

Conventions (OFT / VGA parity, verified against the data on 2026-09-10):
  action (7): xyz(3) rot(3) gripper(1); gripper is already ±1 in the file (binarised). Per-dim quantile q01/q99 → [-1, 1].
  state  (8): eef_pos(3), axis-angle(3), gripper_qpos(2); same quantile scaling ("bounds_q99").
  images: JPEG bytes, RLDS orientation, 256×256, 2 views (0 = agentview, 1 = wrist). We feed them as stored.
  chunk : actions[t : t+8], padded by repeating the last action at episode end.
  lang  : gte-Qwen2-1.5B embedding (1536-d) looked up from the instruction cache (40 rows).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

H5_DEFAULT = "/NHNHOME/nota/marcel/datasets/LIBERO_for_VGA_540625a9/libero_3d_no_noops_aligned.hdf5"
LANG_DIR = "/NHNHOME/nota/marcel/datasets/LIBERO_for_VGA_540625a9/lang_cache"


@dataclass(frozen=True)
class Stats:
    action_lo: np.ndarray
    action_hi: np.ndarray
    proprio_lo: np.ndarray
    proprio_hi: np.ndarray

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("action_lo", "action_hi", "proprio_lo", "proprio_hi")}

    @staticmethod
    def from_dict(d: dict) -> "Stats":
        return Stats(**{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})


def _scale(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (x - lo) / np.maximum(hi - lo, 1e-6) - 1.0, -1.0, 1.0)


def _unscale(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return 0.5 * (y + 1.0) * (hi - lo) + lo


class Normalizer:
    def __init__(self, stats: Stats) -> None:
        self.s = stats

    def action(self, a: np.ndarray) -> np.ndarray:
        out = _scale(a, self.s.action_lo, self.s.action_hi)
        out[..., 6] = np.sign(a[..., 6] + 1e-9)  # keep gripper strictly ±1
        return out.astype(np.float32)

    def unaction(self, y: np.ndarray) -> np.ndarray:
        out = _unscale(y, self.s.action_lo, self.s.action_hi)
        out[..., 6] = np.where(y[..., 6] > 0, 1.0, -1.0)
        return out.astype(np.float32)

    def proprio(self, s: np.ndarray) -> np.ndarray:
        return _scale(s, self.s.proprio_lo, self.s.proprio_hi).astype(np.float32)


def compute_stats(h5_path: str, suite: str, q: tuple[float, float] = (0.01, 0.99)) -> Stats:
    with h5py.File(h5_path, "r") as f:
        g = f[suite]
        eps = [k for k in g if k.startswith("episode_")]
        A = np.concatenate([g[e]["action"][:] for e in eps]).astype(np.float64)
        S = np.concatenate([g[e]["state"][:] for e in eps]).astype(np.float64)
    return Stats(np.quantile(A, q[0], 0).astype(np.float32), np.quantile(A, q[1], 0).astype(np.float32),
                 np.quantile(S, q[0], 0).astype(np.float32), np.quantile(S, q[1], 0).astype(np.float32))


def load_or_compute_stats(h5_path: str, suite: str, cache_dir: str) -> Stats:
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"stats_{suite}.json")
    if os.path.exists(p):
        return Stats.from_dict(json.load(open(p)))
    st = compute_stats(h5_path, suite)
    json.dump(st.to_dict(), open(p, "w"), indent=1)
    return st


class LangTable:
    """Instruction -> embedding. `npy` overrides the default cache (used by the language-fix arms)."""

    def __init__(self, lang_dir: str = LANG_DIR, npy: str | None = None) -> None:
        names_path = os.path.join(os.path.dirname(npy) if npy else lang_dir, "libero_instructions.instructions.json")
        if not os.path.exists(names_path):
            names_path = os.path.join(lang_dir, "libero_instructions.instructions.json")
        names = json.load(open(names_path))
        self.path = npy or os.path.join(lang_dir, "libero_instructions.npy")
        self.emb = np.load(self.path).astype(np.float32)
        if self.emb.shape[0] != len(names):
            raise ValueError(f"{self.path}: {self.emb.shape[0]} rows for {len(names)} instructions")
        self.idx = {n: i for i, n in enumerate(names)}

    def __call__(self, instruction: str) -> np.ndarray:
        return self.emb[self.idx[instruction]]


def decode_jpeg(b: bytes) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class LiberoChunkDataset(Dataset):
    """One item = one timestep t of one episode. Views are returned in the order given by `view_order`
    (default wrist first, so that Ω's reference-frame slot 0 is the wrist view)."""

    def __init__(self, h5_path: str, suite: str, normalizer: Normalizer, chunk: int = 8,
                 view_order: tuple[int, ...] = (1, 0), lang: LangTable | None = None) -> None:
        self.h5_path, self.suite, self.norm, self.chunk, self.view_order = h5_path, suite, normalizer, chunk, view_order
        self.lang = lang or LangTable()
        self._f = None
        with h5py.File(h5_path, "r") as f:
            g = f[suite]
            self.episodes = sorted(k for k in g if k.startswith("episode_"))
            self.lengths = [int(g[e]["action"].shape[0]) for e in self.episodes]
            self.instr = [str(g[e].attrs["language_instruction"]) for e in self.episodes]
        self.index = [(i, t) for i, L in enumerate(self.lengths) for t in range(L)]

    def __len__(self) -> int:
        return len(self.index)

    def _file(self):
        if self._f is None:  # lazily opened per worker
            self._f = h5py.File(self.h5_path, "r")
        return self._f

    def __getitem__(self, k: int) -> dict:
        i, t = self.index[k]
        g = self._file()[self.suite][self.episodes[i]]
        L = self.lengths[i]
        acts = g["action"][t:min(t + self.chunk, L)].astype(np.float32)
        if len(acts) < self.chunk:
            acts = np.concatenate([acts, np.repeat(acts[-1:], self.chunk - len(acts), 0)], 0)
        imgs = np.stack([decode_jpeg(g["image"][t, v]) for v in self.view_order])  # (S,H,W,3) uint8
        return {
            "images": torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous(),  # uint8 (S,3,H,W); /255 on GPU
            "proprio": torch.from_numpy(self.norm.proprio(g["state"][t].astype(np.float32))),
            "lang": torch.from_numpy(self.lang(self.instr[i])),
            "action": torch.from_numpy(self.norm.action(acts)),  # (chunk, 7) in [-1, 1]
        }
