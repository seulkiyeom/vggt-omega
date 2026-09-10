"""Geometry-drift canary: run Ω's own DenseHead on the trunk of a TRAINED run and compare depth AbsRel with the
stock Ω trunk on the same LIBERO frames. Tells whether LoRA + policy tokens moved the trunk away from geometry.

  python scripts/canary_depth.py --run runs/l0_s7 [--ckpt step_0010000.pt]
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, "/NHNHOME/nota/skyeom/projects/vggt_omega_ref")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from vggt_omega.models import VGGTOmega  # noqa: E402
from ovla.eval.policy import OmegaLiberoPolicy  # noqa: E402

H5 = "/NHNHOME/nota/marcel/datasets/LIBERO_for_VGA_540625a9/libero_3d_no_noops_aligned.hdf5"
CKPT = "/NHNHOME/nota/skyeom/models/vggt-omega/vggt_omega_1b_512.pt"
LANG = "/NHNHOME/nota/marcel/datasets/LIBERO_for_VGA_540625a9/lang_cache/libero_instructions.npy"


def decode(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def absrel(pred, gt):
    v = (gt > 0.05) & (gt < 10) & np.isfinite(gt)
    p, g = pred[v], gt[v]
    s = np.median(g / np.clip(p, 1e-6, None))
    return float(np.mean(np.abs(p * s - g) / g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    dev = "cuda"
    pol = OmegaLiberoPolicy(args.run, args.ckpt)
    trunk = pol.model.trunk
    view_order = pol.view_order
    stock = VGGTOmega()
    stock.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
    stock = stock.to(dev).eval()
    dense = stock.dense_head
    lang = torch.from_numpy(np.load(LANG)[:1]).float().to(dev)
    rng = np.random.RandomState(1)
    rs, rt = [], []
    with h5py.File(H5, "r") as f:
        g = f["long"]
        eps = sorted(k for k in g if k.startswith("episode_"))
        for e in rng.choice(eps, args.n, replace=False):
            ge = g[e]
            t = int(0.5 * (ge["action"].shape[0] - 1))
            imgs = np.stack([decode(ge["image"][t, v]) for v in view_order])
            depth = ge["depth"][t].astype(np.float32)[list(view_order)]
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255).unsqueeze(0).to(dev)
            prop = torch.from_numpy(pol.norm.proprio(ge["state"][t].astype(np.float32))).unsqueeze(0).to(dev)
            with torch.no_grad():
                d0 = stock(x)["depth"][0, ..., 0].float().cpu().numpy()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cached, start = trunk(x, pol.model.proj_lang(lang), pol.model.proj_prop(prop))
                with torch.autocast("cuda", enabled=False):
                    d1, _ = dense(cached, images=x, patch_token_start=start)
                d1 = d1[0].float().cpu().numpy()
                if d1.ndim == 4:
                    d1 = d1[..., 0]
            for v in range(2):
                rs.append(absrel(d0[v], depth[v]))
                rt.append(absrel(d1[v], depth[v]))
    names = {1: "wrist", 0: "agent"}
    print(f"run {args.run} step {pol.step}: depth AbsRel stock Ω {np.mean(rs):.4f} -> trained trunk {np.mean(rt):.4f}")
    for i, v in enumerate(view_order):
        print(f"  {names[v]}: {np.mean(rs[i::2]):.4f} -> {np.mean(rt[i::2]):.4f}")


if __name__ == "__main__":
    main()
