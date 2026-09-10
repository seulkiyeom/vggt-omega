"""Stage 1 smoke: build the Ω policy trunk with per-frame policy tokens and check it PHYSICALLY.

Checks (all printed, all must hold before any training):
  S1-a  token layout: per-frame N = 1 + 16 + 10 + 256 = 283 at 256², patch_token_start = 27, cached layers (4,11,17,23)
  S1-b  parameters: frozen Ω aggregator weights loaded strictly (aggregator.* keys), trainable = LoRA + policy tokens +
        projectors + head only; LoRA modules count = 2 * 48 blocks = 96 (qkv, proj in frame + inter-frame blocks)
  S1-c  identity at init: LoRA B = 0 ⇒ patch/cam/register rows of the trunk change ONLY through the inserted policy
        tokens. Quantify that perturbation on real LIBERO frames: depth AbsRel of Ω's own DenseHead fed with
        (i) the stock aggregator and (ii) the policy aggregator (policy tokens present, LoRA identity).
  S1-d  register-attention switch: with policy_in_register_attn=False the cam/reg rows must equal (i) up to bf16 noise
        at R layers only if policy tokens never touch them... (they still share F/G layers) -> report the delta instead.
  S1-e  gradients: backward of an L1 action loss reaches LoRA, policy tokens, projectors, head; frozen params get none.
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, "/NHNHOME/nota/skyeom/projects/vggt_omega_ref")
sys.path.insert(0, "/NHNHOME/nota/skyeom/projects/vggt_omega_vla/src")
from vggt_omega.models import VGGTOmega  # noqa: E402
from ovla.model.policy import OmegaPolicy, PolicyConfig  # noqa: E402

H5 = "/NHNHOME/nota/marcel/datasets/LIBERO_for_VGA_540625a9/libero_3d_no_noops_aligned.hdf5"
CKPT = "/NHNHOME/nota/skyeom/models/vggt-omega/vggt_omega_1b_512.pt"
LANG_NPY = "/NHNHOME/nota/marcel/datasets/LIBERO_for_VGA_540625a9/lang_cache/libero_instructions.npy"


def decode(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def load(n=12, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    with h5py.File(H5, "r") as f:
        g = f["long"]
        eps = sorted(k for k in g if k.startswith("episode_"))
        for e in rng.choice(eps, n, replace=False):
            ge = g[e]
            t = int(0.5 * (ge["action"].shape[0] - 1))
            out.append(dict(img=np.stack([decode(ge["image"][t, v]) for v in range(2)]), depth=ge["depth"][t].astype(np.float32),
                            state=ge["state"][t].astype(np.float32)))
    return out


def absrel(pred, gt):
    v = (gt > 0.05) & (gt < 10) & np.isfinite(gt)
    p, g = pred[v], gt[v]
    s = np.median(g / np.clip(p, 1e-6, None))
    return float(np.mean(np.abs(p * s - g) / g))


def main():
    dev = "cuda"
    torch.manual_seed(0)
    cfg = PolicyConfig(omega_ckpt=CKPT)
    t0 = time.time()
    pol = OmegaPolicy(cfg).to(dev).eval()
    print(f"built in {time.time() - t0:.0f}s")
    rep = pol.param_report()
    print("S1-b params:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()})
    assert rep["n_lora_modules"] == 96, rep["n_lora_modules"]
    trainable_names = sorted({n.split(".")[0] + ("." + n.split(".")[1] if n.startswith("trunk") else "") for n, p in pol.named_parameters() if p.requires_grad})
    print("S1-b trainable groups:", trainable_names)

    frames = load()
    lang_tab = np.load(LANG_NPY)
    B = 4
    imgs = torch.from_numpy(np.stack([fr["img"] for fr in frames[:B]])).permute(0, 1, 4, 2, 3).float().div(255).to(dev)  # (B,S,3,H,W)
    lang = torch.from_numpy(lang_tab[:B]).float().to(dev)
    prop = torch.from_numpy(np.stack([fr["state"] for fr in frames[:B]])).to(dev)
    with torch.no_grad():
        out = pol(imgs, lang, prop)
    cached = [c for c in out["cached"] if c is not None]
    print(f"S1-a layout: images {tuple(imgs.shape)} -> cached x{len(cached)} each {tuple(cached[-1].shape)}; "
          f"patch_token_start {out['patch_token_start']}; action {tuple(out['action'].shape)}")
    assert cached[-1].shape[2] == 283 and out["patch_token_start"] == 27 and out["action"].shape == (B, 8, 7)
    rows = pol.trunk.rows(cached[-1])
    print("S1-a rows:", {k: tuple(v.shape[2:3]) for k, v in rows.items()})

    # ---- S1-c: perturbation of the pretrained depth by inserting policy tokens (LoRA identity) ----
    stock = VGGTOmega()
    stock.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=True)
    stock = stock.to(dev).eval()
    dense = stock.dense_head
    ar_stock, ar_pol, ar_pol_noR = [], [], []
    for fr in frames:
        x = torch.from_numpy(fr["img"]).permute(0, 3, 1, 2).float().div(255).unsqueeze(0).to(dev)
        with torch.no_grad():
            d0 = stock(x)["depth"][0, ..., 0].float().cpu().numpy()
            l = torch.from_numpy(lang_tab[:1]).float().to(dev)
            p = torch.from_numpy(fr["state"][None]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cached, start = pol.trunk(x, pol.proj_lang(l), pol.proj_prop(p))
            with torch.autocast("cuda", enabled=False):
                d1, _ = dense(cached, images=x, patch_token_start=start)
            d1 = d1[0].float().cpu().numpy()
            if d1.ndim == 4:
                d1 = d1[..., 0]
            pol.trunk.policy_in_register_attn = False
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cached2, _ = pol.trunk(x, pol.proj_lang(l), pol.proj_prop(p))
            pol.trunk.policy_in_register_attn = True
            with torch.autocast("cuda", enabled=False):
                d2, _ = dense(cached2, images=x, patch_token_start=start)
            d2 = d2[0].float().cpu().numpy()
            if d2.ndim == 4:
                d2 = d2[..., 0]
        for v in range(2):
            ar_stock.append(absrel(d0[v], fr["depth"][v])); ar_pol.append(absrel(d1[v], fr["depth"][v])); ar_pol_noR.append(absrel(d2[v], fr["depth"][v]))
    print(f"S1-c depth AbsRel on {len(frames)} frames x 2 views: stock Ω {np.mean(ar_stock):.4f} | +policy tokens (LoRA identity) "
          f"{np.mean(ar_pol):.4f} | +policy tokens, excluded from R layers {np.mean(ar_pol_noR):.4f}")
    print(f"      per-view: agent stock {np.mean(ar_stock[0::2]):.4f} -> {np.mean(ar_pol[0::2]):.4f}; wrist stock {np.mean(ar_stock[1::2]):.4f} -> {np.mean(ar_pol[1::2]):.4f}")

    # ---- S1-e: gradient reach ----
    pol.train()
    out = pol(imgs, lang, prop)
    loss = (out["action"].float() - torch.zeros_like(out["action"].float())).abs().mean()
    loss.backward()
    got = {n for n, p in pol.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    frozen_with_grad = [n for n, p in pol.named_parameters() if not p.requires_grad and p.grad is not None]
    print(f"S1-e grads: {len(got)} tensors received gradient; frozen tensors with grad: {len(frozen_with_grad)}; "
          f"action_query {'trunk.action_query' in got}, slot_bias {'trunk.policy_slot_bias' in got}, "
          f"lora_b any {any('lora_b' in n for n in got)}, head {any(n.startswith('head') for n in got)}, proj {any(n.startswith('proj') for n in got)}")
    print(f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
