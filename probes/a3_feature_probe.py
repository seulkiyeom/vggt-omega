"""A-3: does better depth mean better ACTION-relevant features? Linear probes on frozen trunks.

A-2 showed VGGT-Omega's depth is more accurate than VGGT-1B's on our LIBERO frames. That does not
by itself matter for a policy. This probe asks the operative question directly: how much
action-relevant information is *linearly decodable* from each frozen trunk's tokens? Same protocol
as the geometric-foundation-model study (arXiv:2605.24642) but with action targets added.

Arms: v1@224, om@224, om@256, om@512 (joint 2-view input, frozen, last cached layer 23, 2048-d).
Feature sets (all built from the same forward pass, so the comparison is per-arm within-frame):
  patch    mean-pooled patch tokens per view                 (2 x 2048)
  reg      camera token + mean register token per view        (4 x 2048)  <- "registers as interface"
  all      patch + reg                                        (6 x 2048)
Targets (ridge regression, episode-disjoint train/test split, alpha by 5-fold CV inside train):
  ee_xyz     end-effector world position, cm         -> R2 and RMS error in cm
  act0       next action (7)                          -> R2
  act_chunk  next 8 actions (56)                      -> R2
  grip       gripper command sign                     -> accuracy
  depth_ee   GT depth at the EE pixel, cm             -> R2 (geometry sanity, the A-2 quantity)
  obj_rel    (nearest-object xyz - EE xyz) is NOT available for all suites -> omitted
Reading: if Omega's depth advantage carries into action-relevant probes, ee_xyz/act R2 improve too.
If depth_ee improves but act R2 does not, depth accuracy is NOT the operative variable for a policy.
No pre-registered pass/fail: this is a descriptive probe (its job is to say which quantity moves).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

H5 = "/NHNHOME/nota/marcel/datasets/LIBERO_for_VGA_540625a9/libero_3d_no_noops_aligned.hdf5"
OMEGA_CKPT = "/NHNHOME/nota/skyeom/models/vggt-omega/vggt_omega_1b_512.pt"
ORIENT = {0: "asis", 1: "rot180"}  # stored extrinsics vs stored RLDS images (verified 2026-09-09)


def decode(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def project(K, E_c2w, p):
    Xc = np.linalg.inv(E_c2w) @ np.append(p, 1.0)
    x, y, z = Xc[:3]
    return K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2], z


def load_frames(suites, n_eps, per_ep, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    with h5py.File(H5, "r") as f:
        for s in suites:
            K = f[s]["intrinsics"][:]
            eps = sorted(k for k in f[s].keys() if k.startswith("episode_"))
            eps = [eps[i] for i in rng.choice(len(eps), min(n_eps, len(eps)), replace=False)]
            for e in eps:
                g = f[s][e]
                L = g["action"].shape[0]
                acts = g["action"][:]
                for frac in np.linspace(0.1, 0.9, per_ep):
                    t = int(round(frac * (L - 1)))
                    chunk = acts[t:t + 8]
                    if len(chunk) < 8:
                        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], 8 - len(chunk), 0)], 0)
                    ee = g["state"][t, :3].astype(np.float64)
                    u, v, z = project(K[0], g["extrinsics"][t][0], ee)
                    H = g["depth"].shape[-1]
                    ui, vi = int(np.clip(round(u), 0, H - 1)), int(np.clip(round(v), 0, H - 1))
                    d_ee = float(g["depth"][t, 0, vi, ui])
                    out.append(dict(suite=s, ep=f"{s}/{e}", t=t,
                                    img=[decode(g["image"][t, vw]) for vw in range(2)],
                                    ee=ee, chunk=chunk.astype(np.float64), depth_ee=d_ee))
    return out


def to_input(imgs, res):
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
    if res != x.shape[-1]:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=(res < x.shape[-1]))
    return x


def extract(name, model, x, dev):
    """Return dict of feature blocks from the last cached layer: patch mean, cam, reg mean (per view)."""
    x = x.to(dev)
    if x.ndim == 4:
        x = x.unsqueeze(0)  # both aggregators expect (B, S, 3, H, W)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        toks, start = model.aggregator(x)
    t = [tt for tt in toks if tt is not None][-1].float()  # (1, S, N, 2048)
    t = t[0]  # (S, N, 2048)
    cam = t[:, 0]                      # (S, 2048)
    reg = t[:, 1:start].mean(1)        # (S, 2048)
    patch = t[:, start:].mean(1)       # (S, 2048)
    return dict(cam=cam.cpu().numpy(), reg=reg.cpu().numpy(), patch=patch.cpu().numpy())


def ridge_eval(Xtr, ytr, Xte, yte, alphas=(1e1, 1e2, 1e3, 1e4, 1e5, 1e6), folds=5, seed=0):
    """Dual-form ridge (n < d). Returns (R2 per target mean, predictions on test)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean(0)
    Yc = ytr - ym
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(A))
    best, best_a = -np.inf, alphas[0]
    for a in alphas:
        sc = []
        for f in range(folds):
            va = idx[f::folds]
            tr = np.setdiff1d(idx, va)
            K = A[tr] @ A[tr].T
            dual = np.linalg.solve(K + a * np.eye(len(tr)), Yc[tr] - Yc[tr].mean(0))
            pred = (A[va] @ A[tr].T) @ dual + Yc[tr].mean(0)
            ss = ((Yc[va] - pred) ** 2).sum()
            st = ((Yc[va] - Yc[tr].mean(0)) ** 2).sum()
            sc.append(1 - ss / max(st, 1e-12))
        if np.mean(sc) > best:
            best, best_a = np.mean(sc), a
    K = A @ A.T
    dual = np.linalg.solve(K + best_a * np.eye(len(A)), Yc)
    pred = (B @ A.T) @ dual + ym
    ss = ((yte - pred) ** 2).sum(0)
    st = ((yte - ym) ** 2).sum(0)
    r2 = float(np.mean(1 - ss / np.maximum(st, 1e-12)))
    return r2, pred, best_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=["long", "spatial"])
    ap.add_argument("--n-eps", type=int, default=30)
    ap.add_argument("--per-ep", type=int, default=8)
    ap.add_argument("--arms", nargs="+", default=["v1@224", "om@224", "om@256", "om@512"])
    ap.add_argument("--out", default="/NHNHOME/nota/skyeom/omega_probes/a3")
    ap.add_argument("--feat-cache", default=None, help="reuse a saved features npz and skip the GPU pass")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_eps, args.per_ep = 4, 3
    os.makedirs(args.out, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    print(__doc__, flush=True)

    if args.feat_cache and os.path.exists(args.feat_cache):
        z = np.load(args.feat_cache, allow_pickle=True)
        feats = z["feats"].item()
        meta = z["meta"].item()
        print(f"loaded cached features from {args.feat_cache}", flush=True)
    else:
        frames = load_frames(args.suites, args.n_eps, args.per_ep)
        print(f"frames: {len(frames)} from {args.suites}", flush=True)
        dev = "cuda"
        sys.path.insert(0, "/NHNHOME/nota/skyeom/projects/vggt_ref")
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        models = {}
        if any(a.startswith("v1") for a in args.arms):
            from vggt.models.vggt import VGGT
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            models["v1"] = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
        if any(a.startswith("om") for a in args.arms):
            from vggt_omega.models import VGGTOmega
            m = VGGTOmega(enable_camera=False, enable_depth=False)
            sd = torch.load(OMEGA_CKPT, map_location="cpu")
            sd = {k: v for k, v in sd.items() if k.startswith("aggregator.")}
            missing, unexpected = m.load_state_dict(sd, strict=False)
            assert not [k for k in missing if k.startswith("aggregator.")], missing[:5]
            models["om"] = m.to(dev).eval()
            print(f"omega aggregator loaded ({len(sd)} keys)", flush=True)
        feats = {a: {b: [] for b in ("cam", "reg", "patch")} for a in args.arms}
        meta = dict(ep=[], suite=[], ee=[], chunk=[], depth_ee=[])
        t0 = time.time()
        for i, fr in enumerate(frames):
            for a in args.arms:
                nm, res = a.split("@")
                f = extract(nm, models[nm], to_input(fr["img"], int(res)), dev)
                for b in ("cam", "reg", "patch"):
                    feats[a][b].append(f[b].reshape(-1))  # (2*2048,)
            meta["ep"].append(fr["ep"]); meta["suite"].append(fr["suite"])
            meta["ee"].append(fr["ee"]); meta["chunk"].append(fr["chunk"].reshape(-1))
            meta["depth_ee"].append(fr["depth_ee"])
            if i % 50 == 0:
                print(f"  {i}/{len(frames)}  {time.time() - t0:.0f}s", flush=True)
        for a in args.arms:
            for b in feats[a]:
                feats[a][b] = np.stack(feats[a][b]).astype(np.float32)
        for k in ("ee", "chunk", "depth_ee"):
            meta[k] = np.asarray(meta[k], dtype=np.float64)
        np.savez_compressed(f"{args.out}/a3_feats_{tag}.npz", feats=feats, meta=meta)
        print(f"features saved: {args.out}/a3_feats_{tag}.npz", flush=True)

    eps = np.array(meta["ep"])
    uniq = sorted(set(eps.tolist()))
    rng = np.random.RandomState(0)
    te_eps = set(np.array(uniq)[rng.permutation(len(uniq))[: max(1, len(uniq) // 3)]].tolist())
    te = np.array([e in te_eps for e in eps])
    tr = ~te
    print(f"\nsplit: {tr.sum()} train / {te.sum()} test frames, episode-disjoint "
          f"({len(uniq) - len(te_eps)}/{len(te_eps)} episodes)", flush=True)

    targets = {
        "ee_xyz_cm": meta["ee"] * 100.0,
        "act0": meta["chunk"].reshape(len(eps), 8, 7)[:, 0, :],
        "act_chunk": meta["chunk"],
        "depth_ee_cm": meta["depth_ee"].reshape(-1, 1) * 100.0,
        "grip": np.sign(meta["chunk"].reshape(len(eps), 8, 7)[:, 0, 6:7]),
    }
    sets = {"patch": ("patch",), "reg": ("cam", "reg"), "all": ("cam", "reg", "patch")}
    res = {}
    print(f"\n{'arm':9s} {'featset':8s} " + " ".join(f"{k:>13s}" for k in targets) + f" {'ee_rms_cm':>10s} {'grip_acc':>9s}")
    for a in args.arms:
        for sname, blocks in sets.items():
            X = np.concatenate([feats[a][b] for b in blocks], 1)
            row, extra = {}, {}
            for tname, Y in targets.items():
                r2, pred, alpha = ridge_eval(X[tr], Y[tr], X[te], Y[te])
                row[tname] = r2
                if tname == "ee_xyz_cm":
                    extra["ee_rms_cm"] = float(np.sqrt(((pred - Y[te]) ** 2).sum(1)).mean())
                if tname == "grip":
                    extra["grip_acc"] = float((np.sign(pred[:, 0]) == Y[te][:, 0]).mean())
                row[f"{tname}_alpha"] = alpha
            res[f"{a}|{sname}"] = {**row, **extra, "dim": int(X.shape[1])}
            print(f"{a:9s} {sname:8s} " + " ".join(f"{row[k]:13.3f}" for k in targets)
                  + f" {extra['ee_rms_cm']:10.2f} {extra['grip_acc']:9.3f}", flush=True)
    json.dump(res, open(f"{args.out}/a3_probe_{tag}.json", "w"), indent=1)
    print(f"\nsaved {args.out}/a3_probe_{tag}.json")


if __name__ == "__main__":
    main()
