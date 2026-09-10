"""A-2: frozen depth quality of VGGT-1B vs VGGT-Omega on LIBERO frames (no training).

Premise gate for the "swap the geometry trunk" line. Both models are run frozen on the same
LIBERO frames (agentview + wrist, 256x256 RLDS orientation) and their depth is compared to
the simulator GT depth after median scale alignment (both models predict depth up to a global
scale). Metrics: AbsRel and delta<1.25 over (i) all valid pixels, (ii) a disc of radius R_NEAR
px around the projected end-effector ("near gripper"), in both views. Arms: model x input
resolution x {joint 2-view, single view}. Scale alignment: per view ("_pv") and one joint scale
for both views ("_js", tests cross-view scale consistency).

PRE-REGISTERED (written before any number, 2026-09-09):
  PASS  agentview near-gripper AbsRel (joint 2-view, per-view scale) of Omega@256 is >= 15 % lower
        than VGGT-1B@224, with a paired bootstrap 95 % CI on the relative difference excluding 0.
  WEAK  otherwise -> the "better geometry prior" premise for a trunk swap is marked weak.
Orientation self-check: the EE projection is validated against GT depth at the projected pixel for
both the stored orientation and its 180-degree rotation; the consistent one is used and both
residuals are reported. If neither is consistent (median |z - d| > 5 cm) near metrics are INVALID.
"""
from __future__ import annotations

import argparse
import csv
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
R_NEAR = 24  # px at 256 (~6-7 cm on the table)


def decode(img_bytes):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def project(K, E_c2w, p_world):
    Xc = np.linalg.inv(E_c2w) @ np.append(p_world, 1.0)
    x, y, z = Xc[:3]
    return K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2], z


def load_samples(suites, n_eps, per_ep, seed=0):
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
                for frac in np.linspace(0.15, 0.85, per_ep):
                    t = int(round(frac * (L - 1)))
                    out.append(dict(suite=s, ep=e, t=t, K=K, img=[decode(g["image"][t, v]) for v in range(2)],
                                    depth=g["depth"][t].astype(np.float32), E=g["extrinsics"][t],
                                    ee=g["state"][t, :3].astype(np.float64)))
    return out


def orientation_check(samples):
    """Per-view orientation of the stored extrinsics relative to the stored (RLDS) images.

    Visual check 2026-09-09 (probes/orient_diag.py overlays): agentview projects the EE correctly
    'asis'; the wrist view needs a 180-degree rotation. Validity criteria: agentview signed
    z_ee - depth@pixel median within +-5 cm; wrist projected pixel std < 15 px (eye-in-hand => fixed).
    """
    H = samples[0]["depth"].shape[-1]
    res = {}
    dz = []
    for smp in samples:
        u, v, z = project(smp["K"][0], smp["E"][0], smp["ee"])
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < H and 0 <= vi < H:
            dz.append(z - float(smp["depth"][0, vi, ui]))
    res["agent_asis_signed_dz_median"] = float(np.median(dz)) if dz else None
    uw = np.array([project(s["K"][1], s["E"][1], s["ee"])[:2] for s in samples])
    res["wrist_uv_std"] = [float(uw[:, 0].std()), float(uw[:, 1].std())]
    res["wrist_uv_median_rot180"] = [float(H - 1 - np.median(uw[:, 0])), float(H - 1 - np.median(uw[:, 1]))]
    ok_agent = res["agent_asis_signed_dz_median"] is not None and abs(res["agent_asis_signed_dz_median"]) < 0.05
    ok_wrist = max(res["wrist_uv_std"]) < 15
    orient = {"agent": "asis" if ok_agent else None, "wrist": "rot180" if ok_wrist else None}
    return orient, res


def near_mask(H, u, v, R):
    yy, xx = np.mgrid[0:H, 0:H]
    return np.hypot(xx - u, yy - v) <= R


def metrics(pred, gt, valid, scale=None):
    if valid.sum() < 20:
        return None
    p, g = pred[valid], gt[valid]
    s = float(np.median(g / np.clip(p, 1e-6, None))) if scale is None else scale
    p = p * s
    ratio = np.maximum(p / g, g / p)
    return dict(absrel=float(np.mean(np.abs(p - g) / g)), d125=float(np.mean(ratio < 1.25)), scale=s, n=int(valid.sum()))


def to_model_input(imgs, res):
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0  # (S,3,256,256)
    if res != x.shape[-1]:
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=(res < x.shape[-1]))
    return x


def run_model(name, model, x, dev):
    x = x.to(dev)
    with torch.no_grad():
        if name == "v1":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            d = out["depth"][0, ..., 0].float()
            c = out["depth_conf"][0].float()
        else:
            out = model(x)  # Omega autocasts internally
            d = out["depth"][0].float()
            c = out["depth_conf"][0].float()
            if d.ndim == 4:
                d = d[..., 0]
            if c.ndim == 4:
                c = c[..., 0]
    return d.cpu().numpy(), c.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=["long", "spatial"])
    ap.add_argument("--n-eps", type=int, default=40)
    ap.add_argument("--per-ep", type=int, default=3)
    ap.add_argument("--out", default="/NHNHOME/nota/skyeom/omega_probes/a2")
    ap.add_argument("--arms", nargs="+", default=["v1@224", "om@224", "om@256", "om@512"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_eps, args.per_ep = 2, 2
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda"
    print(__doc__.split("PRE-REGISTERED")[1].split("Orientation")[0], flush=True)
    samples = load_samples(args.suites, args.n_eps, args.per_ep)
    print(f"samples: {len(samples)} from suites {args.suites}", flush=True)
    orient, ostats = orientation_check(samples)
    print("orientation check (EE projection vs GT depth):", json.dumps(ostats), "->", orient, flush=True)

    sys.path.insert(0, "/NHNHOME/nota/skyeom/projects/vggt_ref")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models = {}
    if any(a.startswith("v1") for a in args.arms):
        from vggt.models.vggt import VGGT
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        m = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
        models["v1"] = m
        print("v1 params:", sum(p.numel() for p in m.parameters()) / 1e6, "M", flush=True)
    if any(a.startswith("om") for a in args.arms):
        from vggt_omega.models import VGGTOmega
        m = VGGTOmega()
        m.load_state_dict(torch.load(OMEGA_CKPT, map_location="cpu"), strict=True)
        models["om"] = m.to(dev).eval()
        print("omega loaded strict; params:", sum(p.numel() for p in m.parameters()) / 1e6, "M", flush=True)

    rows = []
    t0 = time.time()
    for i, smp in enumerate(samples):
        H = smp["depth"].shape[-1]
        gt = smp["depth"]
        valid = (gt > 0.05) & (gt < 10.0) & np.isfinite(gt)
        masks = {}
        for v in range(2):
            u, vv, _ = project(smp["K"][v], smp["E"][v], smp["ee"])
            o = orient["agent" if v == 0 else "wrist"]
            if o == "rot180":
                u, vv = H - 1 - u, H - 1 - vv
            masks[v] = (near_mask(H, u, vv, R_NEAR) & valid[v]) if o else np.zeros_like(valid[v])
        for arm in args.arms:
            name, res = arm.split("@")
            res = int(res)
            model = models[name]
            for mode in ("joint", "single"):
                if mode == "joint":
                    d, c = run_model(name, model, to_model_input(smp["img"], res), dev)
                else:
                    ds, cs = zip(*[run_model(name, model, to_model_input([smp["img"][v]], res), dev) for v in range(2)])
                    d, c = np.concatenate(ds, 0), np.concatenate(cs, 0)
                if d.shape[-1] != H:
                    d = F.interpolate(torch.from_numpy(d)[None], size=(H, H), mode="bilinear", align_corners=False)[0].numpy()
                    c = F.interpolate(torch.from_numpy(c)[None], size=(H, H), mode="bilinear", align_corners=False)[0].numpy()
                pj, gj = d[valid], gt[valid]
                s_joint = float(np.median(gj / np.clip(pj, 1e-6, None)))
                for v, view in enumerate(("agent", "wrist")):
                    r = dict(i=i, suite=smp["suite"], ep=smp["ep"], t=smp["t"], arm=arm, mode=mode, view=view,
                             conf=float(c[v][valid[v]].mean()))
                    for tag, m_, sc in (("all_pv", valid[v], None), ("near_pv", masks[v], None),
                                        ("all_js", valid[v], s_joint), ("near_js", masks[v], s_joint)):
                        mm = metrics(d[v], gt[v], m_, sc)
                        if mm:
                            r.update({f"{tag}_absrel": mm["absrel"], f"{tag}_d125": mm["d125"],
                                      f"{tag}_scale": mm["scale"], f"{tag}_n": mm["n"]})
                    rows.append(r)
        if i % 20 == 0:
            print(f"  {i}/{len(samples)}  {time.time() - t0:.0f}s", flush=True)

    tag = "smoke" if args.smoke else "full"
    keys = sorted({k for r in rows for k in r})
    with open(f"{args.out}/a2_rows_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    mcols = [c for c in keys if c.endswith("_absrel") or c.endswith("_d125") or c == "conf"]
    groups = {}
    for r in rows:
        groups.setdefault((r["arm"], r["mode"], r["view"]), []).append(r)
    lines = ["arm,mode,view," + ",".join(mcols)]
    print("\n=== mean over samples ===")
    print(f"{'arm':8s} {'mode':6s} {'view':5s} " + " ".join(f"{c:>15s}" for c in mcols))
    for (arm, mode, view), rs in sorted(groups.items()):
        vals = [float(np.nanmean([r.get(c, np.nan) for r in rs])) for c in mcols]
        print(f"{arm:8s} {mode:6s} {view:5s} " + " ".join(f"{v:15.4f}" for v in vals))
        lines.append(f"{arm},{mode},{view}," + ",".join(f"{v:.5f}" for v in vals))
    open(f"{args.out}/a2_summary_{tag}.csv", "w").write("\n".join(lines) + "\n")

    def sel(arm, mode, view):
        return {r["i"]: r for r in rows if r["arm"] == arm and r["mode"] == mode and r["view"] == view}

    verdict = {"orientation": orient, "orientation_stats": ostats, "n_samples": len(samples)}
    a, b = sel("v1@224", "joint", "agent"), sel("om@256", "joint", "agent")
    if a and b and orient["agent"]:
        idx = sorted(set(a) & set(b))
        x = np.array([a[i].get("near_pv_absrel", np.nan) for i in idx])
        y = np.array([b[i].get("near_pv_absrel", np.nan) for i in idx])
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        rel = 1 - y.mean() / x.mean()
        rng = np.random.RandomState(0)
        bs = []
        for _ in range(2000):
            j = rng.randint(0, len(x), len(x))
            bs.append(1 - y[j].mean() / x[j].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        verdict.update(dict(n=int(len(x)), v1_near_absrel=float(x.mean()), om256_near_absrel=float(y.mean()),
                            rel_improvement=float(rel), ci95=[float(lo), float(hi)], PASS=bool(rel >= 0.15 and lo > 0)))
        print(f"\nPRE-REG agent near-gripper AbsRel (joint, per-view scale): v1@224 {x.mean():.4f} vs om@256 {y.mean():.4f}"
              f" -> rel. improvement {rel * 100:.1f}% CI95 [{lo * 100:.1f}, {hi * 100:.1f}] -> "
              f"{'PASS' if verdict['PASS'] else 'WEAK'}", flush=True)
    else:
        verdict["PASS"] = None
        print("PRE-REG comparison not computable (orientation invalid or arms missing)")
    json.dump(verdict, open(f"{args.out}/a2_verdict_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
