"""B-2: GLA-CLIP-style proxy-anchor / attention-boost variants on the a29 PVM readout (xattn2),
training-free, offline on LIBERO-Long demonstrations. a29 is used ONLY as a testbed for the
readout mechanism (user decision 2026-09-09); no architecture conclusion is drawn from it.

Mechanism under test (A4 finding): xattn2 (8 action queries -> [256 agentview | 256 wrist | 1 lang]
rows) reads the language row + sink patches at layer 0 and is near-uniform at layer 23; mass on
patches near the gripper/objects is ~0.02 (uniform ~0.03). GLA-CLIP's remedies are (i) a proxy anchor
built from tokens most similar to the query across all windows, (ii) dynamic re-normalisation of
attention toward that set. Transposed here:
  base            unmodified xattn2 (re-implemented manually; gate: equals the module output)
  anchor_lang     append 2 anchor rows (mean of top-k patch rows per view most cosine-similar to the
                  language row, in the module's LN(kv) space) to the kv set
  anchor_q        append 2 anchor rows built from the patches the action queries themselves attend
                  most (top-k of the base attention, per view)
  boost_lang_bB   add +B to the logits of the top-k language-similar patches per view (no new rows)
  oracle_near_bB  add +B to the logits of patches within R px of the projected end-effector in
                  BOTH views (uses GT state -> not deployable; upper bound of "read near the gripper");
                  B is swept up to 32 because layer-0 logits are very peaked (smoke: +4 barely moved mass)
  oracle_mask     hard-restrict patch keys to the near-EE set (lang row kept) -- the strongest oracle
Metrics per frame: action L1 in normalized action space vs GT chunk (overall, per horizon step,
gripper-closed frames), attention mass at layers 0 and 23 on patches near the EE (agentview and
wrist), language mass, anchor mass.

PRE-REGISTERED (2026-09-09, before any number):
  CONTINUE to rollouts for a deployable variant only if  L1(variant) <= L1(base) (no worsening; paired
  bootstrap CI upper bound of the difference <= 0.5 % of L1(base))  AND  near-EE mass at layer 0 >= 5x base.
  If even oracle_near does not reduce L1 (CI excludes an improvement), the "localise the readout"
  lever is marked DEAD for a29 (A5 intervention validity fails) regardless of the mass metric.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/NHNHOME/nota/skyeom/projects/vga_nota/src")
from vga.data.rlds import views_to_tensor  # noqa: E402
from vga.eval.run import build_policy  # noqa: E402

H5 = "/NHNHOME/nota/marcel/datasets/LIBERO_for_VGA_540625a9/libero_3d_no_noops_aligned.hdf5"
CKPTS = {
    "s7": "/NHNHOME/nota/skyeom/runs_a29/vga_resgate_ln-long/checkpoints/step_0120000.pt",
    "s2": "/NHNHOME/nota/skyeom/runs_a29/vga_resgate_seed2-long/checkpoints/step_0120000.pt",
}
IMG, GRID, PATCH = 224, 16, 14
N_A, N_W, LANG = 256, 256, 512
RECORD_LAYERS = (0, 23)
# Orientation of stored extrinsics vs stored RLDS images (probes/orient_diag.py, 2026-09-09):
ORIENT = {0: "asis", 1: "rot180"}


def decode(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def project(K, E_c2w, p):
    Xc = np.linalg.inv(E_c2w) @ np.append(p, 1.0)
    x, y, z = Xc[:3]
    return K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]


def patch_mask_near(u, v, R, H_src=256):
    """(256,) bool over the 16x16 patch grid of the 224 input: patch centre within R px (224 scale)."""
    su = IMG / H_src
    u, v = u * su, v * su
    rows, cols = np.divmod(np.arange(GRID * GRID), GRID)
    cu, cv_ = cols * PATCH + PATCH / 2, rows * PATCH + PATCH / 2
    return np.hypot(cu - u, cv_ - v) <= R


class XAttnVariant(torch.nn.Module):
    """Wraps one PVMCrossAttention; manual attention with optional anchors / logit boosts."""

    def __init__(self, orig, layer, cfg, rec):
        super().__init__()
        self.orig, self.layer, self.cfg, self.rec = orig, layer, cfg, rec
        self.boost_mask = None  # (B, 512) bool over patch keys, set per frame for oracle variants

    def forward(self, q_tokens, kv_tokens):
        m, c = self.orig, self.cfg
        B, Nq, D = q_tokens.shape
        H, hd = m.num_heads, m.head_dim
        variant, k_top, beta = c["variant"], c["k"], c["beta"]
        kv = kv_tokens
        n_anchor = 0
        sel_mask = None  # (B,512) patches selected for boost
        if variant in ("anchor_lang", "boost_lang"):
            with torch.no_grad():
                nk = m.norm_kv(kv_tokens)
                lang = F.normalize(nk[:, LANG].float(), dim=-1)  # (B,D)
                pat = F.normalize(nk[:, :LANG].float(), dim=-1)  # (B,512,D)
                sim = torch.einsum("bd,bnd->bn", lang, pat)
                sel_mask = torch.zeros(B, LANG, dtype=torch.bool, device=kv.device)
                for v0 in (0, N_A):
                    idx = sim[:, v0:v0 + 256].topk(k_top, dim=-1).indices + v0
                    sel_mask.scatter_(1, idx, True)
        if variant in ("oracle_near", "oracle_mask"):
            sel_mask = self.boost_mask
        if variant == "anchor_q":
            with torch.no_grad():
                w0 = self._weights(m, q_tokens, kv_tokens)  # (B,H,Nq,513)
                score = w0.mean((1, 2))[:, :LANG]  # (B,512)
                sel_mask = torch.zeros(B, LANG, dtype=torch.bool, device=kv.device)
                for v0 in (0, N_A):
                    idx = score[:, v0:v0 + 256].topk(k_top, dim=-1).indices + v0
                    sel_mask.scatter_(1, idx, True)
        if variant in ("anchor_lang", "anchor_q"):
            anchors = []
            for v0 in (0, N_A):
                mm = sel_mask[:, v0:v0 + 256].unsqueeze(-1).to(kv.dtype)  # (B,256,1)
                anchors.append((kv_tokens[:, v0:v0 + 256] * mm).sum(1) / mm.sum(1).clamp(min=1))
            kv = torch.cat([kv_tokens, torch.stack(anchors, 1)], 1)
            n_anchor = 2
        Nkv = kv.shape[1]
        nk = m.norm_kv(kv)
        q = m.q(m.norm_q(q_tokens)).view(B, Nq, H, hd).transpose(1, 2)
        k = m.k(nk).view(B, Nkv, H, hd).transpose(1, 2)
        v = m.v(nk).view(B, Nkv, H, hd).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(hd)  # (B,H,Nq,Nkv)
        if variant in ("boost_lang", "oracle_near") and sel_mask is not None and beta != 0:
            bias = torch.zeros(B, Nkv, device=kv.device, dtype=logits.dtype)
            bias[:, :LANG] = sel_mask.to(logits.dtype) * beta
            logits = logits + bias[:, None, None, :]
        if variant == "oracle_mask" and sel_mask is not None:
            # hard restriction: patch keys outside the near-EE set are removed (lang row kept);
            # a view whose near set is empty (EE out of frame) keeps all its patches.
            keep = torch.ones(B, Nkv, dtype=torch.bool, device=kv.device)
            for v0 in (0, N_A):
                blk = sel_mask[:, v0:v0 + 256]
                has = blk.any(-1, keepdim=True)
                keep[:, v0:v0 + 256] = torch.where(has, blk, torch.ones_like(blk))
            logits = logits.masked_fill(~keep[:, None, None, :], float("-inf"))
        w = torch.softmax(logits.float(), -1)
        y = (w.to(v.dtype) @ v).transpose(1, 2).reshape(B, Nq, D)
        out = q_tokens + m.out(y)
        if self.layer in RECORD_LAYERS:
            lg = logits[0].float()
            lg = lg[torch.isfinite(lg)]
            self.rec[self.layer] = dict(w=w.detach().mean(1)[0].cpu().numpy(), n_anchor=n_anchor,
                                        logit_std=float(lg.std()), logit_range=float(lg.max() - lg.min()),
                                        sel=None if sel_mask is None else sel_mask[0].cpu().numpy())
        return out

    @staticmethod
    def _weights(m, q_tokens, kv_tokens):
        B, Nq, D = q_tokens.shape
        H, hd = m.num_heads, m.head_dim
        nk = m.norm_kv(kv_tokens)
        q = m.q(m.norm_q(q_tokens)).view(B, Nq, H, hd).transpose(1, 2)
        k = m.k(nk).view(B, kv_tokens.shape[1], H, hd).transpose(1, 2)
        return torch.softmax(((q @ k.transpose(-1, -2)) / math.sqrt(hd)).float(), -1)


def load_frames(n_eps, stride, seed=0, suite="long"):
    rng = np.random.RandomState(seed)
    out = []
    with h5py.File(H5, "r") as f:
        g = f[suite]
        K = g["intrinsics"][:]
        eps = sorted(k for k in g if k.startswith("episode_"))
        eps = [eps[i] for i in sorted(rng.choice(len(eps), n_eps, replace=False))]
        for e in eps:
            ge = g[e]
            L = ge["action"].shape[0]
            acts = ge["action"][:]
            states = ge["state"][:]
            E = ge["extrinsics"][:]
            instr = ge["language_instruction"] if "language_instruction" in ge else ge.attrs["language_instruction"]
            for t in range(4, L - 1, stride):
                chunk = acts[t:t + 8]
                if len(chunk) < 8:
                    chunk = np.concatenate([chunk, np.repeat(chunk[-1:], 8 - len(chunk), 0)], 0)
                out.append(dict(ep=e, t=t, instr=str(instr), img=[decode(ge["image"][t, v]) for v in range(2)],
                                state=states[t].astype(np.float32), chunk=chunk.astype(np.float32),
                                grip_cmd=float(acts[t, 6]), K=K, E=E[t], ee=states[t, :3].astype(np.float64)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="s7", choices=list(CKPTS))
    ap.add_argument("--n-eps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--R", type=float, default=28.0, help="near-EE radius in px at 224")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--variants", nargs="+", default=["base", "anchor_lang", "anchor_q", "boost_lang_b2", "boost_lang_b4", "boost_lang_b16",
                                                      "oracle_near_b2", "oracle_near_b4", "oracle_near_b16", "oracle_near_b32", "oracle_mask"])
    ap.add_argument("--out", default="/NHNHOME/nota/skyeom/omega_probes/b2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()
    if args.smoke:
        args.n_eps, args.stride = 2, 40
    os.makedirs(args.out, exist_ok=True)
    print(__doc__.split("PRE-REGISTERED")[1], flush=True)

    frames = load_frames(args.n_eps, args.stride)
    print(f"frames: {len(frames)} from {args.n_eps} Long episodes (stride {args.stride}); ckpt {args.ckpt}", flush=True)
    pol, cfg, meta = build_policy(CKPTS[args.ckpt], device="cuda", precision=args.precision, check_frozen=False)
    print(f"policy step {meta['step']} image_size {pol._image_size} crop {pol._center_crop}", flush=True)
    model, norm = pol._model, pol._normalizer
    head = model.action_head
    assert head.pvm_enabled and head.h_a_len == 513 and head.h_a_lang_position == LANG
    orig_x2 = list(head.xattn2)

    def parse(vname):
        if vname == "base":
            return dict(variant="base", k=args.k, beta=0.0)
        if vname in ("anchor_lang", "anchor_q"):
            return dict(variant=vname, k=args.k, beta=0.0)
        if vname == "oracle_mask":
            return dict(variant="oracle_mask", k=args.k, beta=0.0)
        base, b = vname.rsplit("_b", 1)
        return dict(variant=base, k=args.k, beta=float(b))

    # cache per-frame inputs
    dev = "cuda"
    inputs = []
    for fr in frames:
        images = views_to_tensor(fr["img"], image_size=IMG).unsqueeze(0)
        proprio = torch.from_numpy(norm.normalize_proprio(fr["state"])).unsqueeze(0)
        gt = torch.from_numpy(norm.normalize_action(fr["chunk"]))
        masks = []
        for v in range(2):
            u, vv = project(fr["K"][v], fr["E"][v], fr["ee"])
            if ORIENT[v] == "rot180":
                u, vv = 255 - u, 255 - vv
            masks.append(patch_mask_near(u, vv, args.R))
        inputs.append((images, proprio, gt, np.array(masks)))  # masks (2,256)

    results = {}
    rec = {}
    t0 = time.time()
    base_out = None
    for vname in args.variants:
        c = parse(vname)
        wrappers = [XAttnVariant(orig_x2[l], l, c, rec) for l in range(len(orig_x2))]
        head.xattn2 = torch.nn.ModuleList(wrappers)
        rows = []
        gate_max = 0.0
        for j, (fr, (images, proprio, gt, masks)) in enumerate(zip(frames, inputs)):
            pol.set_task(fr["instr"])
            lang = pol._task_emb.unsqueeze(0).to(dev)
            if c["variant"] in ("oracle_near", "oracle_mask"):
                bm = torch.from_numpy(np.concatenate([masks[0], masks[1]])).unsqueeze(0).to(dev)
                for wmod in wrappers:
                    wmod.boost_mask = bm
            rec.clear()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.precision == "bf16")):
                out = model({"images": images.to(dev), "proprio": proprio.to(dev), "lang": lang})["action"]
            pred = out.squeeze(0).float().cpu()
            if vname == "base":
                # gate: manual attention path == original module path
                head.xattn2 = torch.nn.ModuleList(orig_x2)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(args.precision == "bf16")):
                    ref = model({"images": images.to(dev), "proprio": proprio.to(dev), "lang": lang})["action"].squeeze(0).float().cpu()
                head.xattn2 = torch.nn.ModuleList(wrappers)
                gate_max = max(gate_max, float((pred - ref).abs().max()))
            err = (pred - gt).abs()  # (8,7)
            r = dict(j=j, ep=fr["ep"], t=fr["t"], variant=vname, grip_closed=int(fr["grip_cmd"] > 0),
                     l1=float(err.mean()), l1_step0=float(err[0].mean()), l1_xyz=float(err[:, :3].mean()),
                     l1_rot=float(err[:, 3:6].mean()), l1_grip=float(err[:, 6].mean()))
            for L in RECORD_LAYERS:
                w = rec[L]["w"]  # (8, Nkv)
                wp = w[:, :LANG]
                r[f"L{L}_near_agent"] = float(wp[:, :N_A][:, masks[0]].sum(-1).mean()) if masks[0].any() else np.nan
                r[f"L{L}_near_wrist"] = float(wp[:, N_A:][:, masks[1]].sum(-1).mean()) if masks[1].any() else np.nan
                r[f"L{L}_lang"] = float(w[:, LANG].mean())
                r[f"L{L}_anchor"] = float(w[:, LANG + 1:].sum(-1).mean()) if w.shape[1] > LANG + 1 else 0.0
                r[f"L{L}_agent"] = float(wp[:, :N_A].sum(-1).mean())
                r[f"L{L}_wrist"] = float(wp[:, N_A:].sum(-1).mean())
                r[f"L{L}_ent"] = float(-(w * np.log(np.clip(w, 1e-12, None))).sum(-1).mean())
                r[f"L{L}_logit_std"] = rec[L]["logit_std"]
                r[f"L{L}_logit_range"] = rec[L]["logit_range"]
            rows.append(r)
            if j % 100 == 0:
                print(f"  [{vname}] {j}/{len(frames)} {time.time() - t0:.0f}s", flush=True)
        results[vname] = rows
        if vname == "base":
            print(f"gate: manual base path vs original module, max|d action| = {gate_max:.3e} "
                  f"{'PASS' if gate_max < 2e-2 else 'FAIL'}", flush=True)
            results["_gate_base_max_abs_diff"] = gate_max
    head.xattn2 = torch.nn.ModuleList(orig_x2)

    tag = f"{args.ckpt}_{'smoke' if args.smoke else 'full'}_{args.precision}" if args.precision != "bf16" else f"{args.ckpt}_{'smoke' if args.smoke else 'full'}"
    allrows = [r for v in args.variants for r in results[v]]
    keys = sorted({k for r in allrows for k in r})
    with open(f"{args.out}/b2_rows_{tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(allrows)

    def col(v, key, closed=None):
        rs = results[v] if closed is None else [r for r in results[v] if r["grip_closed"] == closed]
        return np.array([r[key] for r in rs], dtype=float)

    base = col("base", "l1")
    summary = {"gate_base_max_abs_diff": results.get("_gate_base_max_abs_diff"), "n_frames": len(frames), "ckpt": args.ckpt}
    print("\n=== summary (mean over frames) ===")
    hdr = f"{'variant':16s} {'L1':>8s} {'dL1%':>7s} {'CI95 dL1%':>16s} {'L1 closed':>10s} {'L0 nearA':>9s} {'L0 nearW':>9s} {'L0 lang':>8s} {'L0 anchor':>9s} {'L23 nearA':>9s} {'L23 lang':>8s}"
    print(hdr)
    rng = np.random.RandomState(0)
    for v in args.variants:
        x = col(v, "l1")
        d = x - base
        bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5]) / base.mean() * 100
        s = dict(l1=float(x.mean()), dl1_pct=float(d.mean() / base.mean() * 100), dl1_ci95_pct=[float(lo), float(hi)],
                 l1_closed=float(col(v, "l1", 1).mean()) if (col("base", "grip_closed") == 1).any() else None,
                 L0_near_agent=float(np.nanmean(col(v, "L0_near_agent"))), L0_near_wrist=float(np.nanmean(col(v, "L0_near_wrist"))),
                 L0_lang=float(col(v, "L0_lang").mean()), L0_anchor=float(col(v, "L0_anchor").mean()),
                 L23_near_agent=float(np.nanmean(col(v, "L23_near_agent"))), L23_lang=float(col(v, "L23_lang").mean()),
                 L0_ent=float(col(v, "L0_ent").mean()), L23_ent=float(col(v, "L23_ent").mean()),
                 L0_logit_std=float(col(v, "L0_logit_std").mean()), L23_logit_std=float(col(v, "L23_logit_std").mean()))
        summary[v] = s
        print(f"{v:16s} {s['l1']:8.4f} {s['dl1_pct']:7.2f} [{lo:7.2f},{hi:7.2f}] {s['l1_closed'] if s['l1_closed'] is not None else float('nan'):10.4f} "
              f"{s['L0_near_agent']:9.3f} {s['L0_near_wrist']:9.3f} {s['L0_lang']:8.3f} {s['L0_anchor']:9.3f} {s['L23_near_agent']:9.3f} {s['L23_lang']:8.3f}")
    # pre-registered verdicts
    b0 = summary["base"]["L0_near_agent"]
    verd = {}
    for v in args.variants:
        if v == "base":
            continue
        s = summary[v]
        mass_ok = s["L0_near_agent"] >= 5 * b0
        l1_ok = s["dl1_ci95_pct"][1] <= 0.5
        deployable = not v.startswith("oracle")
        verd[v] = dict(mass_ok=bool(mass_ok), l1_not_worse=bool(l1_ok), continue_to_rollouts=bool(deployable and mass_ok and l1_ok),
                       l1_improves=bool(s["dl1_ci95_pct"][1] < 0))
    print(f"logit scale (base): L0 std {summary['base']['L0_logit_std']:.2f}  L23 std {summary['base']['L23_logit_std']:.2f}  (a boost beta must be compared to these)")
    oracle = [v for v in args.variants if v.startswith("oracle")]
    if oracle:
        verd["lever_localise_readout"] = "ALIVE" if any(summary[v]["dl1_ci95_pct"][1] < 0 for v in oracle) else "DEAD (oracle boost does not reduce L1)"
    summary["verdicts"] = verd
    print("\nverdicts:", json.dumps(verd, indent=1))
    json.dump(summary, open(f"{args.out}/b2_summary_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
