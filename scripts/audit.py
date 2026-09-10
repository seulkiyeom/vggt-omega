"""Comprehensive pre-flight audit of every arm: shapes, token layout, weight loading, gradient reach,
batch independence, determinism, data/eval contracts. Prints one PASS/FAIL line per check per arm.

Run before launching any training:  python scripts/audit.py [--arms l0 langfix v1 reghead]
Exit code is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from ovla.data.libero_hdf5 import (H5_DEFAULT, LangTable, LiberoChunkDataset, Normalizer,  # noqa: E402
                                   compute_stats, load_or_compute_stats)
from ovla.eval.policy import normalize_instruction  # noqa: E402
from ovla.model.lora import LoRALinear  # noqa: E402
from ovla.model.policy import OmegaPolicy, PolicyConfig  # noqa: E402

FAIL: list[str] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        FAIL.append(name)


ARMS = {
    "l0":      dict(trunk="omega", head="simple"),
    "langfix": dict(trunk="omega", head="simple"),          # same graph; differs only in the language cache
    "v1":      dict(trunk="v1", head="simple", image_size=224),
    "reghead": dict(trunk="omega", head="register"),
    "noRattn": dict(trunk="omega", head="simple", policy_in_register_attn=False),
    "reglast": dict(trunk="omega", head="register", head_depth_mode="last"),
}
EXPECT = {  # (patch_token_start, per-frame N, patch count)
    "omega": (27, 283, 256),
    "v1": (15, 271, 256),
}


def audit_data(stats_dir: str) -> None:
    print("\n=== DATA / CONTRACTS ===", flush=True)
    stats = load_or_compute_stats(H5_DEFAULT, "long", cache_dir=stats_dir)
    norm = Normalizer(stats)
    rng = np.random.RandomState(0)
    a = rng.uniform(stats.action_lo, stats.action_hi, size=(64, 7)).astype(np.float32)
    a[:, 6] = np.sign(rng.randn(64))
    rt = norm.unaction(norm.action(a))
    chk("normalizer action round-trip", float(np.abs(rt - a).max()) < 1e-3, f"max|Δ| {np.abs(rt - a).max():.2e}")
    y = norm.action(a)
    chk("normalized action in [-1, 1]", float(np.abs(y).max()) <= 1.0 + 1e-6, f"max|y| {np.abs(y).max():.4f}")
    chk("gripper stays ±1 after normalisation", set(np.unique(y[:, 6]).tolist()) <= {-1.0, 1.0})

    lang = LangTable()
    import h5py
    with h5py.File(H5_DEFAULT, "r") as f:
        missing, n_eps = [], 0
        for suite in ("long", "spatial", "object", "goal"):
            for e in [k for k in f[suite] if k.startswith("episode_")]:
                n_eps += 1
                s = str(f[suite][e].attrs["language_instruction"])
                if s not in lang.idx:
                    missing.append(s)
    chk("every instruction is in the language cache", not missing, f"{n_eps} episodes, {len(set(missing))} missing")
    by_norm = {normalize_instruction(k) for k in lang.idx}
    chk("eval instruction normalisation is injective", len(by_norm) == len(lang.idx), f"{len(by_norm)}/{len(lang.idx)}")

    ds = LiberoChunkDataset(H5_DEFAULT, "long", norm, view_order=(1, 0), lang=lang)
    it = ds[0]
    chk("dataset item shapes", tuple(it["images"].shape) == (2, 3, 256, 256) and tuple(it["action"].shape) == (8, 7)
        and it["lang"].shape == (1536,) and it["proprio"].shape == (8,),
        f"images {tuple(it['images'].shape)} action {tuple(it['action'].shape)}")
    chk("images are uint8 (÷255 happens on GPU)", it["images"].dtype == torch.uint8)
    last = ds[sum(ds.lengths[:1]) - 1]  # final timestep of episode 0
    rep = last["action"]
    chk("chunk tail padding repeats the last action", bool((rep[-1] == rep[-2]).all()),
        "verified on the last timestep of episode 0")
    covered = sum(ds.lengths) == len(ds)
    chk("index covers every timestep", covered, f"{len(ds)} items, {sum(ds.lengths)} frames")


def audit_arm(name: str, overrides: dict, stats_dir: str, dev: str = "cuda") -> None:
    print(f"\n=== ARM {name}: {overrides} ===", flush=True)
    cfg = PolicyConfig(**overrides)
    t0 = time.time()
    model = OmegaPolicy(cfg).to(dev).eval()
    rep = model.param_report()
    print(f"  built in {time.time() - t0:.0f}s | {rep}", flush=True)

    # --- weights / freezing ---
    chk("LoRA count = 2 per block x 48 blocks", rep["n_lora_modules"] == 96, str(rep["n_lora_modules"]))
    tok_lora = [n for n, m in model.trunk.named_modules() if isinstance(m, LoRALinear) and n.startswith("patch_embed.")]
    chk("tokenizer carries no LoRA (stays frozen)", not tok_lora, f"{len(tok_lora)} wrapped")
    train_names = {n for n, p in model.named_parameters() if p.requires_grad}
    unexpected = [n for n in train_names
                  if not (("lora_a" in n or "lora_b" in n) or n.startswith(("head.", "proj_lang.", "proj_prop."))
                          or n in ("trunk.action_query", "trunk.policy_slot_bias"))]
    chk("trainable set is exactly LoRA + tokens + projectors + head", not unexpected, f"extra: {unexpected[:3]}")
    chk("LoRA is identity at init (B = 0)",
        all(float(m.lora_b.weight.abs().max()) == 0.0 for m in model.modules() if isinstance(m, LoRALinear)))

    # --- token layout ---
    B, S = 2, 2
    R = cfg.image_size
    torch.manual_seed(0)
    imgs = torch.rand(B, S, 3, R, R, device=dev)
    lang = torch.randn(B, cfg.lang_dim, device=dev)
    prop = torch.randn(B, cfg.proprio_dim, device=dev)
    seen = {}
    blk = model.trunk.frame_blocks[0]
    h = blk.register_forward_pre_hook(lambda m, inp: seen.update(n_in=inp[0].shape))
    with torch.no_grad():
        out = model(imgs, lang, prop)
    h.remove()
    exp_start, exp_n, exp_p = EXPECT[cfg.trunk]
    got_n = seen["n_in"][1]
    chk("per-frame token count", got_n == exp_n, f"{got_n} (expected {exp_n})")
    chk("patch_token_start", out["patch_token_start"] == exp_start, f"{out['patch_token_start']} (expected {exp_start})")
    cached = [c for c in out["cached"] if c is not None]
    chk("cached depths = 4, width = 2 x embed", len(cached) == 4 and cached[-1].shape[-1] == 2 * model.trunk.camera_token.shape[-1],
        f"{len(cached)} x {tuple(cached[-1].shape)}")
    rows = model.trunk.rows(cached[-1])
    n_rows = {k: v.shape[2] for k, v in rows.items()}
    chk("row slices sum to the sequence", sum(n_rows.values()) == got_n, str(n_rows))
    chk("action rows = n_act", n_rows["act"] == cfg.n_act, str(n_rows["act"]))
    chk("patch rows = expected patch count", n_rows["patch"] == exp_p, str(n_rows["patch"]))

    # --- RoPE only on patches (special prefix must be position-free) ---
    if cfg.trunk == "omega":
        sin, _ = model.trunk.rope_embed(H=R // model.trunk.patch_size, W=R // model.trunk.patch_size)
        chk("RoPE length matches the patch block (prefix excluded)", sin.shape[-2] == exp_p,
            f"rope {sin.shape[-2]} vs patches {exp_p}")
    else:
        pos = model.trunk.position_getter(B * S, R // model.trunk.patch_size, R // model.trunk.patch_size, device=dev)
        chk("v1 position grid matches the patch block", pos.shape[1] == exp_p, f"{pos.shape[1]} vs {exp_p}")

    # --- output ---
    act = out["action"]
    chk("action shape", tuple(act.shape) == (B, cfg.n_act, cfg.action_dim), str(tuple(act.shape)))
    chk("action is finite fp32", act.dtype == torch.float32 and bool(torch.isfinite(act).all()), str(act.dtype))

    # --- determinism (eval mode, same input twice) ---
    with torch.no_grad():
        a2 = model(imgs, lang, prop)["action"]
    chk("eval determinism", float((act - a2).abs().max()) == 0.0, f"max|Δ| {float((act - a2).abs().max()):.2e}")

    # --- batch independence: sample 0 must not see sample 1 ---
    with torch.no_grad():
        solo = model(imgs[:1], lang[:1], prop[:1])["action"]
    d = float((solo - act[:1]).abs().max())
    chk("batch independence (no cross-sample leakage)", d < 2e-2, f"max|Δ| {d:.2e} (bf16 noise floor)")

    # --- view sensitivity: swapping the two frames must change the action ---
    with torch.no_grad():
        sw = model(imgs.flip(1), lang, prop)["action"]
    chk("frame order matters (reference slot is real)", float((sw - act).abs().max()) > 1e-3,
        f"max|Δ| {float((sw - act).abs().max()):.3f}")

    # --- language/proprio actually reach the output ---
    with torch.no_grad():
        a_lang = model(imgs, torch.randn_like(lang), prop)["action"]
        a_prop = model(imgs, lang, torch.randn_like(prop))["action"]
    chk("language input changes the action", float((a_lang - act).abs().max()) > 1e-3, f"{float((a_lang - act).abs().max()):.3f}")
    chk("proprio input changes the action", float((a_prop - act).abs().max()) > 1e-3, f"{float((a_prop - act).abs().max()):.3f}")

    # --- backward ---
    model.train()
    out = model(imgs, lang, prop)
    out["action"].abs().mean().backward()
    got = {n for n, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
    frozen_grad = [n for n, p in model.named_parameters() if not p.requires_grad and p.grad is not None]
    chk("gradients reach the action tokens", "trunk.action_query" in got)
    chk("gradients reach LoRA", any("lora_b" in n for n in got))
    chk("gradients reach the head and projectors",
        any(n.startswith("head.") for n in got) and any(n.startswith("proj_lang.") for n in got))
    chk("no gradient on frozen parameters", not frozen_grad, f"{len(frozen_grad)} leaked")
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"  peak memory {mem:.1f} GiB at B={B}", flush=True)
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["l0", "v1", "reghead", "noRattn"])
    ap.add_argument("--stats-dir", default="/NHNHOME/nota/skyeom/projects/vggt_omega_policy/runs/stats")
    ap.add_argument("--skip-data", action="store_true")
    args = ap.parse_args()
    if not args.skip_data:
        audit_data(args.stats_dir)
    for a in args.arms:
        audit_arm(a, ARMS[a], args.stats_dir)
    print("\n" + ("=" * 60))
    if FAIL:
        print(f"AUDIT FAILED: {len(FAIL)} checks — {FAIL}")
        sys.exit(1)
    print("AUDIT PASSED: every check green")


if __name__ == "__main__":
    main()
