"""Train the Ω action-register policy on one LIBERO suite.

Stage-2 smoke:  python scripts/train.py --suite long --steps 300 --micro-batch 16 --out runs/smoke_l0
Full L0:        python scripts/train.py --suite long --steps 120000 --seed 7 --out runs/l0_s7
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ovla.data.libero_hdf5 import H5_DEFAULT, LiberoChunkDataset, Normalizer, load_or_compute_stats  # noqa: E402
from ovla.model.policy import OmegaPolicy, PolicyConfig  # noqa: E402
from ovla.train.loop import TrainConfig, train  # noqa: E402

OMEGA_CKPT = "/NHNHOME/nota/skyeom/models/vggt-omega/vggt_omega_1b_512.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="long")
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--global-batch", type=int, default=32)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=10_000)
    ap.add_argument("--no-policy-in-register-attn", action="store_true", help="L1 ablation switch")
    ap.add_argument("--view-order", default="1,0", help="hdf5 view indices; first = Ω reference slot (1 = wrist)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = load_or_compute_stats(H5_DEFAULT, args.suite, cache_dir=os.path.join(os.path.dirname(args.out.rstrip("/")), "stats"))
    norm = Normalizer(stats)
    view_order = tuple(int(v) for v in args.view_order.split(","))
    ds = LiberoChunkDataset(H5_DEFAULT, args.suite, norm, view_order=view_order)
    print(f"dataset: suite {args.suite}, {len(ds.episodes)} episodes, {len(ds)} frames, view_order {view_order}", flush=True)
    print("action lo/hi:", stats.action_lo.round(3).tolist(), stats.action_hi.round(3).tolist(), flush=True)

    pcfg = PolicyConfig(omega_ckpt=OMEGA_CKPT, policy_in_register_attn=not args.no_policy_in_register_attn)
    model = OmegaPolicy(pcfg)
    print("params:", model.param_report(), flush=True)
    tcfg = TrainConfig(total_steps=args.steps, global_batch=args.global_batch, micro_batch=args.micro_batch, lr=args.lr,
                       warmup_steps=args.warmup, log_every=args.log_every, ckpt_every=args.ckpt_every,
                       num_workers=args.workers, seed=args.seed)
    meta = {"policy": {**pcfg.__dict__, "lora_targets": list(pcfg.lora_targets)}, "suite": args.suite, "view_order": list(view_order),
            "stats": stats.to_dict()}
    train(model, ds, tcfg, args.out, extra_meta=meta)


if __name__ == "__main__":
    main()
