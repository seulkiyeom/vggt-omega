"""Train one arm of the Ω action-register ladder on a LIBERO suite.

  # L0 baseline (Ω trunk, simple head)
  python scripts/train.py --suite long --steps 120000 --out runs/l0_s7 --seed 7
  # A: corrected language cache
  python scripts/train.py --lang-npy /NHNHOME/nota/skyeom/lang_cache_fixed/libero_instructions_bidir_norm.npy ...
  # B: matched control on the VGGT-1B trunk (224 input, patch 14)
  python scripts/train.py --trunk v1 --image-size 224 ...
  # C: hierarchical register head
  python scripts/train.py --head register ...
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from ovla.data.libero_hdf5 import H5_DEFAULT, LangTable, LiberoChunkDataset, Normalizer, load_or_compute_stats  # noqa: E402
from ovla.model.policy import OmegaPolicy, PolicyConfig  # noqa: E402
from ovla.train.loop import TrainConfig, train  # noqa: E402


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
    ap.add_argument("--trunk", default="omega", choices=["omega", "v1"])
    ap.add_argument("--head", default="simple", choices=["simple", "register"])
    ap.add_argument("--image-size", type=int, default=None, help="default: 256 for omega, 224 for v1")
    ap.add_argument("--head-dim", type=int, default=1024)
    ap.add_argument("--head-blocks", type=int, default=4)
    ap.add_argument("--head-hidden", type=int, default=1024)
    ap.add_argument("--head-pooled-patch", action="store_true")
    ap.add_argument("--head-depth-mode", default="multi", choices=["multi", "last"])
    ap.add_argument("--no-policy-in-register-attn", action="store_true", help="L1 ablation (omega trunk only)")
    ap.add_argument("--lang-npy", default=None, help="override the instruction embedding cache")
    ap.add_argument("--view-order", default="1,0", help="hdf5 view indices; first = reference frame (1 = wrist)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    image_size = args.image_size or (224 if args.trunk == "v1" else 256)
    stats = load_or_compute_stats(H5_DEFAULT, args.suite,
                                 cache_dir=os.path.join(os.path.dirname(args.out.rstrip("/")), "stats"))
    norm = Normalizer(stats)
    view_order = tuple(int(v) for v in args.view_order.split(","))
    lang = LangTable(npy=args.lang_npy)
    ds = LiberoChunkDataset(H5_DEFAULT, args.suite, norm, view_order=view_order, lang=lang)
    print(f"dataset: {args.suite}, {len(ds.episodes)} episodes, {len(ds)} frames, views {view_order}", flush=True)
    print(f"language cache: {lang.path}", flush=True)

    pcfg = PolicyConfig(trunk=args.trunk, head=args.head, image_size=image_size,
                        policy_in_register_attn=not args.no_policy_in_register_attn,
                        head_dim=args.head_dim, head_blocks=args.head_blocks, head_hidden=args.head_hidden,
                        head_use_pooled_patch=args.head_pooled_patch, head_depth_mode=args.head_depth_mode)
    model = OmegaPolicy(pcfg)
    print(f"arm: trunk={args.trunk} head={args.head} image={image_size} | params {model.param_report()}", flush=True)
    tcfg = TrainConfig(total_steps=args.steps, global_batch=args.global_batch, micro_batch=args.micro_batch,
                       lr=args.lr, warmup_steps=args.warmup, log_every=args.log_every,
                       ckpt_every=args.ckpt_every, num_workers=args.workers, seed=args.seed)
    meta = {"policy": {**vars(pcfg), "lora_targets": list(pcfg.lora_targets)}, "suite": args.suite,
            "view_order": list(view_order), "stats": stats.to_dict(), "lang_npy": lang.path,
            "params": model.param_report()}
    train(model, ds, tcfg, args.out, extra_meta=meta)


if __name__ == "__main__":
    main()
