"""Evaluate a trained run on LIBERO. Must be launched through scripts/run_eval.sh (EGL + LIBERO env vars).

  scripts/run_eval.sh <gpu> python scripts/eval_libero.py --run runs/l0_s7 --suite libero_10 --tasks 0 1 2 --episodes 50 --seed 7
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from ovla.eval.libero_rollout import eval_task  # noqa: E402
from ovla.eval.policy import OmegaLiberoPolicy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run, "eval")
    pol = OmegaLiberoPolicy(args.run, args.ckpt)
    print(f"policy loaded: step {pol.step}, view_order {pol.view_order}", flush=True)
    for t in args.tasks:
        eval_task(pol, args.suite, t, args.episodes, out, seed=args.seed)


if __name__ == "__main__":
    main()
