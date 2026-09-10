#!/usr/bin/env bash
# Evaluate one run/ckpt on all 10 libero_10 tasks in parallel: 5 tasks per GPU, each task a tmux window.
# usage: scripts/eval_all.sh <run_dir> <ckpt_name> <eval_seed> <episodes> <tmux_prefix>
set -euo pipefail
RUN=$1; CKPT=$2; SEED=$3; EPS=${4:-50}; PFX=${5:-ev}
R=/NHNHOME/nota/skyeom/projects/vggt_omega_policy
OUT=$RUN/eval_${CKPT%.pt}_seed${SEED}
mkdir -p "$OUT"
for t in 0 1 2 3 4 5 6 7 8 9; do
  g=$(( t % 2 ))
  tmux -L skyeom new -d -s "${PFX}_t${t}" \
    "cd $R && scripts/run_eval.sh $g python scripts/eval_libero.py --run $RUN --ckpt $CKPT --suite libero_10 --tasks $t --episodes $EPS --seed $SEED --out $OUT > $OUT/task${t}.log 2>&1"
done
echo "launched 10 eval windows (${PFX}_t0..9) -> $OUT"
