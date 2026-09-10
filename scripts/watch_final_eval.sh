#!/usr/bin/env bash
# Server-side watcher (run inside tmux): when both L0 runs reach 120K, run the pre-registered evaluation
# (libero_10, 10 tasks x 50 episodes, eval seed 7) for seed 7 then seed 2, and write a summary.
set -uo pipefail
cd /NHNHOME/nota/skyeom/projects/vggt_omega_policy
V=/NHNHOME/WORKSPACE/SDV_LGE_A/nota/marcel/envs/vga_eval/venv/bin/python
LOG=runs/final_eval_watch.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
until [ -f runs/l0_s7/step_0120000.pt ] && [ -f runs/l0_s2/step_0120000.pt ]; do sleep 300; done
sleep 60
log "120K checkpoints present"
for r in l0_s7 l0_s2; do
  log "launch eval $r"
  scripts/eval_all.sh runs/$r step_0120000.pt 7 50 "ev_${r}" | tee -a "$LOG"
  until [ "$(ls runs/$r/eval_step_0120000_seed7/libero_10_task*_seed7.json 2>/dev/null | wc -l)" = "10" ]; do sleep 120; done
  log "eval $r done"
done
$V - <<'PY' | tee -a "$LOG"
import json, glob
for r in ("l0_s7", "l0_s2"):
    fs = sorted(glob.glob(f"runs/{r}/eval_step_0120000_seed7/libero_10_task*_seed7.json"))
    srs = [json.load(open(f))["success_rate"] for f in fs]
    print(r, "per-task", [round(s, 2) for s in srs], "mean %.3f" % (sum(srs) / len(srs)))
PY
log "FINAL EVAL COMPLETE"
