#!/usr/bin/env bash
# Unattended overnight batch: four new arms to a common 30K budget, then a screening evaluation of five
# arms (the four plus the existing L0 baseline) at the SAME 30K checkpoint, 10 tasks x 20 episodes each.
# Hard deadline: if training is still running at $DEADLINE, stop it and evaluate the latest common checkpoint.
set -uo pipefail
R=/NHNHOME/nota/skyeom/projects/vggt_omega_policy
V=/NHNHOME/WORKSPACE/SDV_LGE_A/nota/marcel/envs/vga_eval/venv/bin/python
cd "$R"
LOG=runs/overnight.log
STEPS=${STEPS:-30000}
EPISODES=${EPISODES:-20}
DEADLINE=${DEADLINE:-$(date -d '+5 hours' +%s)}
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

launch() {  # launch <name> <gpu> <extra args...>
  local name=$1 gpu=$2; shift 2
  tmux -L skyeom new -d -s "arm_$name" \
    "cd $R && CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL $V scripts/train.py --suite long --steps $STEPS \
     --micro-batch 16 --ckpt-every 10000 --seed 7 $* --out runs/$name 2>&1 \
     | grep --line-buffered -v Warning | tee runs/$name.log"
  log "launched arm $name on GPU$gpu ($*)"
}

log "=== overnight batch: STEPS=$STEPS EPISODES=$EPISODES deadline=$(date -d @$DEADLINE '+%F %T')"
launch v1_s7      0 --trunk v1
launch reghead_s7 0 --head register
launch reglast_s7 1 --head register --head-depth-mode last
launch noRattn_s7 1 --no-policy-in-register-attn
sleep 240
for a in v1_s7 reghead_s7 reglast_s7 noRattn_s7; do
  log "$a: $(tail -1 runs/$a.log | cut -c1-90)"
done

# ---- wait for the common checkpoint, or the deadline ----
CK=$(printf "step_%07d.pt" "$STEPS")
while :; do
  done_n=0
  for a in v1_s7 reghead_s7 reglast_s7 noRattn_s7; do [ -f "runs/$a/$CK" ] && done_n=$((done_n + 1)); done
  [ "$done_n" = 4 ] && { log "all four arms reached $STEPS"; break; }
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    log "DEADLINE reached with $done_n/4 arms at $STEPS — stopping trainers and using the latest common checkpoint"
    for a in v1_s7 reghead_s7 reglast_s7 noRattn_s7; do tmux -L skyeom kill-session -t "arm_$a" 2>/dev/null; done
    sleep 5; pkill -f "[s]cripts/train.py --suite long --steps $STEPS"; sleep 20
    break
  fi
  sleep 120
done
for a in v1_s7 reghead_s7 reglast_s7 noRattn_s7; do log "$a final: $(tail -1 runs/$a.log | cut -c1-90)"; done

# ---- pick the largest checkpoint every arm has (including the L0 baseline) ----
ARMS="l0_s7 v1_s7 reghead_s7 reglast_s7 noRattn_s7"
COMMON=""
for s in 0060000 0050000 0040000 0030000 0020000 0010000; do
  ok=1
  for a in $ARMS; do [ -f "runs/$a/step_$s.pt" ] || ok=0; done
  [ "$ok" = 1 ] && { COMMON="step_$s.pt"; break; }
done
[ -z "$COMMON" ] && { log "FATAL: no checkpoint common to all arms"; exit 1; }
log "common checkpoint = $COMMON; evaluating $EPISODES episodes x 10 tasks per arm"

# ---- screening evaluation, one arm at a time, 10 tasks in parallel (5 per GPU) ----
for a in $ARMS; do
  OUT="runs/$a/eval_${COMMON%.pt}_seed7"
  if [ -f "$OUT/libero_10_task9_seed7.json" ]; then log "$a already evaluated"; continue; fi
  log "eval $a @ $COMMON"
  mkdir -p "$OUT"
  for t in 0 1 2 3 4 5 6 7 8 9; do
    g=$((t % 2))
    tmux -L skyeom new -d -s "ev_${a}_t${t}" \
      "cd $R && scripts/run_eval.sh $g python scripts/eval_libero.py --run runs/$a --ckpt $COMMON \
       --suite libero_10 --tasks $t --episodes $EPISODES --seed 7 --out $OUT > $OUT/task${t}.log 2>&1"
  done
  while [ "$(ls $OUT/libero_10_task*_seed7.json 2>/dev/null | wc -l)" -lt 10 ]; do sleep 60; done
  log "$a done: $($V - <<PY
import json,glob
fs=sorted(glob.glob("$OUT/libero_10_task*_seed7.json")); srs=[json.load(open(f))["success_rate"] for f in fs]
print("per-task", [round(s,2) for s in srs], "mean %.3f" % (sum(srs)/len(srs)))
PY
)"
done

log "=== SUMMARY ($COMMON, $EPISODES episodes/task) ==="
$V - <<PY | tee -a "$LOG"
import json, glob, os
for a in "l0_s7 v1_s7 reghead_s7 reglast_s7 noRattn_s7".split():
    d = f"runs/{a}/eval_${COMMON%.pt}_seed7"
    fs = sorted(glob.glob(os.path.join(d, "libero_10_task*_seed7.json")))
    if not fs:
        print(f"{a:12s} no results"); continue
    srs = [json.load(open(f))["success_rate"] for f in fs]
    meta = json.load(open(f"runs/{a}/run_meta.json"))
    p = meta.get("policy", {})
    print(f"{a:12s} mean {sum(srs)/len(srs):.3f}  per-task {[round(s,2) for s in srs]}  "
          f"[trunk={p.get('trunk')} head={p.get('head')} depth={p.get('head_depth_mode')} "
          f"Rattn={p.get('policy_in_register_attn')} train_M={meta.get('params',{}).get('trainable_M')}]")
PY
log "=== OVERNIGHT BATCH COMPLETE ==="
