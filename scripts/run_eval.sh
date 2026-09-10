#!/usr/bin/env bash
# EGL + LIBERO environment for rollouts on ssh 2 (recipe from memory ssh2-vga-isolation; libs copied to repo/egl_min).
# usage: scripts/run_eval.sh <gpu_id> <command...>
set -euo pipefail
G=$1; shift
R=/NHNHOME/nota/skyeom/projects/vggt_omega_policy
VP=/NHNHOME/WORKSPACE/SDV_LGE_A/nota/marcel/envs/vga_eval/venv/bin/python
exec env \
  LD_LIBRARY_PATH=$R/egl_min \
  __EGL_VENDOR_LIBRARY_FILENAMES=/NHNHOME/WORKSPACE/SDV_LGE_A/nota/skyeom/cache/mmada-vla-nota/nvidia-egl-580.95.05/glvnd/10_nvidia_local.json \
  PYOPENGL_PLATFORM=egl MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G CUDA_VISIBLE_DEVICES=$G \
  PYTHONPATH=$R:/NHNHOME/WORKSPACE/SDV_LGE_A/nota/marcel/envs/vga_eval/LIBERO \
  HF_HUB_OFFLINE=1 \
  "${@/#python/$VP}"
