"""LIBERO rollouts for one suite / task subset (protocol parity with OpenVLA-OFT / VGA):
50 fixed init states per task, 10 settle steps with the do-nothing action [0,0,0,0,0,0,-1], chunk of 8 executed
open-loop, success = env `done`, step budget per suite (libero_10: 520). Writes one JSON per task."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

DUMMY = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def make_env(bddl_path: str, resolution: int = 256):
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(bddl_file_name=bddl_path, camera_heights=resolution, camera_widths=resolution)


def run_episode(env, policy, init_state, max_steps: int, wait_steps: int = 10) -> dict:
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(wait_steps):
        obs, _, _, _ = env.step(DUMMY)
    steps, done = 0, False
    t0 = time.time()
    while steps < max_steps and not done:
        chunk = policy(obs)
        for a in chunk:
            obs, _, done, _ = env.step(a.astype(np.float64))
            steps += 1
            if done or steps >= max_steps:
                break
    return {"success": bool(done), "steps": steps, "wall_s": time.time() - t0}


def eval_task(policy, suite: str, task_id: int, n_episodes: int, out_dir: str, seed: int = 7) -> dict:
    from libero.libero import benchmark
    np.random.seed(seed)
    torch.manual_seed(seed)
    bm = benchmark.get_benchmark_dict()[suite]()
    task = bm.get_task(task_id)
    inits = bm.get_task_init_states(task_id)
    env = make_env(bm.get_task_bddl_file_path(task_id))
    policy.set_task(task.language)
    results = []
    for ep in range(n_episodes):
        r = run_episode(env, policy, inits[ep], MAX_STEPS[suite])
        r.update(ep=ep)
        results.append(r)
        print(json.dumps({"task": task_id, **r}), flush=True)
    env.close()
    summary = {"suite": suite, "task_id": task_id, "task": task.language, "n": n_episodes, "seed": seed,
               "success_rate": float(np.mean([r["success"] for r in results])), "episodes": results,
               "ckpt_step": getattr(policy, "step", None)}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(summary, open(os.path.join(out_dir, f"{suite}_task{task_id}_seed{seed}.json"), "w"), indent=1)
    print(f"TASK {task_id} '{task.language}': SR {summary['success_rate']:.3f} ({n_episodes} eps)", flush=True)
    return summary
