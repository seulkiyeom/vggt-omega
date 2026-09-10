"""Minimal training loop: AdamW over trainable params, linear warmup then constant LR, grad clip, bf16 autocast (inside
the model), L1 chunk loss, periodic logging + checkpoints of the TRAINABLE state only (LoRA, policy tokens, projectors, head)."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..model.lora import LoRALinear


@dataclass(frozen=True)
class TrainConfig:
    total_steps: int = 120_000
    global_batch: int = 32
    micro_batch: int = 16
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    log_every: int = 50
    ckpt_every: int = 10_000
    num_workers: int = 8
    seed: int = 7


def trainable_state_dict(model: nn.Module) -> dict:
    keep = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in keep}


def lora_b_norm(model: nn.Module) -> float:
    return math.sqrt(sum(float((m.lora_b.weight ** 2).sum()) for m in model.modules() if isinstance(m, LoRALinear)))


def train(model: nn.Module, dataset, cfg: TrainConfig, out_dir: str, extra_meta: dict | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    dev = "cuda"
    model = model.to(dev).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    accum = max(1, cfg.global_batch // cfg.micro_batch)
    loader = DataLoader(dataset, batch_size=cfg.micro_batch, shuffle=True, num_workers=cfg.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=cfg.num_workers > 0)
    it = iter(loader)
    q0 = model.trunk.action_query.detach().clone()
    log_path = os.path.join(out_dir, "train_log.jsonl")
    json.dump({"train": asdict(cfg), **(extra_meta or {})}, open(os.path.join(out_dir, "run_meta.json"), "w"), indent=1)
    t0, running, n_run = time.time(), 0.0, 0
    for step in range(1, cfg.total_steps + 1):
        lr = cfg.lr * min(1.0, step / cfg.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            images = batch["images"].to(dev, non_blocking=True).float().div_(255.0)
            out = model(images, batch["lang"].to(dev), batch["proprio"].to(dev))
            loss = (out["action"].float() - batch["action"].to(dev)).abs().mean() / accum
            loss.backward()
            loss_acc += float(loss)
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        running += loss_acc
        n_run += 1
        if step % cfg.log_every == 0 or step == 1:
            rec = dict(step=step, loss=running / n_run, lr=lr, grad_norm=float(gn), lora_b_norm=lora_b_norm(model),
                       action_query_shift=float((model.trunk.action_query.detach() - q0).norm()),
                       steps_per_s=n_run / max(time.time() - t0, 1e-6), mem_GiB=torch.cuda.max_memory_allocated() / 2**30)
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            t0, running, n_run = time.time(), 0.0, 0
        if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
            torch.save({"step": step, "trainable": trainable_state_dict(model), "meta": extra_meta or {}},
                       os.path.join(out_dir, f"step_{step:07d}.pt"))
    return {"final_loss": loss_acc}
