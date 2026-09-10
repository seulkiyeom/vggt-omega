"""LoRA on nn.Linear (incl. subclasses such as Ω's LinearKMaskedBias): y = base(x) + (alpha/r) * B(A(x)).

The base module is called through its own forward, so masked-bias semantics are preserved.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)  # exact identity at init
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # Ω's attention reads qkv.in_features directly; mirror the wrapped Linear's public shape attributes.
    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * self.lora_b(self.lora_a(self.drop(x)))


def apply_lora(root: nn.Module, target_suffixes: Iterable[str], rank: int, alpha: float, dropout: float = 0.0,
               exclude_prefixes: Iterable[str] = ("patch_embed.",)) -> int:
    """Wrap every nn.Linear whose qualified name ends with one of target_suffixes and does not start with an
    excluded prefix (default: the DINOv3 tokenizer stays fully frozen, as VGA froze DINOv2). Returns #wrapped."""
    suffixes, excl = tuple(target_suffixes), tuple(exclude_prefixes)
    wrapped = 0
    for name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if full.startswith(excl):
                continue
            if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear) and full.endswith(suffixes):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                wrapped += 1
    return wrapped


def lora_parameters(root: nn.Module):
    for m in root.modules():
        if isinstance(m, LoRALinear):
            yield from m.lora_a.parameters()
            yield from m.lora_b.parameters()
