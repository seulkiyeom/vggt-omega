"""Regenerate the gte-Qwen2-1.5B instruction cache the way the model is meant to be used.

The cache VGA built (and that our runs inherit) deviates from the official recipe in two ways
(`dataset_generation/build_language_cache.py` says so itself: "loaded WITHOUT trust_remote_code"):
  * causal attention instead of the model's bidirectional attention,
  * no L2 normalisation (norms 156-185), and no "Instruct: ...\nQuery: ..." prefix.
Last-token pooling was already correct and is kept identical.

Writes up to three variants next to the original so any arm can point at one, and prints the
nearest-neighbour cosine structure of each (the quantity that matters for a policy: can the
embedding tell two similar instructions apart).

  bidir_norm        trust_remote_code (bidirectional) + last-token pool + L2 norm     <- primary fix
  bidir_norm_instr  same + the official query prefix
  causal_norm       original causal attention + L2 norm only (isolates normalisation)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

SRC_DIR = "/NHNHOME/nota/marcel/datasets/LIBERO_for_VGA_540625a9/lang_cache"
MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
TASK = "Given a robot manipulation instruction, represent the task to be performed"


def last_token_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if bool((mask[:, -1] == 1).all()):
        return h[:, -1]
    idx = mask.long().sum(dim=1) - 1
    return h[torch.arange(h.shape[0], device=h.device), idx]


def describe(name: str, emb: np.ndarray, names: list[str]) -> dict:
    n = np.linalg.norm(emb, axis=1)
    E = emb / np.clip(n[:, None], 1e-9, None)
    C = E @ E.T
    np.fill_diagonal(C, -1.0)
    nn_cos = C.max(1)
    iu = np.triu_indices(len(E), 1)
    top = np.argsort(-C[iu])[:3]
    print(f"[{name}] norms {n.min():.2f}–{n.max():.2f} | pairwise cos mean {C[iu].mean():.3f} | "
          f"nearest-neighbour cos mean {nn_cos.mean():.3f}, >0.9 for {(nn_cos > 0.9).sum()}/{len(E)}")
    for k in top:
        i, j = iu[0][k], iu[1][k]
        print(f"        {C[i, j]:.3f}  \"{names[i][:46]}\" | \"{names[j][:46]}\"")
    return dict(nn_cos_mean=float(nn_cos.mean()), pair_cos_mean=float(C[iu].mean()),
                n_over_09=int((nn_cos > 0.9).sum()), top_pair_cos=float(C[iu].max()))


@torch.no_grad()
def encode(names: list[str], *, mode: str, instruct: bool, normalize: bool, device: str) -> np.ndarray:
    """mode: causal (stock Qwen2, what VGA used) | remote (trust_remote_code) | bidir (stock weights, 4-D
    non-causal mask so every token sees the whole instruction — the official model's attention pattern)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=(mode == "remote"),
                                      dtype=torch.float32, attn_implementation="eager").to(device).eval()
    texts = [f"Instruct: {TASK}\nQuery: {s}" for s in names] if instruct else list(names)
    out = []
    for i in range(0, len(texts), 8):
        batch = tok(texts[i:i + 8], padding=True, truncation=True, max_length=64, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        if mode == "bidir":
            m = batch["attention_mask"]
            B, L = m.shape
            bias = torch.zeros(B, 1, L, L, device=m.device, dtype=torch.float32)
            bias.masked_fill_(~m.bool()[:, None, None, :], torch.finfo(torch.float32).min)
            h = model(input_ids=batch["input_ids"], attention_mask=bias).last_hidden_state
        else:
            h = model(**batch).last_hidden_state
        e = last_token_pool(h, batch["attention_mask"]).float()
        if normalize:
            e = torch.nn.functional.normalize(e, p=2, dim=1)
        out.append(e.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/NHNHOME/nota/skyeom/lang_cache_fixed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variants", nargs="+", default=["bidir_norm", "bidir_norm_instr", "causal_norm", "causal_norm_instr"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    names = json.load(open(os.path.join(SRC_DIR, "libero_instructions.instructions.json")))
    orig = np.load(os.path.join(SRC_DIR, "libero_instructions.npy"))
    print(f"{len(names)} instructions; original cache {orig.shape}\n")
    report = {"original": describe("original (causal, unnormalised)", orig, names)}
    spec = {"bidir_norm": ("bidir", False, True), "bidir_norm_instr": ("bidir", True, True),
            "causal_norm": ("causal", False, True), "causal_norm_instr": ("causal", True, True),
            "remote_norm": ("remote", False, True)}
    for v in args.variants:
        mode, ins, nrm = spec[v]
        try:
            emb = encode(names, mode=mode, instruct=ins, normalize=nrm, device=args.device)
        except Exception as exc:  # a variant that this transformers version cannot build is skipped, not fatal
            print(f"[{v}] SKIPPED: {type(exc).__name__}: {exc}"[:300], flush=True)
            continue
        assert emb.shape == orig.shape, (emb.shape, orig.shape)
        np.save(os.path.join(args.out_dir, f"libero_instructions_{v}.npy"), emb)
        json.dump(names, open(os.path.join(args.out_dir, "libero_instructions.instructions.json"), "w"))
        report[v] = describe(v, emb, names)
    json.dump(report, open(os.path.join(args.out_dir, "report.json"), "w"), indent=1)
    print(f"\nsaved to {args.out_dir}")


if __name__ == "__main__":
    main()
