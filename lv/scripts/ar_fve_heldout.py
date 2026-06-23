#!/usr/bin/env python3
"""Held-out FVE for the released merged AR critic — the repo-faithful validation
that the AR loads + reconstructs correctly, with the shuffled-pairing control.

This replaces the off-manifold refusal "smoke test": it scores the critic on
(api_explanation, activation_vector) pairs from a DOC-DISJOINT held-out split
(the AV-SFT split is partitioned doc-level away from AR-SFT training), exactly the
way scripts/eval_ar_heldout.py validates a critic — but for the released MERGED
checkpoint, using the repo's own NLACriticModel + nla.schema FVE math.

Loading note (gotcha): NLACriticModel.from_pretrained loads value_head.safetensors
only from a LOCAL dir, so we snapshot_download the repo first — otherwise the value
head is silently left at random init. The released -ar sidecar is a dataset
descriptor (no role / mse_scale), so we supply mse_scale = sqrt(d_model) (the
documented default; nla.schema.resolve_target_scale would do the same for an absent
key).

Built-in correctness checks (no separate oracle needed):
  - heldout FVE should land near docs/qwen3_8b_run.md (~44.5% meannorm / ~32% paper-def).
  - shuffled-pairing FVE must be strongly NEGATIVE (critic reads the explanation);
    ~0 would mean it ignores the text and emits the dataset mean.
  - a single-item vs batched reconstruction assert guards the padding/last-token logic.

NEEDS-GPU. Run from the repo root (so `nla` imports):
  python lv/scripts/ar_fve_heldout.py --ar syvb/nanonla-qwen3-8b-L24-ar \
      --dataset syvb/nanonla-qwen3-8b-L24-data-full --split av_sft_full.parquet -n 1000
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root → import nla

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", default="syvb/nanonla-qwen3-8b-L24-ar")
    ap.add_argument("--dataset", default="syvb/nanonla-qwen3-8b-L24-data-full")
    ap.add_argument("--split", default="av_sft_full.parquet",
                    help="doc-disjoint held-out split (AV split is disjoint from AR training)")
    ap.add_argument("--expl-col", default="api_explanation")
    ap.add_argument("--critic-template",
                    default="Summary of the following text: <text>{explanation}</text> <summary>")
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoTokenizer
    from nla.models import NLACriticModel
    from nla.schema import ACTIVATION_COLUMN, compute_predict_mean_baselines, normalize_activation

    # --- critic: snapshot first so value_head.safetensors loads (HF-id gotcha) ---
    local = snapshot_download(args.ar)
    critic = NLACriticModel.from_pretrained(
        local, torch_dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # last real token = attention_mask.sum(1) - 1
    d_model = critic.config.hidden_size
    mse_scale = math.sqrt(d_model)  # released -ar sidecar omits mse_scale; sqrt(d) is the default
    print(f"[critic] layers={critic.config.num_hidden_layers} d_model={d_model} "
          f"mse_scale={mse_scale:.2f}")

    # --- held-out (explanation, activation) pairs -------------------------------
    pq_path = hf_hub_download(args.dataset, args.split, repo_type="dataset")
    expls: list[str] = []
    chunks: list[np.ndarray] = []
    for batch in pq.ParquetFile(pq_path).iter_batches(
        batch_size=4096, columns=[args.expl_col, ACTIVATION_COLUMN]
    ):
        e = batch.column(args.expl_col).to_pylist()
        a = batch.column(ACTIVATION_COLUMN).flatten().to_numpy(zero_copy_only=False)
        chunks.append(a.astype(np.float32).reshape(len(e), -1))
        expls.extend(e)
        if sum(c.shape[0] for c in chunks) >= args.n:
            break
    gold = torch.tensor(np.concatenate(chunks, 0)[: args.n], dtype=torch.float32)
    expls = expls[: args.n]
    print(f"[data] {len(expls)} held-out pairs from {args.split}")

    bl_meannorm, bl_rawvar = compute_predict_mean_baselines(gold, mse_scale)
    print(f"[baseline] rawvar(paper-def)={bl_rawvar:.4f}  meannorm={bl_meannorm:.4f}")

    @torch.inference_mode()
    def reconstruct(texts: list[str]) -> torch.Tensor:
        prompts = [args.critic_template.format(explanation=t) for t in texts]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len, add_special_tokens=True).to(device)
        out = critic(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        last = enc["attention_mask"].sum(1) - 1  # last real token (right padding)
        return out.values[torch.arange(len(texts), device=device), last].float().cpu()

    # padding/last-token guard: batched row-0 must equal single-item row-0
    one = reconstruct(expls[:1])
    two = reconstruct(expls[:2])
    assert torch.allclose(one[0], two[0], atol=1e-2), "batched last-token != single-item (padding bug)"

    def mean_mse(texts: list[str], golds: torch.Tensor) -> float:
        preds = []
        for i in range(0, len(texts), args.micro_batch):
            preds.append(reconstruct(texts[i:i + args.micro_batch]))
        pn = normalize_activation(torch.cat(preds, 0), mse_scale)
        gn = normalize_activation(golds, mse_scale)
        return ((pn - gn) ** 2).mean().item()

    mse = mean_mse(expls, gold)
    fve_paper = 1.0 - mse / bl_rawvar
    fve_mn = 1.0 - mse / bl_meannorm
    print(f"[heldout] mse={mse:.4f}  FVE(paper-def)={fve_paper*100:.1f}%  "
          f"FVE(meannorm)={fve_mn*100:.1f}%")

    perm = np.random.default_rng(args.seed).permutation(len(expls))
    mse_s = mean_mse(expls, gold[perm])
    fve_s = 1.0 - mse_s / bl_rawvar
    print(f"[control] shuffled-pair mse={mse_s:.4f}  FVE={fve_s*100:.1f}%  "
          f"(strongly negative => critic reads the explanation; ~0 => ignores it)")
    print(f"RESULT heldout_fve_pct={fve_paper*100:.2f} heldout_fve_meannorm_pct={fve_mn*100:.2f} "
          f"shuffled_fve_pct={fve_s*100:.2f} n={len(expls)} baseline_rawvar={bl_rawvar:.4f}")


if __name__ == "__main__":
    main()
