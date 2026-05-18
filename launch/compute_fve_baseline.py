"""Compute predict-the-mean MSE baseline on a parquet, then convert reward → FVE.

Baseline: E[mse_nrm(normalize(μ), normalize(v_i))] where μ = mean of activations.
FVE = 1 - mse_actual / baseline.
"""

import argparse
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from nla.config import load_nla_config
from nla.schema import normalize_activation, resolve_target_scale
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--n", type=int, default=2000, help="rows to use for the baseline")
    p.add_argument("--reward-pre", type=float, default=None,
                   help="if set, also print FVE for this pre-RL reward")
    p.add_argument("--reward-post", type=float, default=None,
                   help="if set, also print FVE for this post-RL reward")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cfg = load_nla_config(args.sidecar, tokenizer)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    print(f"mse_scale_f = {mse_scale_f}, d_model = {cfg.d_model}")

    pf = pq.ParquetFile(args.parquet)
    rg = pf.read_row_group(0, columns=["activation_vector"]).slice(0, args.n)
    acts = rg.column("activation_vector").to_pylist()
    acts_t = torch.tensor(acts, dtype=torch.float32)  # [N, d]
    print(f"loaded {acts_t.shape[0]} activations")

    mu = acts_t.mean(dim=0, keepdim=True)  # [1, d]
    mu_n = normalize_activation(mu, mse_scale_f)  # [1, d]

    acts_n = normalize_activation(acts_t, mse_scale_f)  # [N, d]
    mse_per_row = ((mu_n - acts_n) ** 2).mean(dim=-1)  # [N]
    baseline = mse_per_row.mean().item()
    print(f"\n=== Predict-the-mean baseline (normalized MSE) ===")
    print(f"  baseline mse_nrm = {baseline:.4f}")
    print(f"  baseline reward (=-mse_nrm) = {-baseline:.4f}")

    # Random-orientation upper bound (for comparison)
    print(f"  fully-random ceiling = 2.0 (== FAILED extraction floor)")

    if args.reward_pre is not None:
        mse_pre = -args.reward_pre
        fve_pre = 1.0 - mse_pre / baseline
        print(f"\n=== Pre-RL ===")
        print(f"  reward = {args.reward_pre:.4f}  mse_nrm = {mse_pre:.4f}")
        print(f"  FVE = 1 - {mse_pre:.4f}/{baseline:.4f} = {fve_pre*100:.2f}%")

    if args.reward_post is not None:
        mse_post = -args.reward_post
        fve_post = 1.0 - mse_post / baseline
        print(f"\n=== Post-RL ===")
        print(f"  reward = {args.reward_post:.4f}  mse_nrm = {mse_post:.4f}")
        print(f"  FVE = 1 - {mse_post:.4f}/{baseline:.4f} = {fve_post*100:.2f}%")

    if args.reward_pre is not None and args.reward_post is not None:
        print(f"\n=== Delta ===")
        print(f"  FVE Δ = {(fve_post - fve_pre)*100:+.2f} pp")


if __name__ == "__main__":
    main()
