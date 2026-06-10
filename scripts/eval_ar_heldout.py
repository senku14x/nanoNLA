"""Held-out FVE for an AR-LoRA critic checkpoint.

Scores the critic on (explanation, activation) pairs from the AV-SFT split,
which stage-1 partitions DOC-DISJOINT from the AR-SFT training data — so this
is a clean held-out number, unlike the training-batch FVE train_sft prints.

Also runs a shuffled-pairing control (explanations re-paired with random
activations). Expected: strongly NEGATIVE FVE (~ -(1 - mse/baseline) with
mse ~ 2x variance) — a critic that reads the explanation predicts a specific
wrong direction for a mismatched pair, which is worse than predicting the
mean. FVE ~ 0 on shuffled pairs would mean the critic ignores the explanation
and outputs the dataset mean.

Usage:
  python -m scripts.eval_ar_heldout \
    --ar-ckpt .../qwen3_8b_L24_ar_sft_lora_fixed/iter_0001000 \
    --heldout-parquet $DATA/av_sft_shuf.parquet \
    --base-ckpt Qwen/Qwen3-8B --quant 4bit --n-rows 2000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file
from transformers import AutoTokenizer, BitsAndBytesConfig

from nla.config import load_nla_config
from nla.schema import compute_predict_mean_baselines, resolve_target_scale
from nla.train_sft import (
    heldout_fve_mse,
    init_critic_from_base,
    load_heldout_explanation_pairs,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ar-ckpt", required=True,
                   help="AR-LoRA dir (ar_meta.json + ar_lora_value_head.safetensors)")
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    p.add_argument("--quant", choices=["none", "4bit"], default="4bit")
    p.add_argument("--heldout-parquet", required=True,
                   help="AV-split parquet (doc-disjoint from AR training data)")
    p.add_argument("--sidecar", default=None,
                   help="Sidecar source (defaults to --heldout-parquet)")
    p.add_argument("--n-rows", type=int, default=2000)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda"
    if args.sidecar is None:
        args.sidecar = args.heldout_parquet

    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    cfg = load_nla_config(args.sidecar, tokenizer)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    template = cfg.critic_prompt_template
    assert template is not None, "critic_prompt_template missing from sidecar"

    ar_meta = json.loads((Path(args.ar_ckpt) / "ar_meta.json").read_text())
    print(f"[critic] {args.ar_ckpt}: {ar_meta}")
    quant_config = None
    if args.quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    critic = init_critic_from_base(
        args.base_ckpt, ar_meta["ar_num_layers"], torch.bfloat16, quant_config,
        device_map={"": 0} if quant_config else None,
        strip_final_norm=ar_meta.get("final_norm_stripped", False),
    )
    if quant_config is None:
        critic = critic.to(device)
    inject_adapter_in_model(LoraConfig(
        r=ar_meta["lora_r"], lora_alpha=ar_meta["lora_alpha"], lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
        target_modules=ar_meta["target_modules"],
    ), critic.backbone)
    sd = load_file(str(Path(args.ar_ckpt) / "ar_lora_value_head.safetensors"))
    missing, unexpected = critic.load_state_dict(sd, strict=False)
    n_lora = sum(1 for k in sd if "lora_" in k)
    assert n_lora > 0 and not unexpected, (
        f"AR weights load mismatch: {n_lora} lora tensors, unexpected={unexpected[:3]}"
    )
    critic.eval()

    pairs = load_heldout_explanation_pairs(args.heldout_parquet, args.n_rows)
    print(f"[data] {len(pairs)} held-out (explanation, activation) pairs "
          f"from {args.heldout_parquet}")

    acts = torch.tensor(np.stack([a for _, a in pairs]), dtype=torch.float32)
    bl_meannorm, bl_rawvar = compute_predict_mean_baselines(acts, mse_scale_f)
    print(f"[baseline] paper-def (rawvar) = {bl_rawvar:.4f} | "
          f"meannorm = {bl_meannorm:.4f}  (computed on the held-out activations)")

    mse, n = heldout_fve_mse(
        critic, tokenizer, pairs, template, mse_scale_f, device,
        micro_batch=args.micro_batch, max_len=args.max_len,
    )
    fve = 1.0 - mse / bl_rawvar
    print(f"[heldout] mse = {mse:.4f} over {n} pairs → "
          f"FVE = {fve * 100:.1f}%  (meannorm-def FVE = "
          f"{(1.0 - mse / bl_meannorm) * 100:.1f}%)")

    # Shuffled-pairing control — explanations re-paired with random activations.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pairs))
    shuffled = [(pairs[i][0], pairs[perm[i]][1]) for i in range(len(pairs))]
    mse_s, n_s = heldout_fve_mse(
        critic, tokenizer, shuffled, template, mse_scale_f, device,
        micro_batch=args.micro_batch, max_len=args.max_len,
    )
    fve_s = 1.0 - mse_s / bl_rawvar
    print(f"[control] shuffled-pair mse = {mse_s:.4f} over {n_s} → "
          f"FVE = {fve_s * 100:.1f}%  (strongly negative = critic reads the explanation; "
          f"~0 = critic ignores it)")
    print(f"RESULT heldout_fve_pct={fve * 100:.2f} shuffled_fve_pct={fve_s * 100:.2f} "
          f"n={n} baseline_rawvar={bl_rawvar:.4f}")


if __name__ == "__main__":
    main()
