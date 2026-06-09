"""Standalone CLI to run NLA evals against a saved actor+critic.

Usage:
    python -m evals.run_evals \
        --av-ckpt /workspace/.../av_sft/iter_0001000/hf \
        --rl-lora /workspace/.../rl/iter_000500 \
        --parquet /workspace/.../rl_shuf.parquet \
        --sidecar /workspace/.../rl_shuf.parquet \
        --output-dir /workspace/.../eval_runs/smoke \
        --evals hallucination \
        --n-samples 40 \
        --step 500

Loads the actor (with optional LoRA), wires it up like the RL trainer would,
and runs each registered eval once. Writes per-eval JSON to output-dir/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config

import evals.hallucination  # noqa: F401 — registers the eval
import evals.karvonen_confusion  # noqa: F401 — registers the eval

from .base import EvalConfig
from .registry import REGISTRY, get_eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", required=True,
                   help="HF dir for AV actor base model")
    p.add_argument("--rl-lora", default=None,
                   help="Optional: PEFT LoRA adapter dir to attach on top")
    p.add_argument("--parquet", required=True,
                   help="Held-out parquet (typically rl_shuf.parquet)")
    p.add_argument("--sidecar", default=None,
                   help="Sidecar source; defaults to --parquet")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--evals", default="hallucination",
                   help="Comma-separated eval ids")
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--eval-skip-rows", type=int, default=35000,
                   help="Skip the first N rows of --parquet (training cursor)")
    p.add_argument("--step", type=int, default=0,
                   help="Step number to log (just affects output filename)")
    p.add_argument("--judge-model", default="claude-sonnet-4-6")
    p.add_argument("--anthropic-api-key-env", default="ANTHROPIC_API_KEY")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attn-implementation", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    args = p.parse_args()

    if args.sidecar is None:
        args.sidecar = args.parquet
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # ---- tokenizer + NLA sidecar ----
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    nla_cfg = load_nla_config(args.sidecar, tokenizer)
    print(f"[cfg] inj_id={nla_cfg.injection_token_id} d_model={nla_cfg.d_model}")

    # ---- actor (+ optional LoRA) ----
    print(f"[actor] loading {args.av_ckpt}")
    actor = AutoModelForCausalLM.from_pretrained(
        args.av_ckpt, torch_dtype=dtype, attn_implementation=args.attn_implementation,
    ).to(device).eval()
    if args.rl_lora:
        from peft import PeftModel
        print(f"[actor] attaching LoRA from {args.rl_lora}")
        actor = PeftModel.from_pretrained(actor, args.rl_lora, is_trainable=False)
        actor.eval()

    # ---- run each eval ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_ids = [s.strip() for s in args.evals.split(",") if s.strip()]
    all_results = {}
    for eid in eval_ids:
        cls = get_eval(eid)
        cfg = EvalConfig(
            output_dir=output_dir,
            n_samples=args.n_samples,
            seed=args.seed,
            eval_skip_rows=args.eval_skip_rows,
            parquet_path=args.parquet,
            judge_model=args.judge_model,
            anthropic_api_key_env=args.anthropic_api_key_env,
        )
        ev = cls(cfg)
        print(f"\n=== eval: {eid} ({ev.name}) ===")
        ev.setup(actor, critic=None, tokenizer=tokenizer, nla_cfg=nla_cfg, device=device)
        result = ev.evaluate(args.step)
        ev.teardown()
        all_results[eid] = {"metrics": result.metrics}
        print(f"=== {eid} done ===")
        print(json.dumps(result.metrics, indent=2))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "step": args.step,
        "av_ckpt": args.av_ckpt,
        "rl_lora": args.rl_lora,
        "evals": all_results,
    }, indent=2))
    print(f"\nsummary → {summary_path}")


if __name__ == "__main__":
    main()
