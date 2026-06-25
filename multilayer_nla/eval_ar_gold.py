"""AR-only gold held-out eval: gold explanation -> shared AR -> FIXED [L23,L24,L25] -> FVE.

Distinct from the end-to-end metric (NO AV, NO generation, NO extraction): it isolates
the reconstructor. High AR-gold FVE but low end-to-end FVE => the AV / extraction is the
bottleneck; low AR-gold FVE => the reconstructor is. Predict-the-mean baseline comes
from THIS eval split's targets only. Reloads a saved AR ckpt (no training).

Run:
  python -m multilayer_nla.eval_ar_gold --base-ckpt Qwen/Qwen3-8B \
      --ar-ckpt $CKPT/ar/iter_0001000 --eval-parquet $SWEEP/ar_test.parquet \
      --summary $EVAL/ar_gold_test.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multilayer_nla.datasets import load_ar_sft_dataset
from multilayer_nla.train_ar_multi import _per_tap_baselines, evaluate_ar
from multilayer_nla.evaluate_e2e import load_critic


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    p.add_argument("--ar-ckpt", required=True)
    p.add_argument("--eval-parquet", required=True, help="ar_dev.parquet / ar_test.parquet (gold prompts)")
    p.add_argument("--summary", required=True)
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=0, help="0 = full split")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    critic, mse_scale = load_critic(args.base_ckpt, args.ar_ckpt, args.quant, device)

    rows = load_ar_sft_dataset(args.eval_parquet)
    baselines = _per_tap_baselines(rows, mse_scale)  # predict-the-mean on THIS split
    mse, fve, loss = evaluate_ar(critic, rows, tokenizer, mse_scale, baselines, device,
                                 args.max_len, args.batch_size, args.max_batches or None)
    summary = {
        "eval_parquet": args.eval_parquet, "ar_ckpt": args.ar_ckpt, "n_rows": len(rows),
        "fve_prev": fve[0], "fve_centre": fve[1], "fve_next": fve[2],
        "fve_overall": sum(fve) / len(fve), "loss": loss, "mse_scale": mse_scale,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"[ar-gold] {Path(args.eval_parquet).name}: gold FVE p/c/n "
          f"{fve[0]*100:.1f}/{fve[1]*100:.1f}/{fve[2]*100:.1f}%  overall {summary['fve_overall']*100:.1f}% "
          f"-> {args.summary}")


if __name__ == "__main__":
    main()
