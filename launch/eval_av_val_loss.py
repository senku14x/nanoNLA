"""Held-out val NLL evaluator for the AV SFT sweep.

Loads a trained AV checkpoint (HF format), attaches the requested injection
hook (embedding-replace OR Karvonen layer-1 ADD), runs forward on each val row
with the gold activation injected, and computes mean response-token NLL.

usage:
    python eval_av_val_loss.py \\
        --ckpt-dir /path/to/iter_NNNN/hf \\
        --val-parquet /.../av_val.parquet \\
        --injection {embedding_replace,karvonen} \\
        --injection-scale sqrt_d_model   # only for embedding_replace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.injection import inject_at_marked_positions, karvonen_inject_in_residual
from nla.schema import normalize_activation, resolve_target_scale


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--sidecar", required=True,
                   help="parquet path; load_nla_config appends .nla_meta.yaml itself")
    p.add_argument("--injection", choices=["embedding_replace", "karvonen"], required=True)
    p.add_argument("--injection-scale", default="sqrt_d_model",
                   help="only for embedding_replace; one of sqrt_d_model, null, or float")
    p.add_argument("--max-rows", type=int, default=500,
                   help="cap val rows for speed (default 500)")
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    print(f"loading {args.ckpt_dir}")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cfg = load_nla_config(args.sidecar, tok)
    inj_id = cfg.injection_token_id
    left_id = cfg.injection_left_neighbor_id
    right_id = cfg.injection_right_neighbor_id
    d_model = cfg.d_model
    # Datasource substitutes <INJECT> → injection_char at training time; we must too.
    inject_char = cfg.injection_char if hasattr(cfg, "injection_char") else "㈎"
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt_dir, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    # The hook will read this. We set it before each forward.
    activation_box = {"v": None}

    if args.injection == "embedding_replace":
        # Resolve "sqrt_d_model" / null / float-string into a concrete float (or None).
        raw_scale = None if args.injection_scale in ("null", "None", "raw") else args.injection_scale
        if isinstance(raw_scale, str) and raw_scale != "sqrt_d_model":
            raw_scale = float(raw_scale)
        scale = resolve_target_scale(raw_scale, d_model)

        def embed_hook(_m, inputs, output):
            v = activation_box["v"]
            if v is None:
                return output
            v = normalize_activation(v.unsqueeze(0), scale)  # [1, d]
            return inject_at_marked_positions(
                input_ids=inputs[0], embeddings=output, vectors=v,
                inj_id=inj_id, left_id=left_id, right_id=right_id,
            )

        model.get_input_embeddings().register_forward_hook(embed_hook)
    else:  # karvonen
        input_ids_box = {"v": None}

        def karvonen_hook(_m, _i, output):
            v = activation_box["v"]
            if v is None:
                return output
            if isinstance(output, tuple):
                hidden, *rest = output
            else:
                hidden, rest = output, None
            hidden = karvonen_inject_in_residual(
                input_ids=input_ids_box["v"], resid=hidden,
                vectors=v.unsqueeze(0),
                inj_id=inj_id, left_id=left_id, right_id=right_id,
            )
            return (hidden, *rest) if rest is not None else hidden

        model.model.layers[1].register_forward_hook(karvonen_hook)

    val = pq.read_table(args.val_parquet)
    n_rows = min(args.max_rows, val.num_rows)
    print(f"evaluating on {n_rows} of {val.num_rows} rows")

    nll_sum = 0.0
    n_tokens = 0
    n_done = 0

    for i in range(n_rows):
        prompt_msgs = val.column("prompt")[i].as_py()
        # Substitute the <INJECT> placeholder for the actual marker char.
        for m in prompt_msgs:
            if isinstance(m.get("content"), str):
                m["content"] = m["content"].replace("<INJECT>", inject_char)
        response = val.column("response")[i].as_py()
        activation = torch.tensor(val.column("activation_vector")[i].as_py(),
                                  dtype=torch.float32, device="cuda")
        assert activation.shape == (d_model,)

        prompt_ids = tok.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        full_ids = tok.apply_chat_template(
            prompt_msgs + [{"role": "assistant", "content": response}],
            tokenize=True, return_tensors="pt"
        ).to("cuda")
        response_start = prompt_ids.shape[-1]

        activation_box["v"] = activation
        if args.injection == "karvonen":
            input_ids_box["v"] = full_ids

        with torch.no_grad():
            out = model(full_ids)
        activation_box["v"] = None
        if args.injection == "karvonen":
            input_ids_box["v"] = None

        logits = out.logits[0]  # [seq, vocab]
        shift_logits = logits[response_start - 1 : -1]
        shift_labels = full_ids[0, response_start:]
        if shift_labels.numel() == 0:
            continue
        nll = F.cross_entropy(
            shift_logits.float(), shift_labels, reduction="sum"
        ).item()
        nll_sum += nll
        n_tokens += int(shift_labels.numel())
        n_done += 1
        if n_done % 50 == 0:
            print(f"  [{n_done}/{n_rows}] running NLL/tok = {nll_sum/n_tokens:.4f}")

    mean_nll = nll_sum / max(1, n_tokens)
    ppl = float(torch.tensor(mean_nll).exp())
    print(f"\n=== {args.injection} (scale={args.injection_scale}) ===")
    print(f"rows: {n_done}")
    print(f"tokens: {n_tokens}")
    print(f"mean NLL/token: {mean_nll:.4f}")
    print(f"perplexity: {ppl:.3f}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps({
            "ckpt": args.ckpt_dir, "injection": args.injection,
            "injection_scale": args.injection_scale,
            "rows": n_done, "tokens": n_tokens,
            "mean_nll": mean_nll, "ppl": ppl,
        }, indent=2))


if __name__ == "__main__":
    main()
