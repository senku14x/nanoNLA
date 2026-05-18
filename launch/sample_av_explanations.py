"""Qualitative: pick N val rows, generate explanations from a trained AV ckpt,
print side-by-side with gold + the input text. CJK fraction reported as a
smoke test for injection failure."""

import argparse
import unicodedata

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.injection import inject_at_marked_positions, karvonen_inject_in_residual
from nla.schema import normalize_activation, resolve_target_scale


def cjk_fraction(text):
    if not text:
        return 0.0
    cjk = sum(1 for c in text if 'CJK' in unicodedata.name(c, ''))
    return cjk / len(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--injection", choices=["embedding_replace", "karvonen"], required=True)
    p.add_argument("--injection-scale", default="sqrt_d_model")
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=200)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cfg = load_nla_config(args.sidecar, tok)
    inj_id = cfg.injection_token_id
    left_id = cfg.injection_left_neighbor_id
    right_id = cfg.injection_right_neighbor_id
    d_model = cfg.d_model
    inject_char = cfg.injection_char if hasattr(cfg, "injection_char") else "㈎"

    print(f"loading {args.ckpt_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt_dir, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    activation_box = {"v": None}
    input_ids_box = {"v": None}

    if args.injection == "embedding_replace":
        raw_scale = None if args.injection_scale in ("null", "None", "raw") else args.injection_scale
        if isinstance(raw_scale, str) and raw_scale != "sqrt_d_model":
            raw_scale = float(raw_scale)
        scale = resolve_target_scale(raw_scale, d_model)

        def embed_hook(_m, inputs, output):
            v = activation_box["v"]
            if v is None: return output
            v = normalize_activation(v.unsqueeze(0), scale)
            return inject_at_marked_positions(
                input_ids=inputs[0], embeddings=output, vectors=v,
                inj_id=inj_id, left_id=left_id, right_id=right_id,
            )
        model.get_input_embeddings().register_forward_hook(embed_hook)
    else:
        def karvonen_hook(_m, _i, output):
            v = activation_box["v"]
            if v is None: return output
            if isinstance(output, tuple):
                hidden, *rest = output
            else:
                hidden, rest = output, None
            hidden = karvonen_inject_in_residual(
                input_ids=input_ids_box["v"], resid=hidden, vectors=v.unsqueeze(0),
                inj_id=inj_id, left_id=left_id, right_id=right_id,
            )
            return (hidden, *rest) if rest is not None else hidden
        model.model.layers[1].register_forward_hook(karvonen_hook)

    val = pq.read_table(args.val_parquet)
    n = min(args.n_samples, val.num_rows)

    for i in range(n):
        prompt_msgs = val.column("prompt")[i].as_py()
        for m in prompt_msgs:
            if isinstance(m.get("content"), str):
                m["content"] = m["content"].replace("<INJECT>", inject_char)
        gold = val.column("response")[i].as_py()
        text = val.column("detokenized_text_truncated")[i].as_py()
        activation = torch.tensor(val.column("activation_vector")[i].as_py(),
                                  dtype=torch.float32, device="cuda")

        prompt_ids = tok.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        activation_box["v"] = activation
        # For karvonen, input_ids_box has to match the running forward — generate()
        # extends input_ids each step, so the hook needs to be position-aware.
        # The hook scans for marker token in the FULL input_ids; we feed full ids
        # at prefill and during generation the hook still finds the marker in
        # earlier positions, then no-ops because vec_idx >= expected. But the
        # neighbor check is strict — actually it'll see 1 marker, 1 expected → ok.
        # In generate's decode steps, input_ids is just the new token → 0 markers,
        # 0 expected → ok via no-op when vectors=[].
        # Punt: only inject at prefill by setting vectors=None during decode.
        # Simpler: monkey-patch — track call count.

        # Cleanest path: just do prefill manually + decode greedy w/ hook only on prefill.
        with torch.no_grad():
            # Manual greedy generation
            input_ids_box["v"] = prompt_ids
            input_ids = prompt_ids
            generated = []
            past = None
            for step in range(args.max_new_tokens):
                if step == 0:
                    out = model(input_ids, use_cache=True)
                else:
                    # Decode step — clear injection, no markers in new token
                    activation_box["v"] = None
                    input_ids_box["v"] = input_ids
                    out = model(input_ids=input_ids[:, -1:], past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
                generated.append(next_tok.item())
                input_ids = torch.cat([input_ids, next_tok], dim=-1)
                if next_tok.item() == tok.eos_token_id:
                    break

        gen_text = tok.decode(generated, skip_special_tokens=True)
        cjk = cjk_fraction(gen_text)
        print(f"\n========== Sample {i+1}/{n} ==========")
        print(f"INPUT TEXT (first 200 chars):\n  {text[:200]!r}\n")
        print(f"GOLD EXPLANATION:\n  {gold[:300]}\n")
        print(f"GENERATED ({args.injection}, scale={args.injection_scale}, CJK={cjk:.1%}):")
        print(f"  {gen_text[:400]}")


if __name__ == "__main__":
    main()
