"""Verify vllm-lens's norm_match=True matches our HF-based Karvonen injection.

Both should produce the same next-token logits / generation when given the
same prompt + activation + greedy decoding, IF norm_match implements
h + ||h|| * v/||v||.

Usage:
    python compare_hf_vs_vllm_lens.py --ckpt-hf /.../iter_0001000/hf \\
        --val-parquet /.../av_val.parquet --sidecar /.../av_val.parquet
"""

import argparse
import os
import unicodedata

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm_lens import SteeringVector

from nla.config import load_nla_config
from nla.injection import karvonen_inject_in_residual


def cjk_fraction(text):
    if not text:
        return 0.0
    return sum(1 for c in text if 'CJK' in unicodedata.name(c, '')) / len(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-hf", required=True, help="HF-format AV ckpt (Karvonen-trained)")
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--injection-layer", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=60)
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace-vast/pretrained_ckpts")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cfg = load_nla_config(args.sidecar, tok)
    inj_id = cfg.injection_token_id
    left_id = cfg.injection_left_neighbor_id
    right_id = cfg.injection_right_neighbor_id
    d_model = cfg.d_model
    inject_char = cfg.injection_char

    val = pq.read_table(args.val_parquet)

    # Pull n_samples val rows, get prompts + activations.
    rows = []
    for i in range(args.n_samples):
        msgs = val.column("prompt")[i].as_py()
        for m in msgs:
            if isinstance(m.get("content"), str):
                m["content"] = m["content"].replace("<INJECT>", inject_char)
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(prompt_text, add_special_tokens=False)
        positions = [j for j, t in enumerate(ids) if t == inj_id]
        assert len(positions) == 1
        activation = torch.tensor(val.column("activation_vector")[i].as_py(),
                                  dtype=torch.float32)
        rows.append({"prompt_text": prompt_text, "marker_pos": positions[0],
                     "activation": activation, "ids": ids})

    # =============== HF GENERATE WITH OUR KARVONEN HOOK ==================
    print("\n=== loading HF model ===")
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.ckpt_hf, torch_dtype=torch.bfloat16
    ).to("cuda")
    hf_model.eval()

    activation_box = {"v": None}
    input_ids_box = {"v": None}

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

    hf_model.model.layers[args.injection_layer].register_forward_hook(karvonen_hook)

    hf_outputs = []
    for r in rows:
        input_ids = torch.tensor([r["ids"]], dtype=torch.long, device="cuda")
        activation_box["v"] = r["activation"].to("cuda")
        input_ids_box["v"] = input_ids

        # Greedy generation, temperature=0 → fully deterministic
        with torch.no_grad():
            past = None
            generated = []
            cur = input_ids
            for step in range(args.max_new_tokens):
                if step == 0:
                    out = hf_model(cur, use_cache=True)
                else:
                    activation_box["v"] = None
                    out = hf_model(input_ids=cur[:, -1:], past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[0, -1].argmax(dim=-1, keepdim=True).unsqueeze(0)
                generated.append(next_tok.item())
                cur = torch.cat([cur, next_tok], dim=-1)
                if next_tok.item() == tok.eos_token_id:
                    break
        text = tok.decode(generated, skip_special_tokens=True)
        hf_outputs.append(text)
        print(f"  HF row{len(hf_outputs)-1}: {text[:150]!r}")

    # Free HF
    del hf_model
    torch.cuda.empty_cache()

    # =============== VLLM-LENS GENERATE ==================================
    print("\n=== loading vLLM model ===")
    # Converter only saved weights, no tokenizer files — point at base Qwen3-8B's.
    llm = LLM(model=args.ckpt_hf, tokenizer="Qwen/Qwen3-8B", dtype="bfloat16",
              gpu_memory_utilization=0.85, max_model_len=512, enforce_eager=True)

    vl_outputs = []
    for r in rows:
        sv = SteeringVector(
            activations=r["activation"].unsqueeze(0),
            layer_indices=[args.injection_layer],
            scale=1.0,
            norm_match=True,
            position_indices=[r["marker_pos"]],
        )
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                            extra_args={"apply_steering_vectors": [sv]})
        out = llm.generate([r["prompt_text"]], sp)
        text = out[0].outputs[0].text
        vl_outputs.append(text)
        print(f"  VLLM row{len(vl_outputs)-1}: {text[:150]!r}")

    # =============== COMPARE ============================================
    print("\n=== COMPARISON ===")
    for i, (h, v) in enumerate(zip(hf_outputs, vl_outputs)):
        # Compare token-by-token (after tokenization) — most robust
        h_ids = tok.encode(h, add_special_tokens=False)
        v_ids = tok.encode(v, add_special_tokens=False)
        prefix_match = 0
        for a, b in zip(h_ids, v_ids):
            if a == b: prefix_match += 1
            else: break
        n = min(len(h_ids), len(v_ids))
        print(f"\nrow {i}: HF {len(h_ids)} toks, vLLM {len(v_ids)} toks, "
              f"prefix-match {prefix_match}/{n} ({prefix_match/max(1,n):.0%})")
        print(f"  HF:   {h[:120]!r}")
        print(f"  vLLM: {v[:120]!r}")
        if prefix_match < n:
            print(f"  divergence at token {prefix_match}: "
                  f"HF={tok.decode([h_ids[prefix_match]])!r} vs "
                  f"vLLM={tok.decode([v_ids[prefix_match]])!r}")


if __name__ == "__main__":
    main()
