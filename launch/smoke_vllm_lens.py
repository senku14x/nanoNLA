"""Smoke test vllm-lens Karvonen-style injection on Qwen3-8B.

Steps:
  1. Load Qwen3-8B base via vLLM.
  2. Construct a prompt mirroring the AV training prompt with the marker token.
  3. Inject a unit-random vector at layer 1, marker position, with norm_match=True.
  4. Generate ~50 tokens.
  5. Sanity: no CJK, first few tokens are something English/sensible.

If sample 1 returns CJK at >10%, injection is broken. If it returns format-OK,
vllm-lens is the right vehicle for RL rollouts.
"""

import argparse
import os
import unicodedata

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm_lens import SteeringVector


def cjk_fraction(text):
    if not text:
        return 0.0
    cjk = sum(1 for c in text if 'CJK' in unicodedata.name(c, ''))
    return cjk / len(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--injection-layer", type=int, default=1)
    p.add_argument("--n-trials", type=int, default=3)
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace-vast/pretrained_ckpts")
    os.environ.setdefault("HF_TOKEN", "YOUR_HF_TOKEN")

    print(f"loading {args.model} via vLLM …")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=512,
        enforce_eager=True,  # disable CUDA graphs so hooks fire cleanly
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    d_model = 4096  # Qwen3-8B

    # Marker = ㈎ U+320E per our sidecar. Single token id 149705 in Qwen3.
    inject_char = "㈎"
    inject_id_seq = tok.encode(inject_char, add_special_tokens=False)
    print(f"marker {inject_char!r} → token id {inject_id_seq}")
    assert len(inject_id_seq) == 1, "marker must tokenize to single token"

    # Mimic the AV training prompt (with <INJECT> already substituted)
    user_content = (
        "You are a meticulous AI researcher describing an activation vector.\n\n"
        f"<concept>{inject_char}</concept>\n\nPlease provide an explanation."
    )
    messages = [{"role": "user", "content": user_content}]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(prompt_text, add_special_tokens=False)
    marker_positions = [i for i, t in enumerate(ids) if t == inject_id_seq[0]]
    assert len(marker_positions) == 1, f"need exactly 1 marker, got {marker_positions}"
    marker_pos = marker_positions[0]
    print(f"prompt has {len(ids)} tokens, marker at position {marker_pos}")

    sp = SamplingParams(
        temperature=0.7, max_tokens=80, top_p=0.9,
    )

    # Trial A: BASELINE — no injection. Should produce some explanation of "㈎"
    # treated literally as a character. Expect mostly English (Qwen base does
    # not collapse to CJK on this prompt without the embedding-replace tripwire).
    print("\n=== TRIAL A: no injection (baseline) ===")
    out = llm.generate([prompt_text], sp)
    base_text = out[0].outputs[0].text
    print(f"  output: {base_text[:200]!r}")
    print(f"  CJK fraction: {cjk_fraction(base_text):.1%}")

    # Trials B+: random injected vectors at layer 1, norm_match=True
    for t in range(args.n_trials):
        torch.manual_seed(t)
        # vllm-lens 1.0 SteeringVector requires 2D (N_positions, d_model) or 3D activations.
        act = torch.randn(1, d_model, dtype=torch.float32)
        sv = SteeringVector(
            activations=act,
            layer_indices=[args.injection_layer],
            scale=1.0,
            norm_match=True,
            position_indices=[marker_pos],
        )
        sp_inj = SamplingParams(
            temperature=0.7, max_tokens=80, top_p=0.9,
            extra_args={"apply_steering_vectors": [sv]},
        )
        out = llm.generate([prompt_text], sp_inj)
        text = out[0].outputs[0].text
        print(f"\n=== TRIAL B{t}: random vector @ L{args.injection_layer}, norm_match ===")
        print(f"  output: {text[:200]!r}")
        print(f"  CJK fraction: {cjk_fraction(text):.1%}")


if __name__ == "__main__":
    main()
