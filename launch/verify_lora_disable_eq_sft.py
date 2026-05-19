"""Verify that `peft.disable_adapter()` on a freshly-wrapped (zero-LoRA) actor
produces the same logits as loading the AV-SFT checkpoint directly.

This is the KL anchor invariant for nla/train_rl_self_contained.py and
train_rl_vllm.py: we claim that the reference policy `D_KL(AV_φ || AV_φ_init)`
is computed by toggling LoRA off, which is only true if the disabled-adapter
forward is bit-identical (or floating-point-noise close) to the AV-SFT init.

If max abs diff is large (> ~1e-3 in bf16), the KL anchor is leaking
LoRA-side-effects somewhere and the paper's "KL toward init" semantics
isn't what we're computing.
"""

import argparse
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", required=True,
                   help="AV-SFT checkpoint dir (HF format)")
    p.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--use-rslora", action="store_true", default=True)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    print(f"prompt tokens: {ids.shape[1]}")

    print("\n=== A: fresh AV-SFT ===")
    a = AutoModelForCausalLM.from_pretrained(
        args.av_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    with torch.no_grad():
        logits_a = a(ids).logits.float()
    print(f"logits[0,-1,:5] = {logits_a[0,-1,:5].tolist()}")
    print(f"logits.norm = {logits_a.norm().item():.4f}")
    del a
    torch.cuda.empty_cache()

    print("\n=== B: AV-SFT + LoRA (zero-init) + disable_adapter ===")
    b = AutoModelForCausalLM.from_pretrained(
        args.av_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        use_rslora=args.use_rslora,
    )
    b = get_peft_model(b, lora_cfg)
    b.eval()
    with torch.no_grad(), b.disable_adapter():
        logits_b = b(ids).logits.float()
    print(f"logits[0,-1,:5] = {logits_b[0,-1,:5].tolist()}")
    print(f"logits.norm = {logits_b.norm().item():.4f}")

    print("\n=== diff ===")
    diff = (logits_a - logits_b).abs()
    print(f"max abs diff:  {diff.max().item():.6e}")
    print(f"mean abs diff: {diff.mean().item():.6e}")
    print(f"median abs:    {diff.median().item():.6e}")
    print(f"relative L2:   {(logits_a - logits_b).norm().item() / logits_a.norm().item():.6e}")

    # Also check log_softmax (what KL actually uses)
    lp_a = torch.log_softmax(logits_a, dim=-1)
    lp_b = torch.log_softmax(logits_b, dim=-1)
    lp_diff = (lp_a - lp_b).abs()
    print(f"\nlog_softmax max abs diff:  {lp_diff.max().item():.6e}")
    print(f"log_softmax mean abs diff: {lp_diff.mean().item():.6e}")

    # If max abs diff < ~1e-3 (typical bf16 noise floor), claim holds.
    # If > ~1e-2, something real is happening.
    if diff.max().item() < 1e-3:
        print("\n✅ disable_adapter is bf16-equivalent to fresh AV-SFT load")
    elif diff.max().item() < 1e-1:
        print("\n⚠️  small drift — within reasonable bf16 noise but worth flagging")
    else:
        print("\n❌ significant drift — PEFT's disable_adapter is NOT giving back AV-SFT init")


if __name__ == "__main__":
    main()
