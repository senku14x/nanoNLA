"""Compare pre-RL (AV-SFT only) vs post-RL (AV-SFT + GRPO LoRA) reconstruction reward.

Loads:
  - AV-SFT ckpt as the "pre-RL" actor
  - same AV-SFT ckpt + a LoRA adapter as the "post-RL" actor
  - the frozen AR critic
  - the rl_val.parquet held-out set (so we measure generalisation, not training overfit)

For each row:
  generate explanation with Karvonen injection → score with AR → reward = -mse_nrm

Outputs: mean / std / min / max reward for both, plus pairwise reward delta on
matched prompts. Reuses the proven smoke_rl_pipeline.py components.
"""

import argparse
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.injection import karvonen_inject_in_residual
from nla.models import NLACriticModel
from nla.schema import EXPLANATION_RE, normalize_activation, resolve_target_scale


def cjk_frac(text):
    if not text:
        return 0.0
    return sum(1 for c in text if "CJK" in unicodedata.name(c, "")) / len(text)


def register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1):
    state = {"input_ids": None}

    def embed_hook(module, args, kwargs, output):
        ids = kwargs.get("input") if kwargs else None
        if ids is None and args:
            ids = args[0]
        state["input_ids"] = ids
        return output

    def layer_hook(module, args, output):
        if isinstance(output, tuple):
            resid, *rest = output
        else:
            resid, rest = output, None
        input_ids = state["input_ids"]
        if input_ids is None or resid.shape[1] < 2:
            return output
        v = vectors_ref[0]
        if v is None or v.shape[0] == 0:
            return output
        if (input_ids == inj_id).sum().item() == 0:
            return output
        injected = karvonen_inject_in_residual(
            input_ids, resid, v, inj_id, left_id, right_id,
        )
        if rest is None:
            return injected
        return (injected, *rest)

    model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    target = model.base_model if hasattr(model, "base_model") else model
    while hasattr(target, "model") and not hasattr(target, "layers"):
        target = target.model
    target.layers[layer_idx].register_forward_hook(layer_hook)


def generate_and_score(actor, critic, tokenizer, rows, cfg, mse_scale_f, device, max_new=150):
    """Returns list of {response, explanation, reward, cjk}."""
    inj_id = cfg.injection_token_id
    inject_char = cfg.injection_char
    template = cfg.critic_prompt_template
    results = []
    vectors_ref = [None]
    register_karvonen_hook(actor, vectors_ref, inj_id,
                           cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id)

    actor.eval()
    for i, row in enumerate(rows):
        msgs = [{**m, "content": m["content"].replace("<INJECT>", inject_char)}
                if isinstance(m.get("content"), str) else m
                for m in row["prompt"]]
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_t = torch.tensor([ids], dtype=torch.long, device=device)
        activation = torch.tensor(row["activation"], dtype=torch.float32).unsqueeze(0).to(device)
        vectors_ref[0] = activation
        try:
            with torch.no_grad():
                out = actor.generate(
                    input_ids=prompt_t,
                    attention_mask=torch.ones_like(prompt_t),
                    max_new_tokens=max_new,
                    do_sample=True, temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
        finally:
            vectors_ref[0] = None
        resp_ids = out.sequences[0, prompt_t.shape[1]:].tolist()
        response = tokenizer.decode(resp_ids, skip_special_tokens=True)
        m = EXPLANATION_RE.search(response)
        expl = m.group(1).strip() if m else None
        reward = -2.0
        if expl is not None:
            crit_text = template.format(explanation=expl)
            crit_ids = tokenizer.encode(crit_text, add_special_tokens=False)
            if len(crit_ids) <= 1024:
                x = torch.tensor([crit_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    cout = critic(input_ids=x)
                pred = cout.values[0, -1].float()
                gold = activation[0].float()
                pn = normalize_activation(pred.unsqueeze(0), mse_scale_f)[0]
                gn = normalize_activation(gold.unsqueeze(0), mse_scale_f)[0]
                mse = F.mse_loss(pn, gn).item()
                reward = -mse if np.isfinite(mse) else -2.0
        results.append({
            "idx": i,
            "response": response,
            "explanation": expl,
            "reward": reward,
            "cjk": cjk_frac(response),
        })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", required=True)
    p.add_argument("--ar-ckpt", required=True)
    p.add_argument("--rl-lora", required=True, help="Path to RL-trained LoRA adapter dir")
    p.add_argument("--val-parquet", required=True, help="Held-out rl_val parquet")
    p.add_argument("--sidecar", required=True)
    p.add_argument("--n-rows", type=int, default=64)
    p.add_argument("--max-new", type=int, default=150)
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace-vast/pretrained_ckpts")
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cfg = load_nla_config(args.sidecar, tokenizer)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)

    pf = pq.ParquetFile(args.val_parquet)
    rg = pf.read_row_group(0, columns=["prompt", "activation_vector"]).slice(0, args.n_rows)
    rows = [{"prompt": p_, "activation": a}
            for p_, a in zip(rg.column("prompt").to_pylist(),
                              rg.column("activation_vector").to_pylist())]
    print(f"evaluating on {len(rows)} held-out prompts")

    print(f"\n=== Critic load ===")
    critic = NLACriticModel.from_pretrained(args.ar_ckpt, torch_dtype=torch.bfloat16).to(device)
    critic.eval()
    for p_ in critic.parameters():
        p_.requires_grad_(False)

    print(f"\n=== PRE-RL: {args.av_ckpt} ===")
    actor_pre = AutoModelForCausalLM.from_pretrained(
        args.av_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    pre_results = generate_and_score(actor_pre, critic, tokenizer, rows, cfg, mse_scale_f, device, args.max_new)
    pre_rewards = np.array([r["reward"] for r in pre_results])
    pre_ext = sum(r["explanation"] is not None for r in pre_results) / len(pre_results)
    print(f"  mean reward: {pre_rewards.mean():.4f}  std: {pre_rewards.std():.4f}  "
          f"min: {pre_rewards.min():.4f}  max: {pre_rewards.max():.4f}  "
          f"ext_rate: {pre_ext:.0%}")

    del actor_pre
    torch.cuda.empty_cache()

    print(f"\n=== POST-RL: {args.av_ckpt} + LoRA {args.rl_lora} ===")
    actor_post = AutoModelForCausalLM.from_pretrained(
        args.av_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    actor_post = PeftModel.from_pretrained(actor_post, args.rl_lora)
    actor_post.eval()
    post_results = generate_and_score(actor_post, critic, tokenizer, rows, cfg, mse_scale_f, device, args.max_new)
    post_rewards = np.array([r["reward"] for r in post_results])
    post_ext = sum(r["explanation"] is not None for r in post_results) / len(post_results)
    print(f"  mean reward: {post_rewards.mean():.4f}  std: {post_rewards.std():.4f}  "
          f"min: {post_rewards.min():.4f}  max: {post_rewards.max():.4f}  "
          f"ext_rate: {post_ext:.0%}")

    delta = post_rewards - pre_rewards
    print(f"\n=== DELTA (post - pre, matched prompts) ===")
    print(f"  mean Δ: {delta.mean():+.4f}  std: {delta.std():.4f}")
    print(f"  Δ > 0 (RL improved): {(delta > 0).sum()}/{len(delta)}")
    print(f"  Δ < 0 (RL hurt):    {(delta < 0).sum()}/{len(delta)}")

    print("\n=== sample side-by-sides ===")
    for i in range(min(3, len(rows))):
        print(f"\n[idx {i}] Δ={delta[i]:+.4f}")
        print(f"  pre  (r={pre_results[i]['reward']:.3f}): {pre_results[i]['explanation'][:140] if pre_results[i]['explanation'] else '<extraction failed>'}")
        print(f"  post (r={post_results[i]['reward']:.3f}): {post_results[i]['explanation'][:140] if post_results[i]['explanation'] else '<extraction failed>'}")


if __name__ == "__main__":
    main()
