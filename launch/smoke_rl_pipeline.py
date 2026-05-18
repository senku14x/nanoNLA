"""End-to-end NLA RL-lite pipeline smoke (without Miles GRPO loop).

For each row in rl_shuf.parquet (first N rows):
  1. Build the AV prompt with marker, the activation, the AR critic prompt.
  2. Generate an explanation via vllm-lens with Karvonen norm-matched ADD injection.
  3. Score it: tokenize critic prompt + extracted explanation, run AR forward,
     compute MSE between AR's predicted activation and the real activation.
  4. Aggregate reward = -mse_nrm across N rows.

If the rewards distribution is sensible (mean clearly below the "random" baseline),
the full RL training will be viable.
"""

import argparse
import os
import re
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm_lens import SteeringVector

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.schema import normalize_activation, resolve_target_scale


_EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def cjk_fraction(text):
    if not text:
        return 0.0
    return sum(1 for c in text if "CJK" in unicodedata.name(c, "")) / len(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", required=True, help="Karvonen-trained AV ckpt (HF dir)")
    p.add_argument("--ar-ckpt", required=True, help="Critic ckpt (HF dir w/ value_head.safetensors)")
    p.add_argument("--rl-parquet", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--n-rows", type=int, default=20)
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--injection-layer", type=int, default=1)
    args = p.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace-vast/pretrained_ckpts")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cfg = load_nla_config(args.sidecar, tok)
    inj_id = cfg.injection_token_id
    inject_char = cfg.injection_char
    d_model = cfg.d_model
    mse_scale = cfg.mse_scale

    print(f"loading AV checkpoint {args.av_ckpt} via vLLM …")
    llm = LLM(model=args.av_ckpt, tokenizer="Qwen/Qwen3-8B", dtype="bfloat16",
              gpu_memory_utilization=0.55, max_model_len=1024, enforce_eager=True)

    val = pq.read_table(args.rl_parquet)
    n = min(args.n_rows, val.num_rows)
    print(f"rolling out {n} samples")

    prompt_texts = []
    marker_positions = []
    activations = []
    for i in range(n):
        msgs = val.column("prompt")[i].as_py()
        for m in msgs:
            if isinstance(m.get("content"), str):
                m["content"] = m["content"].replace("<INJECT>", inject_char)
        prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(prompt_text, add_special_tokens=False)
        positions = [j for j, t in enumerate(ids) if t == inj_id]
        assert len(positions) == 1, f"row {i}: {len(positions)} markers"
        prompt_texts.append(prompt_text)
        marker_positions.append(positions[0])
        activations.append(torch.tensor(
            val.column("activation_vector")[i].as_py(), dtype=torch.float32
        ))

    # Generate all in one batched call. vllm-lens accepts a list of steering
    # vector requests via extra_args, one per prompt.
    steering_vectors = [
        SteeringVector(
            activations=activations[i].unsqueeze(0),  # 2D: [1, d_model]
            layer_indices=[args.injection_layer],
            scale=1.0,
            norm_match=True,
            position_indices=[marker_positions[i]],
        )
        for i in range(n)
    ]
    sps = [
        SamplingParams(
            temperature=1.0, max_tokens=args.max_new_tokens, top_p=0.95,
            extra_args={"apply_steering_vectors": [steering_vectors[i]]},
        )
        for i in range(n)
    ]
    outputs = llm.generate(prompt_texts, sps)

    responses = [o.outputs[0].text for o in outputs]
    cjk_frac = [cjk_fraction(r) for r in responses]
    extracted = []
    for r in responses:
        m = _EXPLANATION_RE.search(r)
        extracted.append(m.group(1).strip() if m else None)
    print(f"extraction success: {sum(e is not None for e in extracted)}/{n}")
    print(f"mean CJK fraction: {sum(cjk_frac)/n:.1%}")
    for i in range(min(3, n)):
        print(f"  [{i}] CJK={cjk_frac[i]:.0%} extracted={extracted[i] is not None}")
        if extracted[i]:
            print(f"      {extracted[i][:160]!r}")

    # Free vLLM memory before loading the AR
    del llm
    torch.cuda.empty_cache()

    # Load AR critic (truncated + Linear(d,d) head)
    print(f"\nloading AR critic {args.ar_ckpt} …")
    critic = NLACriticModel.from_pretrained(args.ar_ckpt, torch_dtype=torch.bfloat16).to("cuda")
    critic.eval()
    # diag: confirm value_head weights healthy
    vh = critic.value_head.weight
    print(f"value_head.weight: shape={tuple(vh.shape)} dtype={vh.dtype} "
          f"isnan={vh.isnan().any().item()} norm={vh.float().norm().item():.3f}")
    n_layers = len(critic.backbone.model.layers) if hasattr(critic.backbone, "model") else len(critic.backbone.layers)
    print(f"backbone layers kept: {n_layers}")
    # Smoke forward on a trivial input
    with torch.no_grad():
        smoke_ids = tok.encode("Summary of the following text: <text>hello world</text> <summary>",
                               add_special_tokens=False)
        smoke_x = torch.tensor([smoke_ids], dtype=torch.long, device="cuda")
        smoke_out = critic(input_ids=smoke_x)
        smoke_h = smoke_out.backbone_last_hidden[0, -1].float()
        smoke_v = smoke_out.values[0, -1].float()
        print(f"smoke last-hidden: norm={smoke_h.norm():.3f} isnan={smoke_h.isnan().any().item()}")
        print(f"smoke value: norm={smoke_v.norm():.3f} isnan={smoke_v.isnan().any().item()}")

    # Build critic prompts from extracted explanations + score
    mse_scale_f = resolve_target_scale(mse_scale, d_model)
    rewards = []
    template = cfg.critic_prompt_template
    assert template is not None, "critic_prompt_template missing from sidecar"

    for i, expl in enumerate(extracted):
        if expl is None:
            rewards.append(None)
            continue
        # Sidecar template uses Python format-style {explanation} placeholder.
        critic_text = template.format(explanation=expl)
        ids = tok.encode(critic_text, add_special_tokens=False)
        x = torch.tensor([ids], dtype=torch.long, device="cuda")
        with torch.no_grad():
            out = critic(input_ids=x)
        pred = out.values[0, -1].float()  # [d_model], at last token
        gold = activations[i].to("cuda")
        if i < 3:
            print(f"  [{i}] critic ntoks={len(ids)} pred.norm={pred.norm().item():.3f} "
                  f"gold.norm={gold.norm().item():.3f} pred.isnan={pred.isnan().any().item()} "
                  f"pred[:3]={pred[:3].tolist()}")
        # mse_nrm: both normalized to mse_scale, MSE between them
        pred_n = normalize_activation(pred.unsqueeze(0), mse_scale_f)[0]
        gold_n = normalize_activation(gold.unsqueeze(0), mse_scale_f)[0]
        mse = F.mse_loss(pred_n, gold_n).item()
        if i < 3:
            print(f"      mse_scale_f={mse_scale_f}  pred_n.isnan={pred_n.isnan().any().item()} "
                  f"gold_n.isnan={gold_n.isnan().any().item()}  mse={mse}")
        rewards.append(-mse)

    valid = [r for r in rewards if r is not None]
    n_drop = sum(1 for r in rewards if r is None)
    print(f"\n=== REWARDS ===")
    print(f"valid samples: {len(valid)}, dropped (extract fail): {n_drop}")
    if valid:
        valid_t = torch.tensor(valid)
        print(f"mean reward: {valid_t.mean().item():.4f}")
        print(f"std:         {valid_t.std().item():.4f}")
        print(f"min:         {valid_t.min().item():.4f}")
        print(f"max:         {valid_t.max().item():.4f}")
        # Random-bad baseline: random_explanation → ~-1.0 reward (random direction)
        # Trained baseline AR: ~0.42 train MSE → -0.42 reward on training distribution


if __name__ == "__main__":
    main()
