"""Self-contained NLA GRPO training: Karvonen injection, LoRA actor.

Architecture:
  - Actor: Qwen3-8B + Karvonen layer-1 ADD injection (from AV-SFT ckpt).
           Wrapped with LoRA so backbone stays frozen (memory: 16GB base + ~100MB
           LoRA + small Adam states + activations all fit on one H200).
  - Reference policy: same model with LoRA adapter disabled (PEFT context manager).
           No second model copy.
  - Critic: NLACriticModel (truncated K+1-layer backbone + value head). Frozen,
            bf16, eval-only — produces predicted activation given the actor's
            explanation. Reward = -MSE(pred, gold).
  - Rollout: HF model.generate() with the same Karvonen hook used in training.
             Slower than vLLM but no weight-sync complexity. On-policy.

GRPO objective (DeepSeekMath / DeepSeek-R1):
  L = -E[min(r * A, clip(r, 1-eps, 1+eps) * A)] + beta * KL(pi || pi_ref)
  where r = exp(log_p_new - log_p_old), token-level
        A = group-relative reward, per-prompt baseline
        KL ≈ exp(log_p_ref - log_p_new) - (log_p_ref - log_p_new) - 1 (k3 estimator)

Per step:
  1. Sample B prompts from rl_shuf.parquet (each carries a gold activation v).
  2. Generate G samples per prompt with sampling temperature.
     Collect old log_probs from generate's output_scores.
  3. Extract <explanation>; failed extractions get reward = -2.0 (paper default,
     equals MSE on fully-orthogonal unit vectors — i.e. maximally bad).
  4. Score with critic → r_ij = -mse_nrm.
  5. Group-relative advantage: A_ij = (r_ij - mean_j) / std_j (per prompt group).
  6. Training-mode forward of the actor: compute new log_probs (LoRA active).
  7. Reference forward (same batch, LoRA disabled): compute ref log_probs.
  8. GRPO loss, backward + Adam.
"""

import argparse
import math
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

from nla.config import load_nla_config
from nla.injection import karvonen_inject_in_residual
from nla.models import NLACriticModel
from nla.schema import (
    EXPLANATION_RE,
    extract_explanation,
    normalize_activation,
    resolve_target_scale,
)


def cjk_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if "CJK" in unicodedata.name(c, "")) / len(text)


def _register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1):
    """Attach a forward hook on layer `layer_idx` of the HF actor.

    The hook reads `vectors_ref[0]` (a [N, d] tensor set by the caller before
    each forward). N must equal the number of marker positions in the current
    input_ids. Hook is a no-op when seq_len < 2 (autoregressive cache-step
    forwards pass a single new token; no marker present).
    """
    # input_ids isn't passed to layer hooks — capture it via an embedding hook
    # that stashes a thread-local ref.
    state = {"input_ids": None}

    def embed_hook(module, args, kwargs, output):
        # args[0] is input_ids in HF embeddings; sometimes passed as kwarg.
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
        # vectors_ref is a list with one tensor; updated by caller pre-forward.
        v = vectors_ref[0]
        if v is None or v.shape[0] == 0:
            return output
        # Match the marker count to vectors expected.
        matches_count = (input_ids == inj_id).sum().item()
        if matches_count == 0:
            return output
        # Only inject when marker count matches available vectors — otherwise
        # we'd assert. (Should always match in this flow.)
        injected = karvonen_inject_in_residual(
            input_ids, resid, v, inj_id, left_id, right_id,
        )
        if rest is None:
            return injected
        return (injected, *rest)

    emb_handle = model.get_input_embeddings().register_forward_hook(embed_hook, with_kwargs=True)
    base = model.base_model if hasattr(model, "base_model") else model
    # PEFT-wrapped: layers are under base_model.model.model.layers
    target = base
    while hasattr(target, "model") and not hasattr(target, "layers"):
        target = target.model
    # `target` should now be the inner module with .layers
    layer_handle = target.layers[layer_idx].register_forward_hook(layer_hook)
    return emb_handle, layer_handle


def load_rl_dataset(parquet_path, n_max=None):
    """Streaming load — reads only the columns we need, only the rows we need.

    Full-table read of a 3.7GB parquet with .as_py() on every row's 4096-float
    activation_vector takes 5+ minutes; rowgroup-by-rowgroup streaming with
    early stop is sub-second for n_max=200 and ~30s for n_max=10000.
    """
    import pyarrow.parquet as pq_inner
    pf = pq_inner.ParquetFile(parquet_path)
    rows = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=["prompt", "activation_vector"])
        n_in_rg = rg.num_rows
        # Slice first — to_pylist() on a 5000-row column with 4096-float
        # activations is the bottleneck (~30s); take only what we need.
        take = n_in_rg if n_max is None else min(n_max - len(rows), n_in_rg)
        rg = rg.slice(0, take)
        prompts = rg.column("prompt").to_pylist()
        acts = rg.column("activation_vector").to_pylist()
        for p, a in zip(prompts, acts):
            rows.append({"prompt": p, "activation": a})
    return rows


def build_prompt_text(prompt_msgs, inject_char, tokenizer):
    """Apply chat template; substitute <INJECT> placeholder."""
    msgs = [
        {**m, "content": m["content"].replace("<INJECT>", inject_char)}
        if isinstance(m.get("content"), str)
        else m
        for m in prompt_msgs
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def rollout_one_prompt(
    actor, tokenizer, prompt_text, activation, vectors_ref,
    inj_id, group_size, max_new_tokens, temperature, device,
):
    """Generate `group_size` samples for one prompt; capture old log-probs per response token."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    batched = prompt_t.expand(group_size, -1).contiguous()
    v_batch = activation.unsqueeze(0).expand(group_size, -1).contiguous().to(device).float()
    vectors_ref[0] = v_batch
    try:
        gen_out = actor.generate(
            input_ids=batched,
            attention_mask=torch.ones_like(batched),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            # NO top_p — see comment below. Keeps rollout distribution = training
            # distribution so importance ratio stays well-behaved.
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_logits=True,  # RAW pre-processor logits — scores are post-top_p
                                 # and assign -inf to filtered tokens, which would
                                 # make exp(new_lp - old_lp) = inf in GRPO loss.
        )
    finally:
        vectors_ref[0] = None
    full_ids = gen_out.sequences  # [G, prompt_len + new_len]
    # gen_out.logits: tuple of [G, V], RAW pre-softmax model logits at each step.
    # Captured under the SAME hook-applied forward used for training, so old_lp
    # and new_lp come from the same model snapshot at step 0 (then drift only
    # as LoRA weights update).
    scores = gen_out.logits  # tuple of [G, V]
    prompt_len = prompt_t.shape[1]
    responses = []
    for g in range(group_size):
        resp_ids = full_ids[g, prompt_len:].tolist()
        text = tokenizer.decode(resp_ids, skip_special_tokens=True)
        # Collect old log_p for each generated token.
        old_logp = []
        for t, step_logits in enumerate(scores):
            if t >= len(resp_ids):
                break
            lp = F.log_softmax(step_logits[g].float(), dim=-1)
            old_logp.append(lp[resp_ids[t]].item())
        responses.append({
            "text": text,
            "full_ids": full_ids[g],
            "prompt_len": prompt_len,
            "old_logp": torch.tensor(old_logp, dtype=torch.float32),
            "n_resp": len(old_logp),
        })
    return responses


def score_with_critic(
    critic, tokenizer, explanations, activations, template, mse_scale_f, device,
):
    """Returns list of rewards (None for failed extractions)."""
    rewards = []
    for expl, act in zip(explanations, activations):
        if expl is None:
            rewards.append(None)
            continue
        text = template.format(explanation=expl)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 1024:
            rewards.append(None)
            continue
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = critic(input_ids=x)
        pred = out.values[0, -1].float()
        gold = act.to(device).float()
        pred_n = normalize_activation(pred.unsqueeze(0), mse_scale_f)[0]
        gold_n = normalize_activation(gold.unsqueeze(0), mse_scale_f)[0]
        mse = F.mse_loss(pred_n, gold_n).item()
        if not math.isfinite(mse):
            rewards.append(None)
            continue
        rewards.append(-mse)
    return rewards


def compute_token_logps(
    actor, tokenizer, full_ids_list, prompt_lens, activations, vectors_ref,
    device, micro_batch=2, use_ref=False,
):
    """Compute per-token log P(response_t | prefix_<t) for each sample.

    Returns: list of 1-D tensors (length = n_response_tokens for each sample),
             each ON THE GRAPH (if use_ref=False, with grad; if use_ref=True, no grad).
    """
    out = []
    for chunk_start in range(0, len(full_ids_list), micro_batch):
        chunk = list(range(chunk_start, min(chunk_start + micro_batch, len(full_ids_list))))
        max_len = max(full_ids_list[i].numel() for i in chunk)
        pad_id = tokenizer.eos_token_id
        batch_ids = torch.full(
            (len(chunk), max_len), pad_id, dtype=torch.long, device=device,
        )
        attn = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        for row, i in enumerate(chunk):
            L = full_ids_list[i].numel()
            batch_ids[row, :L] = full_ids_list[i].to(device)
            attn[row, :L] = 1
        v_batch = torch.stack(
            [activations[i].to(device).float() for i in chunk], dim=0,
        )
        vectors_ref[0] = v_batch
        try:
            if use_ref:
                # Reference policy = base model with LoRA disabled.
                with torch.no_grad(), actor.disable_adapter():
                    logits = actor(input_ids=batch_ids, attention_mask=attn).logits
            else:
                logits = actor(input_ids=batch_ids, attention_mask=attn).logits
        finally:
            vectors_ref[0] = None
        logp = F.log_softmax(logits.float(), dim=-1)
        for row, i in enumerate(chunk):
            L = full_ids_list[i].numel()
            p_len = prompt_lens[i]
            if L <= p_len:
                out.append(torch.zeros(0, device=device))
                continue
            target_ids = batch_ids[row, p_len:L]
            pred_logits_idx = torch.arange(p_len - 1, L - 1, device=device)
            gathered = logp[row].index_select(0, pred_logits_idx)
            tok_logp = gathered.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            out.append(tok_logp)
    return out


def grpo_loss(
    new_logps, old_logps, ref_logps, advantages, clip_eps=0.2, kl_beta=0.04,
):
    """GRPO clipped surrogate + k3 KL estimator. Per-token, then per-sample mean,
    then batch mean.

    new_logps, old_logps, ref_logps: lists of 1-D tensors, one per sample, length=n_resp.
    advantages: [N] tensor (one scalar per sample, broadcast over its tokens).
    """
    sample_losses = []
    sample_kls = []
    sample_clip_fracs = []
    for new_lp, old_lp, ref_lp, A in zip(new_logps, old_logps, ref_logps, advantages):
        if new_lp.numel() == 0:
            continue
        # log_p ratio = new - old (per token); ratio = exp(log_p_new - log_p_old).
        # old/ref are detached (no grad needed).
        old_lp = old_lp.detach()
        ref_lp = ref_lp.detach()
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        # A is a scalar; broadcast over tokens.
        unclipped_obj = ratio * A
        clipped_obj = clipped * A
        # GRPO/PPO: take the min — pessimistic (penalize policy moving too far).
        surrogate = torch.minimum(unclipped_obj, clipped_obj)
        # k3 KL estimator (unbiased, low-variance): kl ≈ exp(δ) - δ - 1 where
        # δ = ref - new. Always ≥ 0.
        delta = ref_lp - new_lp
        kl = (torch.exp(delta) - delta - 1.0)
        # Per-sample loss: -mean_t(surrogate - beta * kl).
        per_tok_loss = -(surrogate - kl_beta * kl)
        sample_losses.append(per_tok_loss.mean())
        sample_kls.append(kl.mean().detach())
        sample_clip_fracs.append(
            ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps))
            .float().mean().detach()
        )
    if not sample_losses:
        return None, {}
    loss = torch.stack(sample_losses).mean()
    metrics = {
        "kl_mean": torch.stack(sample_kls).mean().item(),
        "clip_frac": torch.stack(sample_clip_fracs).mean().item(),
    }
    return loss, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--av-ckpt", required=True)
    p.add_argument("--ar-ckpt", required=True)
    p.add_argument("--rl-parquet", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--batch-prompts", type=int, default=8,
                   help="prompts per step")
    p.add_argument("--group-size", type=int, default=4,
                   help="samples per prompt (for group baseline)")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--logp-micro-batch", type=int, default=2)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--max-rows", type=int, default=None,
                   help="cap rows from rl parquet (avoids 3.7GB full-load for smoke runs)")
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-beta", type=float, default=0.04)
    p.add_argument("--wandb-project", default="nla-qwen3-8b")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda"
    os.environ.setdefault("HF_HOME", "/workspace-vast/pretrained_ckpts")

    # ---- tokenizer + nla config ----
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cfg = load_nla_config(args.sidecar, tokenizer)
    inj_id = cfg.injection_token_id
    left_id = cfg.injection_left_neighbor_id
    right_id = cfg.injection_right_neighbor_id
    inject_char = cfg.injection_char
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    template = cfg.critic_prompt_template
    assert template is not None, "critic_prompt_template missing"
    print(f"[cfg] inj_id={inj_id} mse_scale_f={mse_scale_f} d_model={cfg.d_model}")

    # ---- actor (LoRA-wrapped) ----
    print(f"[actor] loading {args.av_ckpt}")
    actor = AutoModelForCausalLM.from_pretrained(
        args.av_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    actor = get_peft_model(actor, lora_cfg)
    actor.print_trainable_parameters()
    actor.train()

    # ---- critic (frozen) ----
    print(f"[critic] loading {args.ar_ckpt}")
    critic = NLACriticModel.from_pretrained(
        args.ar_ckpt, torch_dtype=torch.bfloat16,
    ).to(device)
    critic.eval()
    for p_ in critic.parameters():
        p_.requires_grad_(False)
    print(f"[critic] value_head shape={tuple(critic.value_head.weight.shape)}")

    # ---- karvonen hook on actor ----
    vectors_ref = [None]
    _register_karvonen_hook(actor, vectors_ref, inj_id, left_id, right_id, layer_idx=1)

    # ---- dataset ----
    print(f"[data] loading {args.rl_parquet} (max_rows={args.max_rows})", flush=True)
    rows = load_rl_dataset(args.rl_parquet, n_max=args.max_rows)
    print(f"[data] {len(rows)} rows", flush=True)

    # ---- optimizer ----
    trainable = [p for p in actor.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # ---- wandb ----
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pending_idxs = list(range(len(rows)))
    rng.shuffle(pending_idxs)
    cursor = 0

    for step in range(args.num_steps):
        t0 = time.time()
        # ---- batch select ----
        if cursor + args.batch_prompts > len(pending_idxs):
            rng.shuffle(pending_idxs)
            cursor = 0
        batch_idxs = pending_idxs[cursor : cursor + args.batch_prompts]
        cursor += args.batch_prompts

        # ---- rollouts ----
        actor.eval()
        all_full_ids = []
        all_prompt_lens = []
        all_activations = []
        all_explanations = []
        all_response_text = []
        all_prompt_group = []
        all_old_logps = []  # 1-D tensor per sample
        for gi, row_idx in enumerate(batch_idxs):
            row = rows[row_idx]
            prompt_text = build_prompt_text(row["prompt"], inject_char, tokenizer)
            activation = torch.tensor(row["activation"], dtype=torch.float32)
            responses = rollout_one_prompt(
                actor, tokenizer, prompt_text, activation, vectors_ref,
                inj_id, args.group_size, args.max_new_tokens, args.temperature, device,
            )
            for r in responses:
                expl = extract_explanation(r["text"])
                all_full_ids.append(r["full_ids"])
                all_prompt_lens.append(r["prompt_len"])
                all_activations.append(activation)
                all_explanations.append(expl)
                all_response_text.append(r["text"])
                all_prompt_group.append(gi)
                all_old_logps.append(r["old_logp"].to(device))

        # ---- scoring ----
        rewards = score_with_critic(
            critic, tokenizer, all_explanations, all_activations,
            template, mse_scale_f, device,
        )
        # Paper's FAILED_EXTRACTION_REWARD = -2.0 (nla/reward.py): MSE on
        # fully-orthogonal unit vectors is 2.0, so this is the "worst possible"
        # critic outcome. Same penalty as a fully wrong direction.
        rewards_filled = [-2.0 if r is None else r for r in rewards]
        rewards_t = torch.tensor(rewards_filled, dtype=torch.float32, device=device)

        # ---- GRPO group-relative advantage (per-prompt mean & std) ----
        group_t = torch.tensor(all_prompt_group, dtype=torch.long, device=device)
        adv = torch.zeros_like(rewards_t)
        for gi in range(args.batch_prompts):
            mask = group_t == gi
            if mask.sum() == 0:
                continue
            group_r = rewards_t[mask]
            mu = group_r.mean()
            sd = group_r.std() if group_r.numel() > 1 else torch.tensor(1.0, device=device)
            adv[mask] = (group_r - mu) / (sd + 1e-6)

        # ---- training-mode forward: new log_probs (LoRA active, with grad) ----
        actor.train()
        new_logps = compute_token_logps(
            actor, tokenizer, all_full_ids, all_prompt_lens, all_activations,
            vectors_ref, device, micro_batch=args.logp_micro_batch, use_ref=False,
        )
        # Reference forward: same data, LoRA disabled, no grad.
        ref_logps = compute_token_logps(
            actor, tokenizer, all_full_ids, all_prompt_lens, all_activations,
            vectors_ref, device, micro_batch=args.logp_micro_batch, use_ref=True,
        )

        # ---- GRPO loss ----
        loss, grpo_metrics = grpo_loss(
            new_logps, all_old_logps, ref_logps, adv,
            clip_eps=args.clip_eps, kl_beta=args.kl_beta,
        )
        if loss is None:
            print(f"step {step}: no valid samples (all empty responses), skipping")
            continue
        if not torch.isfinite(loss):
            print(
                f"step {step}: loss={loss.item()} non-finite (kl={grpo_metrics.get('kl_mean')}, "
                f"clip_frac={grpo_metrics.get('clip_frac')}). Skipping update to avoid "
                f"corrupting LoRA weights.",
                flush=True,
            )
            optim.zero_grad()
            continue

        optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optim.step()

        # ---- logging ----
        valid_rewards = [r for r in rewards if r is not None]
        n_valid = len(valid_rewards)
        n_total = len(rewards)
        extraction_rate = n_valid / n_total if n_total else 0
        mean_cjk = (
            sum(cjk_fraction(t) for t in all_response_text) / max(len(all_response_text), 1)
        )
        n_resps_t = torch.tensor(
            [lp.numel() for lp in new_logps], dtype=torch.float32, device=device,
        )
        log = {
            "step": step,
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "reward_mean": float(np.mean(valid_rewards)) if valid_rewards else float("nan"),
            "reward_std": float(np.std(valid_rewards)) if valid_rewards else float("nan"),
            "reward_min": float(np.min(valid_rewards)) if valid_rewards else float("nan"),
            "reward_max": float(np.max(valid_rewards)) if valid_rewards else float("nan"),
            "advantage_mean": adv.mean().item(),
            "advantage_std": adv.std().item(),
            "extraction_rate": extraction_rate,
            "mean_cjk": mean_cjk,
            "mean_resp_len": n_resps_t.mean().item(),
            "kl_mean": grpo_metrics.get("kl_mean", 0.0),
            "clip_frac": grpo_metrics.get("clip_frac", 0.0),
            "wall_s": time.time() - t0,
        }
        print(
            f"step {step:04d} | loss {log['loss']:.4f} | r {log['reward_mean']:.3f} "
            f"| kl {log['kl_mean']:.4f} | clip {log['clip_frac']:.2%} | "
            f"ext {extraction_rate:.0%} | t {log['wall_s']:.0f}s",
            flush=True,
        )
        if not args.no_wandb:
            wandb.log(log, step=step)

        # ---- save LoRA periodically ----
        if (step + 1) % args.save_every == 0:
            out_dir = save_dir / f"iter_{step + 1:06d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(str(out_dir))
            print(f"[save] LoRA → {out_dir}")

    print("done.")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
