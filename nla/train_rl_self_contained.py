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
            # Explicitly override sampler — Qwen3's generation_config.json may
            # set top_p<1.0 / top_k>0 / repetition_penalty by default. Any of
            # these introduce -inf in the sampler logits, breaking the
            # importance ratio (exp(new_lp - old_lp) → inf).
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
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


def critic_predict(critic, input_ids, attention_mask, mse_scale_f):
    """Forward the critic and produce a per-sample prediction vector.

    Architecture tweak (vs upstream NLACriticModel.forward):
      pred = value_head(normalize(backbone_last_hidden, mse_scale))

    The upstream forward does value_head(backbone_last_hidden) directly,
    which leaves value_head's input norm unbounded. With bf16+Adam on a
    near-identity value_head, that's exactly the path that NaN'd AR SFT
    8+ times. Normalising the backbone-last-hidden BEFORE the value_head
    bounds the input to a fixed norm (mse_scale), so a tiny weight update
    can't blow up the output norm by 100×. At identity init the two
    formulas agree (after the loss's final normalize), so swapping is
    backward-compatible with AR-SFT checkpoints.

    Returns: [B, d_model] fp32 pred tensor. Caller is responsible for
    grad / no_grad context; this function does not toggle.
    """
    cout = critic(input_ids=input_ids, attention_mask=attention_mask)
    backbone_last = cout.backbone_last_hidden  # [B, T, D] (bf16)
    if attention_mask is not None:
        last_idx = attention_mask.sum(dim=1) - 1
    else:
        last_idx = torch.full(
            (input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device,
        )
    bs = input_ids.shape[0]
    last_h = backbone_last[
        torch.arange(bs, device=input_ids.device), last_idx
    ].float()  # [B, D]
    last_h_norm = normalize_activation(last_h, mse_scale_f)
    pred = critic.value_head(
        last_h_norm.to(critic.value_head.weight.dtype)
    ).float()
    return pred


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
            pred = critic_predict(critic, x, None, mse_scale_f)[0]  # [d]
        gold = act.to(device).float()
        pred_n = normalize_activation(pred.unsqueeze(0), mse_scale_f)[0]
        gold_n = normalize_activation(gold.unsqueeze(0), mse_scale_f)[0]
        mse = F.mse_loss(pred_n, gold_n).item()
        if not math.isfinite(mse):
            rewards.append(None)
            continue
        rewards.append(-mse)
    return rewards


def grpo_update_microbatched(
    actor, optim, tokenizer, full_ids_list, prompt_lens, activations,
    old_logps_list, advantages, vectors_ref, device,
    micro_batch=2, clip_eps=0.2, kl_beta=0.04, max_grad_norm=1.0,
):
    """Fused micro-batched forward+loss+backward for GRPO.

    Each micro-batch: forward (LoRA on, grad) → ref forward (LoRA off, no grad)
    → per-chunk GRPO loss → backward → release graph → next chunk. Single
    optim.step() at the end. Peak memory = one micro-batch graph instead of
    N retained graphs (which is what OOMs at B*G=256).

    Returns (mean_loss, grad_norm, metrics_dict).
    """
    optim.zero_grad()
    n = len(full_ids_list)
    sample_losses_log = []
    sample_kls_log = []
    sample_clipfrac_log = []
    advantages = advantages.detach()  # no grad through advantage
    for cs in range(0, n, micro_batch):
        idxs = list(range(cs, min(cs + micro_batch, n)))
        bs = len(idxs)
        max_len = max(full_ids_list[i].numel() for i in idxs)
        pad_id = tokenizer.eos_token_id
        batch_ids = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
        for row, i in enumerate(idxs):
            L = full_ids_list[i].numel()
            batch_ids[row, :L] = full_ids_list[i].to(device)
            attn[row, :L] = 1
        v_batch = torch.stack(
            [activations[i].to(device).float() for i in idxs], dim=0,
        )
        # --- new_logp (with grad) ---
        vectors_ref[0] = v_batch
        try:
            new_logits = actor(input_ids=batch_ids, attention_mask=attn).logits
        finally:
            vectors_ref[0] = None
        new_logp = F.log_softmax(new_logits.float(), dim=-1)
        # --- ref_logp (no grad, LoRA off) ---
        vectors_ref[0] = v_batch
        try:
            with torch.no_grad(), actor.disable_adapter():
                ref_logits = actor(input_ids=batch_ids, attention_mask=attn).logits
        finally:
            vectors_ref[0] = None
        ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
        del ref_logits
        # --- per-sample GRPO loss for this chunk ---
        chunk_losses = []
        for row, i in enumerate(idxs):
            L = full_ids_list[i].numel()
            p_len = prompt_lens[i]
            if L <= p_len:
                continue
            target_ids = batch_ids[row, p_len:L]
            pred_idx = torch.arange(p_len - 1, L - 1, device=device)
            new_lp = (
                new_logp[row].index_select(0, pred_idx)
                .gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            )
            ref_lp = (
                ref_logp[row].index_select(0, pred_idx)
                .gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                .detach()
            )
            old_lp = old_logps_list[i].to(device).detach()
            if new_lp.numel() == 0 or old_lp.numel() != new_lp.numel():
                continue
            ratio = torch.exp(new_lp - old_lp)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            A = advantages[i]
            surrogate = torch.minimum(ratio * A, clipped * A)
            delta = ref_lp - new_lp
            kl = torch.exp(delta) - delta - 1.0
            per_tok = -(surrogate - kl_beta * kl)
            sample_loss = per_tok.mean()
            chunk_losses.append(sample_loss)
            sample_kls_log.append(kl.detach().mean().item())
            sample_clipfrac_log.append(
                ((ratio < 1 - clip_eps) | (ratio > 1 + clip_eps)).float().mean().item()
            )
        # Free retained logp / logits before backward to bound peak.
        del new_logits, ref_logp
        if not chunk_losses:
            del new_logp
            continue
        # Scale so summed chunk losses give batch-mean.
        chunk_loss = torch.stack(chunk_losses).sum() / n
        chunk_loss.backward()
        sample_losses_log.append(chunk_loss.item() * n / len(chunk_losses))
        del new_logp
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in actor.parameters() if p.requires_grad], max_grad_norm,
    )
    optim.step()
    metrics = {
        "kl_mean": float(np.mean(sample_kls_log)) if sample_kls_log else 0.0,
        "clip_frac": float(np.mean(sample_clipfrac_log)) if sample_clipfrac_log else 0.0,
    }
    mean_loss = float(np.mean(sample_losses_log)) if sample_losses_log else 0.0
    gn = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
    return mean_loss, gn, metrics


def compute_token_logps(
    actor, tokenizer, full_ids_list, prompt_lens, activations, vectors_ref,
    device, micro_batch=2, use_ref=False,
):
    """[LEGACY — kept for reference] Compute per-token log P(response_t | prefix_<t).

    Returns: list of 1-D tensors. Memory issue: each returned tensor retains
    its forward graph; with N chunks, retained activations = N × per-chunk.
    Use grpo_update_microbatched() instead, which does forward+loss+backward
    per chunk and releases each graph before the next.
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
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--use-rslora", action="store_true", default=True,
                   help="Use rsLoRA scaling (alpha/sqrt(r) instead of alpha/r). "
                        "Default ON because we use r=128 where vanilla LoRA's "
                        "alpha/r=0.125 collapses the effective learning rate.")
    p.add_argument("--train-critic", action="store_true", default=False,
                   help="Co-train the AR critic (paper-faithful). Adds a "
                        "separate optimizer for the critic and supervised MSE "
                        "loss on (explanation, gold_activation) pairs each step.")
    p.add_argument("--critic-lr", type=float, default=1e-5)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False,
                   help="Recompute activations during backward (saves ~50% "
                        "activation memory at ~30%% compute cost). Off by "
                        "default — 8-bit Adam on critic gives bigger savings.")
    p.add_argument("--critic-micro-batch", type=int, default=4,
                   help="Micro-batch size for the critic's training-time forward. "
                        "Single full-batch forward OOMs at B*G=256.")
    p.add_argument("--logp-micro-batch", type=int, default=2)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume-from-lora", type=str, default=None,
                   help="Directory containing a saved LoRA adapter (iter_NNNNNN); "
                        "loaded onto the AV-SFT base so training continues "
                        "from those weights.")
    p.add_argument("--start-step", type=int, default=0,
                   help="Initial step counter — useful when resuming so wandb "
                        "x-axis lines up with the previous run.")
    p.add_argument("--eval-every", type=int, default=10,
                   help="Run a held-out qualitative eval every N steps. "
                        "Logs explanation texts to wandb Table; 0 disables.")
    p.add_argument("--eval-n-prompts", type=int, default=20,
                   help="Number of fixed held-out prompts for per-step eval.")
    p.add_argument("--eval-skip-rows", type=int, default=30000,
                   help="Take eval prompts from rl_shuf rows starting here "
                        "(past --max-rows training cursor).")
    p.add_argument("--max-rows", type=int, default=None,
                   help="cap rows from rl parquet (avoids 3.7GB full-load for smoke runs)")
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-beta", type=float, default=0.04)
    p.add_argument("--wandb-project", default="nla-qwen3-8b")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--external-evals", default="",
        help="Comma-sep list of evals/ IDs to run every --eval-every step "
             "(e.g. 'hallucination,karvonen_confusion'). Empty = none.",
    )
    p.add_argument("--eval-n-hallucination", type=int, default=40,
                   help="N held-out prompts for hallucination eval.")
    p.add_argument("--eval-n-karvonen", type=int, default=97,
                   help="N records (out of 97 filtered) for karvonen_confusion eval.")
    p.add_argument("--judge-key-env", default="ANTHROPIC_API_KEY_FALLBACK",
                   help="Env var holding the judge API key (default: high-prio).")
    p.add_argument("--judge-concurrency", type=int, default=32,
                   help="Parallel judge calls (Anthropic sync, NOT batch API).")
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
        use_rslora=args.use_rslora,
    )
    # CRITICAL for LoRA + gradient_checkpointing: the base model has no
    # requires_grad params, so gradient_checkpointing's input-grad check
    # fails ("element 0 of tensors does not require grad"). This hook
    # forces input embeddings to require grad, propagating grad to LoRA.
    if args.gradient_checkpointing:
        actor.enable_input_require_grads()
    if args.resume_from_lora is not None:
        # Resume: load a previously-saved LoRA adapter onto the base.
        # peft.PeftModel.from_pretrained handles attaching the LoRA layers and
        # copying weights. Skips the get_peft_model wrapping flow.
        from peft import PeftModel
        print(f"[actor] RESUMING from LoRA {args.resume_from_lora}")
        actor = PeftModel.from_pretrained(actor, args.resume_from_lora, is_trainable=True)
        # Sanity check we got non-zero LoRA weights (not a freshly-init adapter)
        _lora_norm = 0.0
        for n, p_ in actor.named_parameters():
            if "lora_" in n:
                _lora_norm += p_.detach().float().pow(2).sum().item()
        print(f"[actor] resumed; sum(lora_param²) = {_lora_norm:.2e}")
    else:
        actor = get_peft_model(actor, lora_cfg)
    actor.print_trainable_parameters()
    actor.train()
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable()
        # PEFT wraps the model; the inner module's gradient_checkpointing flag
        # must be set explicitly or HF silently no-ops.
        if hasattr(actor, "base_model"):
            inner = actor.base_model
            while hasattr(inner, "model"):
                inner = inner.model
                if hasattr(inner, "gradient_checkpointing"):
                    inner.gradient_checkpointing = True
        # NOTE: do NOT set config.use_cache=False globally — that breaks
        # generate() in rollout (autoregressive without KV cache is O(T²)).
        # HF auto-disables use_cache per-forward when gradient_checkpointing
        # fires AND there are gradients; rollout (.eval() + no_grad) is unaffected.
        print(f"[actor] gradient_checkpointing ENABLED")

    # ---- critic (frozen or co-trained) ----
    print(f"[critic] loading {args.ar_ckpt}")
    critic = NLACriticModel.from_pretrained(
        args.ar_ckpt, torch_dtype=torch.bfloat16,
    ).to(device)
    # NLACriticModel.from_pretrained returns params with requires_grad=True by
    # default. Freeze everything first, then conditionally unfreeze backbone.
    for p_ in critic.parameters():
        p_.requires_grad_(False)
    critic_optim = None
    if args.train_critic:
        # Per paper §RL training: AR is co-trained simultaneously with AV on
        # the SAME explanations the actor produces this step. Loss = MSE against
        # the gold activation, normalised. AR's gradient does NOT flow back into
        # the actor (the explanation tokens are discrete — gradient stops there
        # automatically). Both backbone AND value_head train; the bf16+Adam
        # blow-up that NaN'd AR SFT is now neutralised by critic_predict's
        # normalize-before-value_head trick (bounds value_head input norm).
        for p_ in critic.backbone.parameters():
            p_.requires_grad_(True)
        for p_ in critic.value_head.parameters():
            p_.requires_grad_(True)
        critic_trainable = [p for p in critic.parameters() if p.requires_grad]
        try:
            import bitsandbytes as _bnb
            critic_optim = _bnb.optim.AdamW8bit(
                critic_trainable, lr=args.critic_lr, betas=(0.9, 0.95),
                weight_decay=0.0,
            )
        except ImportError:
            critic_optim = torch.optim.AdamW(
                critic_trainable, lr=args.critic_lr, betas=(0.9, 0.95),
                weight_decay=0.0,
            )
        n_trainable = sum(p.numel() for p in critic_trainable)
        print(f"[critic] CO-TRAINED, lr={args.critic_lr}, "
              f"trainable={n_trainable/1e9:.2f}B (backbone + value_head)")
    else:
        print(f"[critic] FROZEN (eval-only scorer)")
    critic.eval()  # Qwen3 has no dropout — eval mode is fine for both grad/no-grad
    print(f"[critic] value_head shape={tuple(critic.value_head.weight.shape)}")

    # ---- karvonen hook on actor ----
    vectors_ref = [None]
    _register_karvonen_hook(actor, vectors_ref, inj_id, left_id, right_id, layer_idx=1)

    # ---- dataset ----
    print(f"[data] loading {args.rl_parquet} (max_rows={args.max_rows})", flush=True)
    rows = load_rl_dataset(args.rl_parquet, n_max=args.max_rows)
    print(f"[data] {len(rows)} rows", flush=True)

    # ---- FVE baseline: predict-the-mean MSE on this dataset ----
    # FVE = 1 - mse_actual / baseline_mse. baseline = MSE between
    # normalize(μ) and normalize(v_i), where μ = mean of activations.
    # 0% = no better than constant prediction; 100% = perfect reconstruction.
    # Paper's Qwen2.5-7B critic-SL alone hit FVE ≈ 37.5%.
    _act_stack = torch.tensor(
        [r["activation"] for r in rows[: min(len(rows), 4000)]],
        dtype=torch.float32,
    )
    _mu = _act_stack.mean(dim=0, keepdim=True)
    _mu_n = normalize_activation(_mu, mse_scale_f)
    _act_n = normalize_activation(_act_stack, mse_scale_f)
    fve_baseline = ((_mu_n - _act_n) ** 2).mean(dim=-1).mean().item()
    del _act_stack, _act_n, _mu, _mu_n
    print(f"[fve] predict-the-mean baseline mse_nrm = {fve_baseline:.4f}", flush=True)

    # ---- optimizer ----
    # 8-bit Adam (bitsandbytes) for both actor LoRA and critic — block-wise
    # int8 quantization of (m, v) state cuts optimizer memory ~4×. "Paged"
    # variant CPU-offloads pages under memory pressure. Standard choice for
    # memory-constrained LLM fine-tuning; numerically equivalent to fp32 Adam
    # within bf16 noise for our use case.
    try:
        import bitsandbytes as bnb
        _adam_cls = bnb.optim.AdamW8bit
        print(f"[optim] using bitsandbytes AdamW8bit (bnb {bnb.__version__})")
    except ImportError:
        _adam_cls = torch.optim.AdamW
        print(f"[optim] bitsandbytes unavailable, falling back to torch AdamW (fp32 m,v)")
    trainable = [p for p in actor.parameters() if p.requires_grad]
    optim = _adam_cls(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # ---- wandb ----
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pending_idxs = list(range(len(rows)))
    rng.shuffle(pending_idxs)
    cursor = 0

    # ---- Fixed held-out eval prompts, DOC-DISJOINT from training rows.
    # Stage-1 only guarantees disjointness BETWEEN av_sft/ar_sft/rl FILES;
    # within rl_shuf.parquet, rows past --eval-skip-rows can share doc_id
    # with rows before it (the file is row-shuffled, not doc-partitioned
    # internally). Without explicit filtering we measured ~50% doc-overlap.
    # Fix: scan training-window (rows 0..eval_skip_rows) to collect doc_ids,
    # then take eval rows past the cursor whose doc_id is NOT in that set.
    eval_rows = []
    if args.eval_every > 0 and args.eval_n_prompts > 0:
        import pyarrow.parquet as _pq
        _pf = _pq.ParquetFile(args.rl_parquet)
        # Pass 1: training-window doc_ids
        _train_doc_ids: set = set()
        _seen = 0
        for _rg_idx in range(_pf.num_row_groups):
            if _seen >= args.eval_skip_rows:
                break
            _rg = _pf.read_row_group(_rg_idx, columns=["doc_id"])
            _ids = _rg.column("doc_id").to_pylist()
            _nrg = len(_ids)
            _take = min(_nrg, args.eval_skip_rows - _seen)
            _train_doc_ids.update(_ids[:_take])
            _seen += _nrg
        # Pass 2: doc-disjoint rows past the cursor
        _seen = 0
        for _rg_idx in range(_pf.num_row_groups):
            if len(eval_rows) >= args.eval_n_prompts:
                break
            _rg = _pf.read_row_group(
                _rg_idx, columns=["prompt", "activation_vector", "doc_id"],
            )
            _n = _rg.num_rows
            if _seen + _n <= args.eval_skip_rows:
                _seen += _n
                continue
            _start = max(0, args.eval_skip_rows - _seen)
            _prompts = _rg.column("prompt").to_pylist()
            _acts = _rg.column("activation_vector").to_pylist()
            _dids = _rg.column("doc_id").to_pylist()
            for _i in range(_start, _n):
                if _dids[_i] in _train_doc_ids:
                    continue
                eval_rows.append({"prompt": _prompts[_i], "activation": _acts[_i]})
                if len(eval_rows) >= args.eval_n_prompts:
                    break
            _seen += _n
        print(f"[eval] {len(eval_rows)} doc-disjoint prompts loaded "
              f"(rows past {args.eval_skip_rows}, excluding "
              f"{len(_train_doc_ids)} training doc_ids)", flush=True)
    eval_table_data = []  # accumulates [step, idx, reward, fve, extracted, explanation]

    # ---- External evals (evals/ pluggable, run every --eval-every step) ----
    # Reuses the trainer's `vectors_ref` so each eval shares the injection hook
    # already attached to the actor (no stacked hooks, no extra Qwen3-8B load).
    # Judge calls go through ANTHROPIC_API_KEY_FALLBACK (high-prio) in a
    # ThreadPoolExecutor — see evals/base.py.
    external_evals = []
    if args.external_evals.strip():
        if not os.environ.get(args.judge_key_env):
            raise RuntimeError(
                f"--external-evals requires ${args.judge_key_env} to be set "
                f"(judge API key for Sonnet 4.6). See CLAUDE.md."
            )
        # Side-effect imports register the eval classes via @register decorator.
        import evals.hallucination  # noqa: F401
        import evals.karvonen_confusion  # noqa: F401
        from evals.base import EvalConfig
        from evals.registry import get_eval

        n_samples_for = {
            "hallucination": args.eval_n_hallucination,
            "karvonen_confusion": args.eval_n_karvonen,
        }
        for eid in [s.strip() for s in args.external_evals.split(",") if s.strip()]:
            ev_cfg = EvalConfig(
                output_dir=save_dir / "eval_runs",
                n_samples=n_samples_for.get(eid, 40),
                seed=args.seed,
                eval_skip_rows=args.eval_skip_rows,
                parquet_path=args.rl_parquet,
                judge_model="claude-sonnet-4-6",
                judge_temperature=0.0,
                judge_max_concurrency=args.judge_concurrency,
                anthropic_api_key_env=args.judge_key_env,
            )
            ev_cls = get_eval(eid)
            ev = ev_cls(ev_cfg)
            ev.setup(actor, critic, tokenizer, cfg, device,
                     shared_vectors_ref=vectors_ref)
            external_evals.append(ev)
            print(f"[external eval] {eid} ready (n={ev_cfg.n_samples}, "
                  f"judge={ev_cfg.judge_model}, key=${args.judge_key_env})",
                  flush=True)

    # History for the multi-line Pareto chart.
    pareto_history: list[dict] = []

    for step in range(args.start_step, args.num_steps):
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

        # ---- GRPO update: fused forward+loss+backward per micro-batch ----
        # Previous code did all forwards then all backwards, which retained
        # every micro-batch's compute graph and OOM'd at B*G=256. The fused
        # version releases each chunk's graph before starting the next.
        actor.train()
        mean_loss_val, grad_norm_val, grpo_metrics = grpo_update_microbatched(
            actor, optim, tokenizer,
            all_full_ids, all_prompt_lens, all_activations,
            all_old_logps, adv, vectors_ref, device,
            micro_batch=args.logp_micro_batch,
            clip_eps=args.clip_eps, kl_beta=args.kl_beta,
            max_grad_norm=args.max_grad_norm,
        )
        # Build a scalar-tensor stand-in for the existing logging path that
        # expects a `loss` tensor with .item().
        loss = torch.tensor(mean_loss_val, device=device)
        grad_norm = torch.tensor(grad_norm_val, device=device)
        if not math.isfinite(mean_loss_val):
            print(
                f"step {step}: loss={mean_loss_val} non-finite "
                f"(kl={grpo_metrics.get('kl_mean')}, "
                f"clip_frac={grpo_metrics.get('clip_frac')}). Skipping critic update.",
                flush=True,
            )
            # actor optimizer.step was already called inside the helper, but if
            # the loss was nan/inf, gradients were nan/inf too — Adam may have
            # corrupted weights. Best we can do is continue; safe-guard via the
            # grad-norm clip already applied.
            continue

        # ---- AR critic co-training (paper-faithful, optional) ----
        # Per paper §RL: "Update the AR by one step of gradient descent on the
        # regression loss ||h_l − AR_θ(z)||²_2". Inputs z = the explanations the
        # actor just produced this step; targets h_l = the gold activations.
        # Gradient from this update does NOT flow into the actor (z is discrete).
        critic_loss_val = float("nan")
        critic_grad_norm_val = float("nan")
        if args.train_critic and critic_optim is not None:
            crit_inputs = []
            crit_golds = []
            for expl, act in zip(all_explanations, all_activations):
                if expl is None:
                    continue
                text = template.format(explanation=expl)
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > 1024 or len(ids) == 0:
                    continue
                crit_inputs.append(torch.tensor(ids, dtype=torch.long))
                crit_golds.append(act)
            if crit_inputs:
                # Micro-batch the critic update — single forward on 256 sequences
                # × 200 tokens × 5.5B-param critic with grad blows past 130GB.
                # Accumulate gradient across micro-batches, single step at the
                # end (loss is divided by total bs so it averages correctly).
                bs_total = len(crit_inputs)
                pad_id = tokenizer.eos_token_id
                critic_optim.zero_grad()
                accumulated = 0.0
                finite = True
                cmb = max(1, args.critic_micro_batch)
                for cs in range(0, bs_total, cmb):
                    chunk = list(range(cs, min(cs + cmb, bs_total)))
                    max_len = max(crit_inputs[i].numel() for i in chunk)
                    bs = len(chunk)
                    batch_ids = torch.full(
                        (bs, max_len), pad_id, dtype=torch.long, device=device,
                    )
                    attn = torch.zeros((bs, max_len), dtype=torch.long, device=device)
                    for row, i in enumerate(chunk):
                        L = crit_inputs[i].numel()
                        batch_ids[row, :L] = crit_inputs[i].to(device)
                        attn[row, :L] = 1
                    pred = critic_predict(critic, batch_ids, attn, mse_scale_f)
                    gold = torch.stack([crit_golds[i] for i in chunk]).to(device).float()
                    pred_n = normalize_activation(pred, mse_scale_f)
                    gold_n = normalize_activation(gold, mse_scale_f)
                    # Scale so the sum across micro-batches = MSE over full batch.
                    chunk_loss = F.mse_loss(pred_n, gold_n) * (bs / bs_total)
                    if not torch.isfinite(chunk_loss):
                        print(f"step {step}: critic loss non-finite (chunk {cs}), skipping", flush=True)
                        finite = False
                        break
                    chunk_loss.backward()
                    accumulated += chunk_loss.item()
                if finite:
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        critic_trainable, args.max_grad_norm,
                    )
                    critic_optim.step()
                    critic_loss_val = accumulated  # already the full-batch mean
                    critic_grad_norm_val = (
                        critic_grad_norm.item()
                        if hasattr(critic_grad_norm, "item")
                        else float(critic_grad_norm)
                    )

        # ---- logging ----
        valid_rewards = [r for r in rewards if r is not None]
        n_valid = len(valid_rewards)
        n_total = len(rewards)
        extraction_rate = n_valid / n_total if n_total else 0
        mean_cjk = (
            sum(cjk_fraction(t) for t in all_response_text) / max(len(all_response_text), 1)
        )
        # Response lengths come from the rollout's old_logps (one entry per sample).
        n_resps_t = torch.tensor(
            [lp.numel() for lp in all_old_logps], dtype=torch.float32, device=device,
        )
        # FVE on valid (non-extraction-failed) samples — gives an
        # interpretable curve in wandb that maps to paper's reported numbers.
        # Use valid rewards only so extraction failures don't bias FVE down.
        fve = (
            1.0 - (-float(np.mean(valid_rewards))) / fve_baseline
            if valid_rewards else float("nan")
        )
        log = {
            "step": step,
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "reward_mean": float(np.mean(valid_rewards)) if valid_rewards else float("nan"),
            "reward_std": float(np.std(valid_rewards)) if valid_rewards else float("nan"),
            "reward_min": float(np.min(valid_rewards)) if valid_rewards else float("nan"),
            "reward_max": float(np.max(valid_rewards)) if valid_rewards else float("nan"),
            "fve": fve,
            "fve_pct": fve * 100.0,
            "fve_baseline": fve_baseline,
            "advantage_mean": adv.mean().item(),
            "advantage_std": adv.std().item(),
            "extraction_rate": extraction_rate,
            "mean_cjk": mean_cjk,
            "mean_resp_len": n_resps_t.mean().item(),
            "kl_mean": grpo_metrics.get("kl_mean", 0.0),
            "clip_frac": grpo_metrics.get("clip_frac", 0.0),
            "critic_loss": critic_loss_val,
            "critic_grad_norm": critic_grad_norm_val,
            "wall_s": time.time() - t0,
        }
        crit_str = (
            f"| crit {critic_loss_val:.4f} " if args.train_critic else ""
        )
        print(
            f"step {step:04d} | loss {log['loss']:.4f} | r {log['reward_mean']:.3f} "
            f"| FVE {log['fve_pct']:.1f}% {crit_str}| kl {log['kl_mean']:.4f} | "
            f"clip {log['clip_frac']:.2%} | ext {extraction_rate:.0%} | "
            f"t {log['wall_s']:.0f}s",
            flush=True,
        )

        # ---- per-step eval: every N steps, run actor (current weights) on a
        # FIXED set of held-out prompts and log explanations as a wandb Table.
        # Lets you scrub through the run and watch explanations evolve.
        if args.eval_every > 0 and step % args.eval_every == 0:
            actor.eval()
            eval_rewards_s = []
            eval_records = []
            for ei, row in enumerate(eval_rows):
                prompt_text = build_prompt_text(row["prompt"], inject_char, tokenizer)
                activation = torch.tensor(row["activation"], dtype=torch.float32)
                ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                pt = torch.tensor([ids], dtype=torch.long, device=device)
                vectors_ref[0] = activation.unsqueeze(0).to(device).float()
                try:
                    with torch.no_grad():
                        gen = actor.generate(
                            input_ids=pt, attention_mask=torch.ones_like(pt),
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True, temperature=1.0,
                            top_p=1.0, top_k=0, repetition_penalty=1.0,
                            pad_token_id=tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                        )
                finally:
                    vectors_ref[0] = None
                resp = tokenizer.decode(
                    gen.sequences[0, pt.shape[1]:], skip_special_tokens=True,
                )
                expl = extract_explanation(resp)
                e_reward = -2.0
                if expl is not None:
                    ctext = template.format(explanation=expl)
                    cids = tokenizer.encode(ctext, add_special_tokens=False)
                    if 0 < len(cids) <= 1024:
                        x = torch.tensor([cids], dtype=torch.long, device=device)
                        with torch.no_grad():
                            pred = critic_predict(critic, x, None, mse_scale_f)[0]
                        gold = activation.to(device).float()
                        pn = normalize_activation(pred.unsqueeze(0), mse_scale_f)[0]
                        gn = normalize_activation(gold.unsqueeze(0), mse_scale_f)[0]
                        mse = F.mse_loss(pn, gn).item()
                        if math.isfinite(mse):
                            e_reward = -mse
                eval_rewards_s.append(e_reward)
                eval_records.append({
                    "step": step, "idx": ei, "reward": e_reward,
                    "fve": (1.0 - (-e_reward) / fve_baseline) if e_reward > -2.0 else float("nan"),
                    "extracted": expl is not None,
                    "explanation": expl if expl is not None else "<extraction failed>",
                })
            # Aggregate eval scalars
            valid_e = [r for r in eval_rewards_s if r > -2.0]
            log["eval/reward_mean"] = (
                float(np.mean(eval_rewards_s)) if eval_rewards_s else float("nan")
            )
            log["eval/reward_mean_valid"] = (
                float(np.mean(valid_e)) if valid_e else float("nan")
            )
            log["eval/fve_pct"] = (
                (1.0 - (-float(np.mean(valid_e))) / fve_baseline) * 100.0
                if valid_e else float("nan")
            )
            log["eval/extraction_rate"] = (
                sum(1 for r in eval_records if r["extracted"]) / len(eval_records)
                if eval_records else 0.0
            )
            # Persistent table — accumulates across the whole run.
            for r in eval_records:
                eval_table_data.append([
                    r["step"], r["idx"], r["reward"], r["fve"],
                    r["extracted"], r["explanation"][:500],
                ])
            if not args.no_wandb:
                log["eval/samples"] = wandb.Table(
                    columns=["step", "idx", "reward", "fve", "extracted", "explanation"],
                    data=list(eval_table_data),
                )
            print(
                f"  [eval@{step}] reward {log['eval/reward_mean']:.3f} "
                f"| FVE {log['eval/fve_pct']:.1f}% "
                f"| ext {log['eval/extraction_rate']:.0%}",
                flush=True,
            )
            # Print 3 sample explanations so the log itself shows how outputs
            # evolve. Pick indices 0, 7, 14 — spread across the eval set.
            for _ei in (0, 7, 14):
                if _ei < len(eval_records):
                    _r = eval_records[_ei]
                    _expl = _r["explanation"][:200].replace("\n", " ")
                    print(
                        f"    [eval@{step} idx={_ei} r={_r['reward']:.3f}] {_expl}",
                        flush=True,
                    )

            # ---- External evals (hallucination, karvonen_confusion, …) ----
            # Each eval reuses `actor` + `vectors_ref`, so no extra GPU load.
            # Judge calls are parallelised inside each eval's evaluate().
            for ev in external_evals:
                try:
                    result = ev.evaluate(step)
                    for k, v in result.metrics.items():
                        log[f"eval/{ev.id}/{k}"] = v
                    if not args.no_wandb and result.table_rows:
                        cols = list(result.table_rows[0].keys())
                        log[f"eval/{ev.id}/rollouts"] = wandb.Table(
                            columns=cols,
                            data=[[r[c] for c in cols] for r in result.table_rows],
                        )
                except Exception as _ev_err:
                    print(f"[external eval {ev.id}@{step}] FAILED: "
                          f"{type(_ev_err).__name__}: {_ev_err}", flush=True)

            # ---- Pareto chart: FVE (capability) vs hallucinations (faithful-
            # ness) vs Karvonen captures-quirk (depth) on a shared X-axis.
            # Tells you at a glance whether RL is buying capability at the
            # cost of faithfulness or whether it's Pareto-improving.
            pareto_history.append({
                "step": step,
                "fve_pct": float(log.get("eval/fve_pct", float("nan"))),
                "halluc_x10": float(log.get(
                    "eval/hallucination/hallucinations_mean", float("nan"))) * 10.0,
                "captures_quirk_pct": float(log.get(
                    "eval/karvonen_confusion/captures_quirk_rate", float("nan"))) * 100.0,
                "clean_rate_pct": float(log.get(
                    "eval/hallucination/clean_rate", float("nan"))) * 100.0,
            })
            if not args.no_wandb and len(pareto_history) >= 2:
                _xs = [h["step"] for h in pareto_history]
                log["pareto/capability_vs_faithfulness"] = wandb.plot.line_series(
                    xs=_xs,
                    ys=[
                        [h["fve_pct"] for h in pareto_history],
                        [h["halluc_x10"] for h in pareto_history],
                        [h["captures_quirk_pct"] for h in pareto_history],
                        [h["clean_rate_pct"] for h in pareto_history],
                    ],
                    keys=["FVE %", "hallucinations × 10",
                          "captures_quirk %", "clean_rate %"],
                    title="Capability vs faithfulness (shared X = step)",
                    xname="step",
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
