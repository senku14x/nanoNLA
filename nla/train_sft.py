"""Self-contained NLA SFT (AV + AR), no Miles dependency.

Single entry point with `--mode {av,ar}`:
  - AV: AutoModelForCausalLM + Karvonen layer-1 injection hook
        loss = cross-entropy on response tokens only
        target: actor learns to verbalise injected activations
  - AR: NLACriticModel (truncated K+1-layer backbone + Linear(d,d) value_head)
        loss = MSE on L2-normalised (pred, gold) at last-token position
        target: critic learns to reconstruct activation from explanation text

Replaces:
  - nla/train_actor.py (NLAFSDPActor — Miles FSDP subclass)
  - nla/loss.py (nla_critic_loss, plugged in via Miles --custom-loss-function-path)
  - nla/rollout/sft_actor.py, nla/rollout/sft_critic.py (Miles rollout adapters)
  - configs/actor_sft.sh, configs/critic_sft.sh (shell wrappers for Miles train.py)
  - nla/scripts/prepare_critic_checkpoint.py (truncation now happens in-script for AR)

Loads bf16 model + bitsandbytes AdamW8bit (~4 GB optim states on 8B model
instead of 64 GB for fp32 AdamW). Single GPU; activation memory bounded by
gradient_checkpointing on the AV path.

Saves HF format checkpoints directly — no DCP→HF conversion step.
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
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.injection import karvonen_inject_in_residual
from nla.models import NLACriticModel
from nla.schema import (
    INJECT_PLACEHOLDER,
    EXPLANATION_RE,
    normalize_activation,
    resolve_target_scale,
)


# ----------------------------------------------------------------------------
# Helpers shared with train_rl_self_contained.py (kept inline so this file
# stays self-contained — they're small, and importing creates an awkward
# coupling between SFT and RL trainers).
# ----------------------------------------------------------------------------


def cjk_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if "CJK" in unicodedata.name(c, "")) / len(text)


def _register_karvonen_hook(model, vectors_ref, inj_id, left_id, right_id, layer_idx=1):
    """Register an embed-token-id capture + layer-1 residual-modification hook.

    On every forward: the embedding hook stashes input_ids; the layer-1 hook
    reads them, finds marker positions, and adds the norm-matched activation
    vector (from vectors_ref[0]) onto the residual at those positions.

    No-op when seq_len < 2 (autoregressive cache steps after rollout's prefill).
    """
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
    target = model
    while hasattr(target, "model") and not hasattr(target, "layers"):
        target = target.model
    target.layers[layer_idx].register_forward_hook(layer_hook)


def critic_predict(critic, input_ids, attention_mask, mse_scale_f):
    """pred = value_head(normalize(backbone_last_hidden, mse_scale)).

    Same trick as train_rl_self_contained.py — bounds value_head's input norm
    so its weight updates can't blow up the output norm by 100× (which is what
    NaN'd AR SFT 8+ times before NLA_FREEZE_VALUE_HEAD=1 in the Miles path).
    At identity init, equivalent to the paper's direct value_head(backbone_last).
    """
    cout = critic(input_ids=input_ids, attention_mask=attention_mask)
    backbone_last = cout.backbone_last_hidden
    if attention_mask is not None:
        last_idx = attention_mask.sum(dim=1) - 1
    else:
        last_idx = torch.full(
            (input_ids.shape[0],), input_ids.shape[1] - 1, device=input_ids.device,
        )
    bs = input_ids.shape[0]
    last_h = backbone_last[
        torch.arange(bs, device=input_ids.device), last_idx
    ].float()
    last_h_norm = normalize_activation(last_h, mse_scale_f)
    pred = critic.value_head(
        last_h_norm.to(critic.value_head.weight.dtype)
    ).float()
    return pred


def load_sft_dataset(parquet_path, n_max=None, *, mode):
    """Stream-load AV (prompt: list[dict], response: str, activation_vector)
    or AR (prompt: str, activation_vector). Slice rowgroups so n_max=N takes
    only N rows, not the full first rowgroup."""
    cols = (
        ["prompt", "response", "activation_vector"] if mode == "av"
        else ["prompt", "activation_vector"]
    )
    pf = pq.ParquetFile(parquet_path)
    rows = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        n_in_rg = rg.num_rows
        take = n_in_rg if n_max is None else min(n_max - len(rows), n_in_rg)
        rg = rg.slice(0, take)
        rg_cols = {c: rg.column(c).to_pylist() for c in cols}
        for i in range(take):
            row = {c: rg_cols[c][i] for c in cols}
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# AR critic init: truncate base Qwen3 to K+1 layers + Linear(d, d) value_head,
# identity-init the head. Replaces nla/scripts/prepare_critic_checkpoint.py.
# ----------------------------------------------------------------------------

def init_critic_from_base(base_ckpt: str, num_layers: int, dtype):
    """Truncate base to first `num_layers` transformer blocks, attach an
    identity-init Linear(d, d) value_head. NLACriticModel handles the wrapping.

    identity-init is critical: at step 0, pred = value_head(last_h) = last_h
    when value_head = I, so the initial reconstruction loss starts at the
    backbone's own representational ceiling instead of `kaiming_uniform`'s
    1/√3 scaling which would crush pred_norm. See TRAINING_NOTES.md.
    """
    # First load the full base, truncate the layers list, then construct
    # NLACriticModel around it.
    from copy import deepcopy
    base = AutoModelForCausalLM.from_pretrained(
        base_ckpt, torch_dtype=dtype, attn_implementation="sdpa",
    )
    cfg = deepcopy(base.config)
    cfg.num_hidden_layers = num_layers
    if hasattr(cfg, "layer_types") and cfg.layer_types is not None:
        cfg.layer_types = list(cfg.layer_types)[:num_layers]
    # Walk into the inner module to get .layers
    inner = base
    while hasattr(inner, "model") and not hasattr(inner, "layers"):
        inner = inner.model
    # Keep only the first num_layers blocks
    inner.layers = torch.nn.ModuleList(list(inner.layers)[:num_layers])
    d_model = cfg.hidden_size
    # NLACriticModel wraps backbone + value_head. Constructor takes both.
    critic = NLACriticModel(cfg, base)
    # Identity init the value head (Linear has bias=False per models.py:82)
    with torch.no_grad():
        critic.value_head.weight.copy_(torch.eye(d_model, dtype=dtype))
    critic = critic.to(dtype)
    print(f"[critic] truncated to {num_layers} layers, value_head identity-init "
          f"(weight norm = {critic.value_head.weight.float().norm().item():.3f})")
    return critic


# ----------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay to min_lr
# ----------------------------------------------------------------------------

def build_lr_lambda(warmup_steps, total_steps, min_lr_ratio):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        prog = min(1.0, prog)
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return min_lr_ratio + (1 - min_lr_ratio) * cos
    return fn


# ----------------------------------------------------------------------------
# AV forward: encode chat-template prompt + response, build response-only loss
# mask, forward through model with Karvonen hook firing on the marker token.
# ----------------------------------------------------------------------------

def _av_prepare_chunk(rows, tokenizer, inject_char, device, max_len=1024):
    """Return (input_ids, attn, loss_mask, v_batch) — all [B, T] (or [B, d])."""
    full_ids_list = []
    prompt_lens = []
    for row in rows:
        # row["prompt"] is list[{"role","content"}] with INJECT_PLACEHOLDER inside.
        # Replace with the actual injection char so the tokenizer emits the
        # marker token id at the right position.
        msgs = [
            {**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inject_char)}
            if isinstance(m.get("content"), str) else m
            for m in row["prompt"]
        ]
        prompt_str = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        # Response gets a trailing EOS so the model learns to stop.
        resp = row["response"] + (tokenizer.eos_token or "")
        resp_ids = tokenizer.encode(resp, add_special_tokens=False)
        full = prompt_ids + resp_ids
        if len(full) > max_len:
            # Truncate response from the right to fit. Prompt is fixed.
            full = full[:max_len]
        full_ids_list.append(torch.tensor(full, dtype=torch.long))
        prompt_lens.append(len(prompt_ids))

    bs = len(full_ids_list)
    T = max(t.numel() for t in full_ids_list)
    pad_id = tokenizer.eos_token_id
    batch_ids = torch.full((bs, T), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, T), dtype=torch.long, device=device)
    loss_mask = torch.zeros((bs, T), dtype=torch.float32, device=device)
    for i, t in enumerate(full_ids_list):
        L = t.numel()
        batch_ids[i, :L] = t.to(device)
        attn[i, :L] = 1
        # 1 on response positions, 0 on prompt + pad. The shift-by-one for CE
        # is applied later (in the loss computation), so this mask is in
        # "target token" space — positions whose CE we want to count.
        loss_mask[i, prompt_lens[i]:L] = 1
    v_batch = torch.tensor(
        np.stack([r["activation_vector"] for r in rows]),
        dtype=torch.float32, device=device,
    )
    return batch_ids, attn, loss_mask, v_batch


# ----------------------------------------------------------------------------
# AR forward: tokenize the already-built critic prompt, forward, take MSE on
# normalised (pred, gold).
# ----------------------------------------------------------------------------

def _ar_prepare_chunk(rows, tokenizer, device, max_len=1024):
    full_ids_list = []
    for row in rows:
        # AR's prompt is the already-filled critic template string.
        ids = tokenizer.encode(row["prompt"], add_special_tokens=True)
        if len(ids) > max_len:
            ids = ids[:max_len]
        full_ids_list.append(torch.tensor(ids, dtype=torch.long))
    bs = len(full_ids_list)
    T = max(t.numel() for t in full_ids_list)
    pad_id = tokenizer.eos_token_id
    batch_ids = torch.full((bs, T), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, T), dtype=torch.long, device=device)
    for i, t in enumerate(full_ids_list):
        L = t.numel()
        batch_ids[i, :L] = t.to(device)
        attn[i, :L] = 1
    gold = torch.tensor(
        np.stack([r["activation_vector"] for r in rows]),
        dtype=torch.float32, device=device,
    )
    return batch_ids, attn, gold


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["av", "ar"])
    p.add_argument("--base-ckpt", required=True,
                   help="HF dir for AV (base model) or AR (base model to truncate, "
                        "OR an already-prepared NLACriticModel checkpoint).")
    p.add_argument("--parquet", required=True, help="SFT data parquet")
    p.add_argument("--sidecar", default=None,
                   help="Sidecar source (defaults to --parquet for the dataset sidecar)")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Per-forward batch (= 'micro batch'). Effective batch = "
                        "batch_size × gradient_accumulation_steps.")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--ar-num-layers", type=int, default=25,
                   help="K+1 for AR mode — truncate base to this many transformer blocks")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--lr-warmup-steps", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Default: ON for AV (fits 8B + batch=64 + FA2 on 141 GB H200), "
                        "OFF for AR (smaller model + shorter seq fits comfortably).")
    p.add_argument("--attn-implementation", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap training rows (smoke runs)")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="nla-qwen3-8b")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16
    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = (args.mode == "av")
    if args.sidecar is None:
        args.sidecar = args.parquet

    # ---- tokenizer + nla config ----
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    cfg = load_nla_config(args.sidecar, tokenizer)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    print(f"[cfg] mode={args.mode} d_model={cfg.d_model} mse_scale={mse_scale_f}")

    # ---- model ----
    if args.mode == "av":
        print(f"[av] loading {args.base_ckpt}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_ckpt, torch_dtype=dtype,
            attn_implementation=args.attn_implementation,
        ).to(device)
        vectors_ref = [None]
        _register_karvonen_hook(
            model, vectors_ref,
            cfg.injection_token_id,
            cfg.injection_left_neighbor_id,
            cfg.injection_right_neighbor_id,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            print("[av] gradient_checkpointing ENABLED")
    else:  # ar
        # Check if --base-ckpt is already a critic ckpt (has value_head.safetensors)
        is_prepared_critic = (Path(args.base_ckpt) / "value_head.safetensors").exists()
        if is_prepared_critic:
            print(f"[ar] loading pre-prepared critic from {args.base_ckpt}")
            model = NLACriticModel.from_pretrained(
                args.base_ckpt, torch_dtype=dtype,
                attn_implementation=args.attn_implementation,
            ).to(device)
        else:
            print(f"[ar] truncating base {args.base_ckpt} to {args.ar_num_layers} layers")
            model = init_critic_from_base(args.base_ckpt, args.ar_num_layers, dtype).to(device)
        vectors_ref = None
        if args.gradient_checkpointing:
            # NLACriticModel wraps backbone; enable on inner module
            if hasattr(model.backbone, "gradient_checkpointing_enable"):
                model.backbone.gradient_checkpointing_enable()
                print("[ar] gradient_checkpointing ENABLED (backbone)")
    model.train()

    # ---- data ----
    print(f"[data] loading {args.parquet} (max_rows={args.max_rows})", flush=True)
    rows = load_sft_dataset(args.parquet, n_max=args.max_rows, mode=args.mode)
    print(f"[data] {len(rows)} rows", flush=True)

    # ---- optimizer + LR schedule ----
    try:
        import bitsandbytes as bnb
        optim_cls = bnb.optim.AdamW8bit
        print(f"[optim] using bitsandbytes AdamW8bit (bnb {bnb.__version__})")
    except ImportError:
        optim_cls = torch.optim.AdamW
        print("[optim] bitsandbytes unavailable, falling back to torch AdamW (fp32 m,v)")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = optim_cls(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        build_lr_lambda(args.lr_warmup_steps, args.num_steps,
                        args.min_lr / max(args.lr, 1e-12)),
    )
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[optim] trainable params: {n_trainable / 1e9:.2f} B")

    # ---- AR-only: predict-the-mean baseline for FVE logging ----
    fve_baseline = None
    if args.mode == "ar":
        _act = torch.tensor(
            [r["activation_vector"] for r in rows[: min(len(rows), 4000)]],
            dtype=torch.float32,
        )
        _mu = _act.mean(dim=0, keepdim=True)
        _mu_n = normalize_activation(_mu, mse_scale_f)
        _act_n = normalize_activation(_act, mse_scale_f)
        fve_baseline = ((_mu_n - _act_n) ** 2).mean(dim=-1).mean().item()
        print(f"[ar] predict-the-mean MSE baseline = {fve_baseline:.4f}")

    # ---- wandb ----
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- training loop ----
    rng = np.random.default_rng(args.seed)
    perm = list(range(len(rows)))
    rng.shuffle(perm)
    cursor = 0

    grad_accum = args.gradient_accumulation_steps
    eff_batch = args.batch_size * grad_accum
    print(f"[loop] {args.num_steps} steps, batch={args.batch_size} × "
          f"grad_accum={grad_accum} = eff_batch={eff_batch}")

    for step in range(args.num_steps):
        t0 = time.time()
        optim.zero_grad()
        accum_loss = 0.0
        accum_resp_tokens = 0  # AV only: total response tokens for normalization
        accum_n = 0

        for accum_idx in range(grad_accum):
            # ---- pick batch ----
            if cursor + args.batch_size > len(perm):
                rng.shuffle(perm)
                cursor = 0
            chunk_rows = [rows[i] for i in perm[cursor:cursor + args.batch_size]]
            cursor += args.batch_size

            # ---- forward + loss ----
            if args.mode == "av":
                ids, attn, loss_mask, v_batch = _av_prepare_chunk(
                    chunk_rows, tokenizer, cfg.injection_char, device,
                    max_len=args.max_len,
                )
                vectors_ref[0] = v_batch
                try:
                    logits = model(input_ids=ids, attention_mask=attn).logits.float()
                finally:
                    vectors_ref[0] = None
                # Shift-by-one CE on response tokens. Predict ids[:, t+1] from
                # logits[:, t]. Mask is in TARGET space (positions of tokens
                # to predict), so mask[:, 1:] aligned with logits[:, :-1].
                shift_logits = logits[:, :-1].contiguous()
                shift_targets = ids[:, 1:].contiguous()
                shift_mask = loss_mask[:, 1:].contiguous()
                V = shift_logits.size(-1)
                per_tok = F.cross_entropy(
                    shift_logits.view(-1, V),
                    shift_targets.view(-1),
                    reduction="none",
                ).view(shift_targets.shape)
                n_resp = shift_mask.sum().clamp(min=1)
                loss = (per_tok * shift_mask).sum() / n_resp
                accum_resp_tokens += int(n_resp.item())
            else:  # ar
                ids, attn, gold = _ar_prepare_chunk(
                    chunk_rows, tokenizer, device, max_len=args.max_len,
                )
                pred = critic_predict(model, ids, attn, mse_scale_f)
                pred_n = normalize_activation(pred, mse_scale_f)
                gold_n = normalize_activation(gold, mse_scale_f)
                loss = F.mse_loss(pred_n, gold_n)

            # Scale loss for accumulation; gradients sum correctly.
            (loss / grad_accum).backward()
            accum_loss += loss.item()
            accum_n += 1

        # ---- step ----
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optim.step()
        sched.step()

        mean_loss = accum_loss / max(accum_n, 1)
        cur_lr = sched.get_last_lr()[0]

        log = {
            "step": step,
            "loss": mean_loss,
            "lr": cur_lr,
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "wall_s": time.time() - t0,
        }
        line = (f"step {step:04d} | loss {mean_loss:.4f} | lr {cur_lr:.2e} "
                f"| grad {log['grad_norm']:.3f} | t {log['wall_s']:.1f}s")
        if args.mode == "ar" and fve_baseline is not None:
            fve = (1.0 - mean_loss / fve_baseline) * 100.0
            log["fve_pct"] = fve
            line += f" | FVE {fve:.1f}%"
        if args.mode == "av":
            log["resp_tokens"] = accum_resp_tokens
            line += f" | resp_toks {accum_resp_tokens}"
        print(line, flush=True)
        if not args.no_wandb:
            wandb.log(log, step=step)

        # ---- save ----
        if (step + 1) % args.save_every == 0 or (step + 1) == args.num_steps:
            out_dir = save_dir / f"iter_{step + 1:07d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[save] → {out_dir}", flush=True)
            if args.mode == "av":
                model.save_pretrained(str(out_dir))
                tokenizer.save_pretrained(str(out_dir))
            else:
                model.save_pretrained(str(out_dir))
                tokenizer.save_pretrained(str(out_dir))
            # Copy the sidecar so the RL trainer can find injection_token_id etc.
            import shutil
            sidecar_src = Path(args.sidecar)
            if sidecar_src.is_file() and sidecar_src.suffix == ".parquet":
                sidecar_yaml = sidecar_src.with_suffix(".parquet.nla_meta.yaml")
                if sidecar_yaml.exists():
                    shutil.copy2(sidecar_yaml, out_dir / "nla_meta.yaml")

    print("done.", flush=True)
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
