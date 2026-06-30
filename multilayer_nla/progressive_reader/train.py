"""Progressive Reader v0 — trainer (spec §8, §9). Mirrors train_ar_multi's LoRA/optimizer/
LR/checkpoint setup; the only new bits are the stage-bucketed batching and the masked
progressive loss. Progressive and Flat share architecture/optimizer/#updates/seed; the
objective lives entirely in the per-stage active-layer mask + the config schedule.

  python -m multilayer_nla.progressive_reader.train --config configs/..._progressive.yaml \
      --seed 0 --run-dir runs/progressive_reader_v0/progressive_seed0
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def _load_config(path):
    import yaml
    import os
    c = yaml.safe_load(Path(path).read_text())
    # expand ${ENV} in the data path
    c["data"]["path"] = os.path.expandvars(c["data"]["path"])
    c["stages"] = {int(k): tuple(v) for k, v in c["stages"].items()}
    return c


def _draw_batch(ds, base_indices, stage, device, pad):
    idx = [b * len(ds.budgets) + stage for b in base_indices]
    return ds.collate([ds[i] for i in idx], device, pad)


def main():
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, inject_adapter_in_model
    from safetensors.torch import save_file

    from multilayer_nla.train_ar_multi import build_lr_lambda
    from multilayer_nla.progressive_reader.schedule import PREFIX_BUDGETS, TARGET_LAYERS, validate_schedule
    from multilayer_nla.progressive_reader.model import init_reader
    from multilayer_nla.progressive_reader.data import load_base_rows, ProgressiveReaderDataset
    from multilayer_nla.progressive_reader.loss import masked_stage_loss, layer_balanced_weights, predict_mean_baseline
    from multilayer_nla.progressive_reader import evaluate as ev

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--loss-mode", choices=["progressive_stage_mean", "progressive_layer_balanced"],
                   default=None, help="override config loss.mode (the headline runs BOTH)")
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None,
                   help="override config (flip on for micro-batch 64+ on an 80GB H100)")
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    cfg = _load_config(args.config)
    if args.loss_mode:
        cfg["loss"]["mode"] = args.loss_mode
    if args.gradient_checkpointing is not None:
        cfg["train"]["gradient_checkpointing"] = args.gradient_checkpointing
    if args.num_steps is not None:
        cfg["train"]["num_steps"] = args.num_steps
    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    target_layers = tuple(cfg["target_layers"])
    budgets = tuple(cfg.get("prefix_budgets", PREFIX_BUDGETS))
    stages = cfg["stages"]
    require_nested = cfg["objective"] == "progressive"
    validate_schedule(stages, target_layers, require_nested=require_nested)

    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda"
    run = Path(args.run_dir); run.mkdir(parents=True, exist_ok=True)
    Path(run / "resolved_config.yaml").write_text(json.dumps(cfg, indent=2))
    tok = AutoTokenizer.from_pretrained(cfg["reader"]["base_ckpt"])

    # data: tokenize-once, strict-128, doc split
    rc = cfg["data"]
    base = load_base_rows(rc["path"], tok, target_layers=target_layers,
                          teacher_field=rc.get("teacher_field", "auto"),
                          require_full_max_budget=rc.get("require_full_prefix_for_max_budget", True),
                          max_budget=max(budgets),
                          fracs=tuple(rc["split"]["fracs"]), seed=rc["split"]["seed"],
                          names=tuple(rc["split"]["names"]), max_documents=rc.get("max_documents"))
    tr, dv = base["train"], base["dev"]
    d_model = int(tr["targets"].shape[-1])
    mse_scale = math.sqrt(d_model)
    print(f"[reader] train {len(tr['full_ids'])} / dev {len(dv['full_ids'])} rows · d={d_model} · "
          f"objective={cfg['objective']} loss={cfg['loss']['mode']}")

    # per-LAYER predict-mean baselines from TRAIN ONLY (FVE denominator; spec §1.4/§9)
    train_baselines = [predict_mean_baseline(torch.tensor(tr["targets"][:, j, :], dtype=torch.float32), mse_scale)
                       for j in range(len(target_layers))]
    print("[reader] train baselines: " + ", ".join(f"L{l}={b:.3f}" for l, b in zip(target_layers, train_baselines)))

    # model + LoRA (+ optional gradient checkpointing for H100 headroom)
    o = cfg["optim"]
    model = init_reader(cfg["reader"]["base_ckpt"], target_layers=target_layers,
                        strip_final_norm=cfg["reader"].get("strip_final_norm", True)).to(device)
    if cfg["train"].get("gradient_checkpointing"):
        model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone, "config"):
            model.backbone.config.use_cache = False
    inject_adapter_in_model(LoraConfig(
        r=o["lora_r"], lora_alpha=o["lora_alpha"], lora_dropout=o.get("lora_dropout", 0.0),
        bias="none", task_type="CAUSAL_LM", use_rslora=o.get("use_rslora", True),
        target_modules=o.get("target_modules", TARGET_MODULES)), model.backbone)
    for n_, p_ in model.named_parameters():
        p_.requires_grad_(("lora_" in n_) or n_.startswith("heads."))
    model.train()
    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    print(f"[reader] trainable {sum(p_.numel() for p_ in trainable)/1e6:.1f}M")

    layer_w = None
    if cfg["loss"]["mode"] == "progressive_layer_balanced":
        layer_w = torch.tensor(layer_balanced_weights(target_layers, stages), device=device)

    try:
        import bitsandbytes as bnb
        optim_cls = bnb.optim.AdamW8bit
    except ImportError:
        optim_cls = torch.optim.AdamW
    optim = optim_cls(trainable, lr=o["lr"], betas=tuple(o.get("betas", (0.9, 0.95))),
                      weight_decay=o.get("weight_decay", 0.0))
    n_steps = cfg["train"]["num_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, build_lr_lambda(o["lr_warmup_steps"], n_steps, o["min_lr"] / max(o["lr"], 1e-12)))

    if not args.no_wandb:
        import wandb
        wandb.init(project=cfg.get("wandb_project", "multi layer nla"),
                   name=f"reader-{cfg['objective']}-{cfg['loss']['mode']}-s{seed}", config=cfg)

    train_ds = ProgressiveReaderDataset(tr, tok, stages, target_layers=target_layers, budgets=budgets)
    dev_ds = dv  # rows dict; matrix builds its own dataset
    pad = tok.eos_token_id
    bs = cfg["train"]["batch_size"]
    grad_accum = cfg["train"]["gradient_accumulation_steps"]
    rng = np.random.default_rng(seed)
    # per-budget shuffled base-index streams (balanced: every row hits every budget per pass)
    streams = {s: (list(rng.permutation(train_ds.n_base)), [0]) for s in range(len(budgets))}
    log_f = open(run / "train_log.jsonl", "w")
    best_metric, best_dir = -1e9, None
    eval_every, save_every = cfg["train"]["eval_every"], cfg["train"]["save_every"]

    def next_bases(stage, k):
        order, cur = streams[stage]
        if cur[0] + k > len(order):
            rng.shuffle(order); cur[0] = 0
        b = order[cur[0]:cur[0] + k]; cur[0] += k
        return b

    for step in range(n_steps):
        t0 = time.time(); optim.zero_grad(); accum = 0.0
        for _ in range(grad_accum):
            stage = step % len(budgets)              # round-robin budgets -> balanced + bucketed
            bases = next_bases(stage, bs)
            b = _draw_batch(train_ds, bases, stage, device, pad)
            from multilayer_nla.progressive_reader.model import reader_predict
            pred = reader_predict(model, b["input_ids"], b["attention_mask"], mse_scale)
            loss = masked_stage_loss(pred, b["targets"], mse_scale, b["active_mask"], layer_w)
            (loss / grad_accum).backward(); accum += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(trainable, o["max_grad_norm"])
        optim.step(); sched.step()
        rec = {"step": step, "loss": accum / grad_accum, "lr": sched.get_last_lr()[0],
               "grad_norm": float(gn), "wall_s": time.time() - t0}
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if step % 20 == 0:
            print(f"step {step:04d} | loss {rec['loss']:.4f} | lr {rec['lr']:.2e} | {rec['wall_s']:.1f}s", flush=True)
        if not args.no_wandb:
            import wandb; wandb.log(rec, step=step)

        if (step + 1) % eval_every == 0 or (step + 1) == n_steps:
            model.eval()
            recs = ev.run_matrix(model, dev_ds, tok, stages, mse_scale, train_baselines,
                                 target_layers=target_layers, budgets=budgets, text_mode="real",
                                 device=device, batch_size=cfg["train"].get("eval_batch_size", 128),
                                 max_len=o["max_len"])
            cells = {f"{B},{l}": {"fve": ev._cell_fve(recs, B, l, train_baselines[i])}
                     for B in budgets for i, l in enumerate(target_layers)}
            m = ev.m_scheduled(cells, budgets)
            (run / f"dev_matrix_step{step+1}.json").write_text(json.dumps(cells, indent=2))
            print(f"  [dev] M_scheduled = {m*100:.2f}%", flush=True)
            if not args.no_wandb:
                import wandb; wandb.log({"dev/M_scheduled": m}, step=step)
            if m > best_metric:
                best_metric = m
                _save(run / "best", model, tok, cfg, target_layers, mse_scale, d_model, stages, train_baselines, seed, best_metric)
            model.train()
        if (step + 1) % save_every == 0 or (step + 1) == n_steps:
            _save(run / f"iter_{step+1:07d}", model, tok, cfg, target_layers, mse_scale, d_model, stages, train_baselines, seed, best_metric)

    log_f.close()
    if not args.no_wandb:
        import wandb; wandb.finish()
    print(f"[reader] done. best dev M_scheduled = {best_metric*100:.2f}% -> {run/'best'}")


def _save(out, model, tok, cfg, target_layers, mse_scale, d_model, stages, train_baselines, seed, best_metric):
    import torch
    from safetensors.torch import save_file
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    sd = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters()
          if ("lora_" in n) or n.startswith("heads.")}
    save_file(sd, str(out / "reader.safetensors"))
    o = cfg["optim"]
    (out / "reader_meta.json").write_text(json.dumps({
        "target_layers": list(target_layers), "lora_r": o["lora_r"], "lora_alpha": o["lora_alpha"],
        "target_modules": o.get("target_modules", TARGET_MODULES), "mse_scale": mse_scale,
        "d_model": d_model, "strip_final_norm": cfg["reader"].get("strip_final_norm", True),
        "stages": {str(k): list(v) for k, v in stages.items()},
        "train_baselines": train_baselines, "objective": cfg["objective"],
        "loss_mode": cfg["loss"]["mode"], "seed": seed, "best_dev_M_scheduled": best_metric,
    }, indent=2))
    tok.save_pretrained(str(out))
    print(f"[save] -> {out}")


if __name__ == "__main__":
    main()
