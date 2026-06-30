"""Progressive Reader v0 — evaluation matrix + controls + comparisons (spec §10-§12).

Produces, for a checkpoint, the full 3x7 FVE(B, ℓ) matrix on real / no-text / shuffled
teacher text (every cell, incl. ones the Progressive objective never directly supervised —
the model predicts all 7 layers at every forward), with document-level bootstrap CIs, the
stage-oriented gains (G_local / G_outer), per-example records for paired comparisons, a
summary.md (labelled CONDITIONAL gold-prefix reader ceilings, NOT absolute), and heatmaps.
Reuses the repo's directional FVE (normalize_activation + compute_predict_mean_baselines).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from multilayer_nla.progressive_reader.schedule import PREFIX_BUDGETS, TARGET_LAYERS

CLAIM = ("Results are CONDITIONAL gold-prefix reader ceilings C_{B,ℓ}^{gold;H,D} — conditional on "
         "the current teacher-label distribution, AR architecture, source activation bank, exact "
         "prefix budgets, target normalization, and optimization recipe. NOT absolute "
         "information-theoretic ceilings, NOT semantic faithfulness, NOT causal faithfulness.")


# ---------------------------------------------------------------- checkpoint IO

def load_reader(base_ckpt, ckpt_dir, device="cuda", dtype=None):
    """Rebuild the 7-tap reader from a saved ckpt (mirror evaluate_e2e.load_critic): init the
    truncated backbone + heads, inject the LoRA config, load lora+heads weights, load meta
    (target_layers, mse_scale, train_baselines, stages)."""
    import torch
    from safetensors.torch import load_file
    from peft import LoraConfig, inject_adapter_in_model
    from multilayer_nla.progressive_reader.model import init_reader
    meta = json.loads((Path(ckpt_dir) / "reader_meta.json").read_text())
    model = init_reader(base_ckpt, target_layers=tuple(meta["target_layers"]),
                        dtype=dtype or torch.bfloat16, strip_final_norm=meta.get("strip_final_norm", True))
    model = model.to(device)
    inject_adapter_in_model(LoraConfig(
        r=meta["lora_r"], lora_alpha=meta["lora_alpha"], lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM", use_rslora=True, target_modules=meta["target_modules"]), model.backbone)
    sd = load_file(str(Path(ckpt_dir) / "reader.safetensors"))
    _, unexp = model.load_state_dict(sd, strict=False)
    assert not unexp, f"unexpected keys {unexp[:3]}"
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, meta


# ---------------------------------------------------------------- the matrix

def run_matrix(model, rows, tokenizer, stages, mse_scale, train_baselines, *,
               target_layers=TARGET_LAYERS, budgets=PREFIX_BUDGETS, text_mode="real",
               shuffle_perm=None, device="cuda", batch_size=128, max_len=1024):
    """Forward the eval split at all (budget) stages; return per-(budget, layer) records.

    Records (one per (stage-view, layer)) carry the normalized squared error, cosine, norms,
    doc_id, src_row_id, budget, layer, eff length — enough for per-cell FVE + doc-bootstrap +
    paired comparisons (spec §10). The model predicts ALL 7 layers every forward, so every
    cell is filled regardless of the training schedule."""
    import numpy as np
    import torch
    from multilayer_nla.progressive_reader.data import ProgressiveReaderDataset
    from multilayer_nla.progressive_reader.loss import per_tap_dir_loss

    ds = ProgressiveReaderDataset(rows, tokenizer, stages, target_layers=target_layers,
                                  budgets=budgets, text_mode=text_mode, shuffle_perm=shuffle_perm)
    pad = tokenizer.eos_token_id
    recs = []
    order = list(range(len(ds)))
    with torch.no_grad():
        for cs in range(0, len(order), batch_size):
            batch = [ds[i] for i in order[cs:cs + batch_size]]
            b = ds.collate(batch, device, pad)
            if b["input_ids"].shape[1] > max_len:        # readout is the LAST token; never truncate it
                continue
            from multilayer_nla.progressive_reader.model import reader_predict
            pred = reader_predict(model, b["input_ids"], b["attention_mask"], mse_scale)   # [B,k,d]
            gold = b["targets"]                          # [B,k,d] raw
            sqerr = per_tap_dir_loss(pred, gold, mse_scale).cpu().numpy()                  # [B,k]
            from nla.schema import normalize_activation
            pn = normalize_activation(pred, mse_scale)
            cos = (pn * normalize_activation(gold, mse_scale)).sum(-1).cpu().numpy() / (mse_scale ** 2)  # [B,k]
            pnorm = pred.float().norm(dim=-1).cpu().numpy()
            gnorm = gold.float().norm(dim=-1).cpu().numpy()
            for r in range(len(batch)):
                bud = b["budgets"][r]
                for j, layer in enumerate(target_layers):
                    recs.append({
                        "budget": int(bud), "target_layer": int(layer),
                        "doc_id": b["doc_ids"][r], "src_row_id": int(b["src_row_ids"][r]),
                        "fve_sqerr": float(sqerr[r, j]), "cosine": float(cos[r, j]),
                        "directional_loss": float(sqerr[r, j] / 2.0),
                        "target_norm": float(gnorm[r, j]), "prediction_norm": float(pnorm[r, j]),
                        "effective_teacher_prefix_length": int(b["eff_lengths"][r]),
                        "is_no_text": text_mode == "no_text", "is_shuffled": text_mode == "shuffled",
                    })
    return recs


def _cell_fve(recs, budget, layer, baseline):
    import numpy as np
    se = np.array([r["fve_sqerr"] for r in recs if r["budget"] == budget and r["target_layer"] == layer])
    return (1.0 - se.mean() / baseline) if len(se) else float("nan")


def _cell_bootstrap(recs, budget, layer, baseline, n_boot=1000, seed=0):
    """Document-level bootstrap (spec §10): resample docs, recompute FVE. (lo, hi) 95% CI."""
    import numpy as np
    by_doc = {}
    for r in recs:
        if r["budget"] == budget and r["target_layer"] == layer:
            by_doc.setdefault(r["doc_id"], []).append(r["fve_sqerr"])
    docs = list(by_doc)
    if not docs:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(docs), size=len(docs), replace=True)
        se = np.concatenate([by_doc[docs[i]] for i in pick])
        vals.append(1.0 - se.mean() / baseline)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def matrix_from_records(real, no_text, shuffled, baselines, *, target_layers=TARGET_LAYERS,
                        budgets=PREFIX_BUDGETS, n_boot=1000):
    """Assemble the 3x7 matrix with real / no_text / shuffled FVE + real−shuffled + doc-bootstrap CI."""
    bl = {l: baselines[i] for i, l in enumerate(target_layers)}
    cells = {}
    for B in budgets:
        for l in target_layers:
            real_fve = _cell_fve(real, B, l, bl[l])
            sh_fve = _cell_fve(shuffled, B, l, bl[l]) if shuffled else float("nan")
            nt_fve = _cell_fve(no_text, B, l, bl[l]) if no_text else float("nan")
            ci = _cell_bootstrap(real, B, l, bl[l], n_boot=n_boot)
            cells[f"{B},{l}"] = {
                "budget": B, "layer": l, "fve": real_fve, "fve_ci95": list(ci),
                "fve_no_text": nt_fve, "fve_shuffled": sh_fve,
                "fve_real_minus_shuffled": (real_fve - sh_fve) if shuffled else float("nan"),
                "cosine": _cell_cos(real, B, l),
            }
    return cells


def _cell_cos(recs, budget, layer):
    import numpy as np
    c = np.array([r["cosine"] for r in recs if r["budget"] == budget and r["target_layer"] == layer])
    return float(c.mean()) if len(c) else float("nan")


def stage_gains(cells, budgets=PREFIX_BUDGETS):
    """G_local = ½Σ_{23,25}[FVE(b1,ℓ)−FVE(b0,ℓ)];  G_outer = ¼Σ_{20,22,26,28}[FVE(b2,ℓ)−FVE(b1,ℓ)],
    where (b0,b1,b2) are the three ascending budgets (32/64/96)."""
    b0, b1, b2 = budgets
    def f(B, l):
        return cells[f"{B},{l}"]["fve"]
    g_local = sum(f(b1, l) - f(b0, l) for l in (23, 25)) / 2.0
    g_outer = sum(f(b2, l) - f(b1, l) for l in (20, 22, 26, 28)) / 4.0
    return {"G_local": g_local, "G_outer": g_outer}


def m_scheduled(cells, budgets=PREFIX_BUDGETS):
    """Common dev selection metric M_scheduled (spec §9), applied identically to all conditions.
    Stage 0 (smallest budget) -> L24; stage 1 -> {23,24,25}; stage 2 (largest) -> all 7."""
    b0, b1, b2 = budgets
    def f(B, l):
        return cells[f"{B},{l}"]["fve"]
    return (f(b0, 24)
            + sum(f(b1, l) for l in (23, 24, 25)) / 3.0
            + sum(f(b2, l) for l in TARGET_LAYERS) / len(TARGET_LAYERS)) / 3.0


# ---------------------------------------------------------------- paired comparison (vs flat)

def compare_conditions(recs_a, recs_b, baselines, *, target_layers=TARGET_LAYERS, n_boot=2000, seed=0):
    """ΔG_local / ΔG_outer = (A − B), paired DOCUMENT bootstrap (spec §11). recs_* are the
    REAL per-example records of two conditions on the SAME docs. Resample docs, recompute each
    condition's G on the shared resample, take the difference."""
    import numpy as np
    bl = {l: baselines[i] for i, l in enumerate(target_layers)}

    def by_doc(recs):
        d = {}
        for r in recs:
            d.setdefault(r["doc_id"], []).append(r)
        return d

    da, db = by_doc(recs_a), by_doc(recs_b)
    shared = [x for x in da if x in db]

    def gains(recs_list):
        flat = [r for rs in recs_list for r in rs]
        cells = {f"{B},{l}": {"fve": _cell_fve(flat, B, l, bl[l])}
                 for B in PREFIX_BUDGETS for l in target_layers}
        return stage_gains(cells)

    rng = np.random.default_rng(seed)
    dloc, dout = [], []
    for _ in range(n_boot):
        pick = rng.choice(len(shared), size=len(shared), replace=True)
        ga = gains([da[shared[i]] for i in pick])
        gb = gains([db[shared[i]] for i in pick])
        dloc.append(ga["G_local"] - gb["G_local"])
        dout.append(ga["G_outer"] - gb["G_outer"])
    return {
        "n_shared_docs": len(shared),
        "delta_G_local": {"mean": float(np.mean(dloc)),
                          "ci95": [float(np.percentile(dloc, 2.5)), float(np.percentile(dloc, 97.5))],
                          "frac_gt0": float(np.mean(np.array(dloc) > 0))},
        "delta_G_outer": {"mean": float(np.mean(dout)),
                          "ci95": [float(np.percentile(dout, 2.5)), float(np.percentile(dout, 97.5))],
                          "frac_gt0": float(np.mean(np.array(dout) > 0))},
    }


# ---------------------------------------------------------------- outputs

def write_summary(out_md, cells, gains, conditional_label, *, target_layers=TARGET_LAYERS,
                  budgets=PREFIX_BUDGETS):
    lines = [f"# Progressive Reader v0 — {conditional_label}\n", f"_{CLAIM}_\n",
             "## FVE(B, ℓ) — gold-prefix conditional reader ceilings\n",
             "| Budget | " + " | ".join(f"L{l}" for l in target_layers) + " |",
             "| --- | " + " | ".join("--:" for _ in target_layers) + " |"]
    for B in budgets:
        row = [f"{cells[f'{B},{l}']['fve']*100:+.1f}" for l in target_layers]
        lines.append(f"| {B} | " + " | ".join(row) + " |")
    lines += ["", "## real − shuffled (text dependence)\n",
              "| Budget | " + " | ".join(f"L{l}" for l in target_layers) + " |",
              "| --- | " + " | ".join("--:" for _ in target_layers) + " |"]
    for B in budgets:
        row = [f"{cells[f'{B},{l}']['fve_real_minus_shuffled']*100:+.1f}" for l in target_layers]
        lines.append(f"| {B} | " + " | ".join(row) + " |")
    lines += ["", f"G_local = {gains['G_local']*100:+.2f}pp  ·  G_outer = {gains['G_outer']*100:+.2f}pp",
              "(G_local: 32→64 gain on {L23,L25}; G_outer: 64→128 gain on {L20,L22,L26,L28}.)",
              "", "Do not read a high real-text FVE where real−shuffled is small."]
    Path(out_md).write_text("\n".join(lines))


def plot_heatmaps(plot_dir, cells, *, target_layers=TARGET_LAYERS, budgets=PREFIX_BUDGETS, tag="test"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    for key, fname in (("fve", f"{tag}_fve_heatmap.png"),
                       ("fve_real_minus_shuffled", "real_minus_shuffled_heatmap.png")):
        M = np.array([[cells[f"{B},{l}"][key] * 100 for l in target_layers] for B in budgets])
        fig, ax = plt.subplots(figsize=(7, 3))
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(target_layers))); ax.set_xticklabels([f"L{l}" for l in target_layers])
        ax.set_yticks(range(len(budgets))); ax.set_yticklabels(budgets); ax.set_ylabel("budget")
        for i in range(len(budgets)):
            for j in range(len(target_layers)):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, label="FVE %"); fig.tight_layout()
        fig.savefig(str(Path(plot_dir) / fname), dpi=140); plt.close(fig)


# ---------------------------------------------------------------- CLI

def main():
    import argparse
    import os
    import numpy as np
    import yaml
    from transformers import AutoTokenizer
    from multilayer_nla.progressive_reader.data import load_base_rows
    from multilayer_nla.progressive_reader.controls import doc_derangement

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--controls", nargs="+", default=["real", "no_text", "shuffled"])
    p.add_argument("--out", required=True)
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--compare-to", default=None, help="another condition's <out>/{split}_per_example.jsonl "
                                                      "(REAL records) for the paired ΔG vs Flat")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config)); cfg["data"]["path"] = os.path.expandvars(cfg["data"]["path"])
    stages = {int(k): tuple(v) for k, v in cfg["stages"].items()}
    target_layers = tuple(cfg["target_layers"])
    budgets = tuple(cfg.get("prefix_budgets", PREFIX_BUDGETS))
    rc = cfg["data"]

    model, meta = load_reader(args.base_ckpt, args.checkpoint)
    mse_scale = meta["mse_scale"]; baselines = meta["train_baselines"]
    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    base = load_base_rows(rc["path"], tok, target_layers=target_layers,
                          teacher_field=rc.get("teacher_field", "auto"),
                          require_full_max_budget=rc.get("require_full_prefix_for_max_budget", True),
                          max_budget=max(budgets),
                          fracs=tuple(rc["split"]["fracs"]), seed=rc["split"]["seed"],
                          names=tuple(rc["split"]["names"]))
    rows = base[args.split]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    by_mode = {}
    for mode in args.controls:
        perm = doc_derangement(rows["doc_ids"], seed=0) if mode == "shuffled" else None
        by_mode[mode] = run_matrix(model, rows, tok, stages, mse_scale, baselines,
                                   target_layers=target_layers, budgets=budgets, text_mode=mode,
                                   shuffle_perm=perm, batch_size=args.batch_size, max_len=cfg["optim"]["max_len"])
    cells = matrix_from_records(by_mode.get("real", []), by_mode.get("no_text"), by_mode.get("shuffled"),
                                baselines, target_layers=target_layers, budgets=budgets, n_boot=args.n_boot)
    gains = stage_gains(cells, budgets)
    (out / f"{args.split}_matrix.json").write_text(json.dumps(
        {"cells": cells, "gains": gains, "M_scheduled": m_scheduled(cells, budgets),
         "objective": meta["objective"], "loss_mode": meta["loss_mode"]}, indent=2))
    with open(out / f"{args.split}_per_example.jsonl", "w") as f:
        for mode, recs in by_mode.items():
            for r in recs:
                f.write(json.dumps({"condition": meta["objective"], "loss_mode": meta["loss_mode"],
                                    "seed": meta.get("seed"), "split": args.split, **r}) + "\n")
    label = f"{meta['objective']} / {meta['loss_mode']} (split={args.split})"
    write_summary(out / "summary.md", cells, gains, label, target_layers=target_layers, budgets=budgets)
    plot_heatmaps(out / "plots", cells, target_layers=target_layers, budgets=budgets, tag=args.split)

    if args.compare_to:
        b_real = [json.loads(l) for l in open(args.compare_to) if not json.loads(l)["is_shuffled"]
                  and not json.loads(l)["is_no_text"]]
        comp = compare_conditions(by_mode["real"], b_real, baselines, target_layers=target_layers)
        (out / "bootstrap_comparisons.json").write_text(json.dumps(comp, indent=2))
        print(f"[compare] ΔG_local {comp['delta_G_local']['mean']*100:+.2f}pp "
              f"{[round(x*100,2) for x in comp['delta_G_local']['ci95']]}  ·  "
              f"ΔG_outer {comp['delta_G_outer']['mean']*100:+.2f}pp "
              f"{[round(x*100,2) for x in comp['delta_G_outer']['ci95']]}")
    print(f"[eval] {label}: M_sched {m_scheduled(cells, budgets)*100:.2f}%  G_local {gains['G_local']*100:+.2f} "
          f"G_outer {gains['G_outer']*100:+.2f} -> {out}")


if __name__ == "__main__":
    main()

