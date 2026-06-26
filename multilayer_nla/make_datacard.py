"""Generate the self-contained DATACARD / results card for the §7 SFT control sweep.

READ-ONLY reporting layer over `analyze_sweep` — it re-uses the same loaders and the
paired-bootstrap stats so EVERY number in the card is computed from the frozen eval
outputs on disk (no hand-transcription). Meant to be the human-readable index that ships
next to the weights + raw results when the sweep is published.

  python -m multilayer_nla.make_datacard \
      --eval-dir   $EVALC                 \  # 3-tap sweep results ($EVALC/test/*)
      --arl24-dir  $EVALC/test_arL24       \  # optional: L24-only-AR cut
      --sweep-dir  $SWEEP                   \  # optional: dataset row counts
      --weights-repo  <hf-repo-id>          \  # optional: pointer string for the card
      --out        $EVALC/DATACARD.md

Nothing here trains, evaluates, or mutates data. If a directory is absent its section is
simply skipped, so the card degrades gracefully on a partial run.
"""

import argparse
from pathlib import Path

from .analyze_sweep import (AV_INPUT_LAYERS, CONDS, bottleneck, contrasts,
                            headline, load_results, table)


def _layer_geometry(spec):
    """(distinct, span) for an 'a,b,c' AV-input spec — descriptive only."""
    xs = [int(t) for t in spec.split(",")]
    return len(set(xs)), (max(xs) - min(xs))


def _conditions_table():
    lines = ["| condition | AV input layers | distinct layers | span | markers (k) |",
             "| --- | --- | ---: | ---: | ---: |"]
    for c in CONDS:
        spec = AV_INPUT_LAYERS[c]
        nd, span = _layer_geometry(spec)
        k = len(spec.split(","))
        lines.append(f"| {c} | {spec} | {nd} | {span} | {k} |")
    return "\n".join(lines)


def _dataset_section(sweep_dir):
    """Row counts straight from parquet metadata (cheap; no full read)."""
    try:
        import pyarrow.parquet as pq
    except Exception:
        return ""
    sweep_dir = Path(sweep_dir)
    wanted = (["ar_common", "ar_dev", "ar_test"] +
              [f"av_{c}" for c in CONDS] +
              [f"rl_dev_{c}" for c in CONDS] + [f"rl_test_{c}" for c in CONDS])
    counts = {}
    rows = []
    for name in wanted:
        p = sweep_dir / f"{name}.parquet"
        if p.exists():
            try:
                n = pq.ParquetFile(p).metadata.num_rows
                counts[name] = n
                rows.append(f"| `{name}.parquet` | {n:,} |")
            except Exception:
                rows.append(f"| `{name}.parquet` | (unreadable) |")
    if not rows:
        return ""
    # INTEGRITY: every av_<cond> streams the full av_sft (no filter), so all counts must be
    # equal. Flag any outlier loudly — an unequal count means that condition's AV trained on
    # a truncated/biased subset and its comparison is confounded.
    av_counts = {c: counts[f"av_{c}"] for c in CONDS if f"av_{c}" in counts}
    warn = ""
    if av_counts:
        mode = max(set(av_counts.values()), key=list(av_counts.values()).count)
        bad = {c: n for c, n in av_counts.items() if n != mode}
        if bad:
            warn = ("\n> ⚠ **AV row-count mismatch — DATA DEFECT.** Every `av_<cond>` must have the\n"
                    f"> same row count ({mode:,}, the full `av_sft`); these do not: "
                    + ", ".join(f"`{c}`={n:,}" for c, n in bad.items())
                    + ".\n> The affected condition(s)' AV trained on a truncated/biased subset — "
                      "their\n> end-to-end numbers are CONFOUNDED and must be rebuilt or flagged "
                      "provisional.\n")
    return ("\n## Datasets (built by `build_sweep.py`; raw vectors, `norm=\"none\"`)\n\n"
            "All AV-input columns are positional `av_in_*`; AR targets are the fixed\n"
            "`activation_prev/centre/next` (== L23/L24/L25) — distinct names so the target\n"
            "cannot follow the input. The condition lives in the data, not a train-time flag.\n"
            + warn +
            "\n| parquet | rows |\n| --- | ---: |\n" + "\n".join(rows) + "\n")


ABSTRACT = """\
# §7 SFT control sweep — multi-layer AV input → fixed-target reconstruction

**Question.** Does giving the activation *verbalizer* (AV) a multi-layer slice of the
residual stream improve end-to-end reconstruction of the SAME fixed target state
[L23, L24, L25], versus a single layer or a redundant repeat? This is a pre-registered,
held-out SFT control sweep (one H200, sequential, **no RL**).

**Design (what makes it causal).** Every condition reconstructs the identical fixed target
[L23, L24, L25]; only the **AV input layers** vary. A single shared AR reconstructor is
trained once and frozen — identical for all conditions — so any difference is purely the
verbalizer's. Document-level 80/10/10 splits; checkpoints selected on **dev only**; one-shot
held-out **test**. Predict-the-mean baselines from the eval split; the shuffled-generation
control permutes across documents and must collapse.

**Headline contrast.** `local` (23,24,25) vs `duplicate` (24,24,24): 3 distinct adjacent
layers vs the same layer thrice, at fixed marker count — isolating layer *diversity* from
injection bandwidth.
"""

CONTROLS = """\
## Controls & caveats (read before citing any gap)

- **Shuffled-generation control** collapses to ≈ −80% penalized FVE in every condition →
  no document→activation leakage; the FVE is not distributional luck.
- **Test⟂dev** document-disjoint, and every test doc re-hashes to the test bucket under the
  same `doc_bucket` formula used to build the split (re-derived in `analysis.md`).
- **Shared, frozen AR** — differences are AV-side only; this is not co-training (that would
  be the deferred RL phase and would confound the gap).
- **Verbalizer is the dominant bottleneck.** Every condition sits ~18–22 pp below the
  AR-only gold ceiling (gold explanation → AR), so the across-condition spread (~4 pp) is
  real-but-secondary to the verbalize step.
- **Warm-start labels are layer-blind** (single-layer L24, text-derived next-token feature
  descriptions). Any multi-layer-input benefit therefore arises *despite* the AV never
  being supervised to describe the extra layers — it is an input-diversity effect, not a
  label effect.
- Point estimates are not claims: the verdicts come from the **paired bootstrap over shared
  documents** (the `Key paired contrasts` block), not from marginal-CI overlap.
"""


def build_card(eval_dir, arl24_dir=None, sweep_dir=None, weights_repo=None,
               results_repo=None, n_boot=2000, seed=0):
    results = load_results(eval_dir)
    if not results:
        raise SystemExit(f"no test_<cond>.json under {Path(eval_dir)/'test'}")
    parts = [ABSTRACT,
             "## Conditions (the AV input is the only thing that varies)\n",
             _conditions_table(), "",
             "## Held-out TEST results (3-tap target [L23,L24,L25])\n",
             table(results), "",
             headline(results, n_boot, seed), "",
             contrasts(results, n_boot, seed), "",
             bottleneck(results, eval_dir), ""]

    if arl24_dir and (Path(arl24_dir) / "test_local.json").exists():
        r24 = load_results(eval_dir, test_dir=arl24_dir)
        parts += ["## L24-only-AR cut — reconstruct a SINGLE fixed target (L24)\n",
                  "_Same AV checkpoints, but the shared AR reconstructs only L24, removing the\n"
                  "multi-tap averaging. `overall` ≡ the L24 contrast. Cleanest version of the question._\n",
                  table(r24), "",
                  contrasts(r24, n_boot, seed), ""]

    parts += [CONTROLS]
    ds = _dataset_section(sweep_dir) if sweep_dir else ""
    if ds:
        parts.append(ds)
    if weights_repo or results_repo:
        art = ["\n## Weights & raw artifacts\n"]
        if weights_repo:
            art.append(f"- Weights (LoRA adapters): shared AR + per-condition AV — model repo "
                       f"[`{weights_repo}`](https://huggingface.co/{weights_repo}).")
        if results_repo:
            art.append(f"- Results + this datacard: dataset repo "
                       f"[`{results_repo}`](https://huggingface.co/datasets/{results_repo}) "
                       f"(alongside the L19-29 activation bank).")
        art += ["- Raw per-condition summaries: `test/test_<cond>.json`; per-example: `test/test_<cond>.jsonl`.",
                "- Full analysis (distributions, qualitative samples, leakage): `analysis.md`"
                + (", `analysis_arL24.md`" if arl24_dir else "") + ".",
                "- Best-verbalization showcase (top-FVE per multi-layer condition): `best_samples.md`.",
                "- Headline table: `result_table.md`."]
        parts += art
    return "\n".join(parts)


def build_model_card(weights_repo=None, results_repo=None, ar_step=3000, av_step=1000,
                     base="Qwen/Qwen3-8B", weights_prefix=""):
    """README/model card for the WEIGHTS repo: base model, adapter inventory, how to load,
    cross-link to the dataset repo. Standalone — needs no eval outputs."""
    def rp(p):
        return f"{weights_prefix}/{p}" if weights_prefix else p
    title = weights_repo or "NLA multi-layer sweep"
    lines = [f"# {title} — LoRA adapters", "",
             f"LoRA adapters for the §7 multi-layer NLA SFT control sweep, all on base "
             f"**[`{base}`](https://huggingface.co/{base})**.", "",
             "An NLA pairs an **AV** (activation→text verbalizer) with an **AR** (text→activation "
             "reconstructor) through a natural-language bottleneck. The base model is NOT included — "
             "load it from the Hub and apply an adapter.", "",
             "## Adapter inventory", "",
             "| path | role | detail |", "| --- | --- | --- |",
             f"| `{rp('ar/iter_%07d' % ar_step)}` | shared AR | reconstructs the FIXED target "
             f"[L23,L24,L25]; used by every condition's 3-tap eval |",
             f"| `{rp('ar_L24')}` | L24-only AR | reconstructs just [L24] (single-target cut) |"]
    for c in CONDS:
        lines.append(f"| `{rp('av_%s/iter_%07d' % (c, av_step))}` | AV ({c}) | verbalizer; "
                     f"AV input layers {AV_INPUT_LAYERS[c]} |")
    lines += ["",
        "Each **AR** dir also contains `ar_multitap.safetensors` (per-tap identity-init heads) and "
        "`ar_meta.json` (tap layers, mse_scale, d_model) — both required to reconstruct; see "
        "`multilayer_nla/evaluate_e2e.py:load_critic`.", "",
        "## Load an AV (verbalizer)", "",
        "```python", "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        f"base = AutoModelForCausalLM.from_pretrained('{base}', dtype='auto', device_map='auto')",
        f"tok  = AutoTokenizer.from_pretrained('{base}')",
        f"av = PeftModel.from_pretrained(base, '{weights_repo or 'REPO'}', "
        f"subfolder='{rp('av_local/iter_%07d' % av_step)}')",
        "```", ""]
    if results_repo:
        lines += ["## Results & datacard", "",
                  f"Held-out results, paired contrasts, qualitative samples and the datacard live in "
                  f"the dataset repo [`{results_repo}`]"
                  f"(https://huggingface.co/datasets/{results_repo}) under `results/sft_control_sweep/`."]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True, help="3-tap sweep dir (expects test/ subdir)")
    p.add_argument("--arl24-dir", help="L24-only-AR test dir ($EVALC/test_arL24), optional")
    p.add_argument("--sweep-dir", help="built-datasets dir ($SWEEP) for row counts, optional")
    p.add_argument("--weights-repo", help="HF model repo id (weights pointer in the card), optional")
    p.add_argument("--results-repo", help="HF dataset repo id (results pointer in the card), optional")
    p.add_argument("--model-card-out", help="also write the WEIGHTS-repo model card (README.md) here")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    card = build_card(args.eval_dir, args.arl24_dir, args.sweep_dir, args.weights_repo,
                      args.results_repo, args.n_boot, args.seed)
    Path(args.out).write_text(card + "\n")
    print(card)
    print(f"\n[datacard] -> {args.out}")
    if args.model_card_out:
        mc = build_model_card(args.weights_repo, args.results_repo)
        Path(args.model_card_out).write_text(mc + "\n")
        print(f"[model-card] -> {args.model_card_out}")


if __name__ == "__main__":
    main()
