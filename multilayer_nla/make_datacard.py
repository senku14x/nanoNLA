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
    rows = []
    for name in wanted:
        p = sweep_dir / f"{name}.parquet"
        if p.exists():
            try:
                n = pq.ParquetFile(p).metadata.num_rows
                rows.append(f"| `{name}.parquet` | {n:,} |")
            except Exception:
                rows.append(f"| `{name}.parquet` | (unreadable) |")
    if not rows:
        return ""
    return ("\n## Datasets (built by `build_sweep.py`; raw vectors, `norm=\"none\"`)\n\n"
            "All AV-input columns are positional `av_in_*`; AR targets are the fixed\n"
            "`activation_prev/centre/next` (== L23/L24/L25) — distinct names so the target\n"
            "cannot follow the input. The condition lives in the data, not a train-time flag.\n\n"
            "| parquet | rows |\n| --- | ---: |\n" + "\n".join(rows) + "\n")


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
               n_boot=2000, seed=0):
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
    if weights_repo:
        parts += [f"\n## Weights & raw artifacts\n",
                  f"- Weights (LoRA adapters): shared AR + per-condition AV under `{weights_repo}`.",
                  "- Raw per-condition summaries: `test/test_<cond>.json`; per-example: `test/test_<cond>.jsonl`.",
                  "- Full analysis (distributions, qualitative samples, leakage): `analysis.md`"
                  + (", `analysis_arL24.md`" if arl24_dir else "") + ".",
                  "- Headline table: `result_table.md`."]
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True, help="3-tap sweep dir (expects test/ subdir)")
    p.add_argument("--arl24-dir", help="L24-only-AR test dir ($EVALC/test_arL24), optional")
    p.add_argument("--sweep-dir", help="built-datasets dir ($SWEEP) for row counts, optional")
    p.add_argument("--weights-repo", help="HF repo id pointer string for the card, optional")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    card = build_card(args.eval_dir, args.arl24_dir, args.sweep_dir, args.weights_repo,
                      args.n_boot, args.seed)
    Path(args.out).write_text(card + "\n")
    print(card)
    print(f"\n[datacard] -> {args.out}")


if __name__ == "__main__":
    main()
