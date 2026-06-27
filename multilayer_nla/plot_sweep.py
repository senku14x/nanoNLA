"""FVE-vs-AV-input bar plots, one figure per reconstruction target.

For each condition (= AV input config) draws the held-out TEST end-to-end FVE with its 95%
bootstrap CI, against the AR-gold ceiling and the predict-the-mean baseline (FVE=0). Two
figures, identical axes, so the multi-layer-input effect is comparable across targets:

  * target [L23,L24,L25]  (3-tap)  <- <eval-dir>/test/test_<cond>.json
  * target [L24]          (1-tap)  <- <eval-dir>/test_arL24/test_<cond>.json

Conditions are ordered single-layer first, then multi-layer by span, so the "diversity +
span help" ladder reads left-to-right. Bars: success-only FVE (fve_overall); whiskers: 95%
CI over documents; dashed red: AR-gold ceiling (the reconstructor's headroom).

  python -m multilayer_nla.plot_sweep --eval-dir $EVALC --out-dir $EVALC/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from multilayer_nla.analyze_sweep import AV_INPUT_LAYERS

# single-layer first, then multi-layer by span — left→right is the "diversity+span" ladder
ORDER = ["single", "duplicate", "local", "s2_19_21_23", "s2_20_22_24", "wide"]
C_MULTI, C_SINGLE, C_CEIL = "#1a73e8", "#9aa0a6", "#d93025"


def _distinct(c):
    return len(set(AV_INPUT_LAYERS[c].split(",")))


def _series(eval_dir, sub):
    """[(cond, fve%, (lo_err, hi_err))] for conditions present under <eval-dir>/<sub>."""
    out = []
    for c in ORDER:
        p = Path(eval_dir) / sub / f"test_{c}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        f = d["fve_overall"] * 100
        ci = d.get("fve_overall_ci95")
        err = ((f - ci[0] * 100, ci[1] * 100 - f) if ci else (0.0, 0.0))
        out.append((c, f, err))
    return out


def _ceiling(eval_dir, key):
    p = Path(eval_dir) / "test" / "ar_gold_test.json"
    return json.loads(p.read_text())[key] * 100 if p.exists() else None


def _draw(ax, data, ceil, target_label):
    xs = range(len(data))
    fves = [f for _, f, _ in data]
    yerr = list(zip(*[e for *_, e in data])) if data else [[], []]
    colors = [C_MULTI if _distinct(c) > 1 else C_SINGLE for c, _, _ in data]
    ax.bar(xs, fves, yerr=yerr, capsize=4, color=colors, edgecolor="black", linewidth=0.6, zorder=3)
    for x, f in zip(xs, fves):
        ax.text(x, f + 0.5, f"{f:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    if ceil is not None:
        ax.axhline(ceil, ls="--", color=C_CEIL, lw=1.4, zorder=4)
        ax.text(len(data) - 0.45, ceil + 0.4, f"AR-gold ceiling = {ceil:.1f}",
                color=C_CEIL, ha="right", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", lw=0.9, zorder=4)
    ax.text(-0.4, 1.2, "predict-the-mean baseline (FVE = 0)", fontsize=8, color="black", va="bottom")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{c}\n[{AV_INPUT_LAYERS[c]}]\n{_distinct(c)} distinct" for c, _, _ in data],
                       fontsize=9)
    ax.set_ylabel("end-to-end FVE (%)  —  higher = better reconstruction")
    ax.set_xlabel("AV input layers (condition)")
    ax.set_title(f"End-to-end reconstruction FVE vs AV input\n"
                 f"reconstruction target = {target_label}   ·   held-out TEST (1,000 docs)   ·   "
                 f"shared frozen AR", fontsize=11)
    top = max([ceil or 0] + fves) + 7
    ax.set_ylim(0, top)
    ax.grid(axis="y", ls=":", alpha=0.4, zorder=0)
    ax.legend(handles=[Patch(facecolor=C_MULTI, edgecolor="black", label="multi-layer input (3 distinct layers)"),
                       Patch(facecolor=C_SINGLE, edgecolor="black", label="single-layer input (1 distinct layer)"),
                       Line2D([0], [0], ls="--", color=C_CEIL, label="AR-gold ceiling (reconstructor headroom)")],
              loc="upper left", fontsize=8, framealpha=0.95)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True, help="dir with test/ and test_arL24/ summaries")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    targets = [
        ("test", "[L23, L24, L25]  (3-tap)", "fve_overall", "fve_vs_input__target_L23_L24_L25.png"),
        ("test_arL24", "[L24]  (single target)", "fve_centre", "fve_vs_input__target_L24.png"),
    ]
    made = []
    panels = []
    for sub, label, ceil_key, fname in targets:
        data = _series(args.eval_dir, sub)
        if not data:
            print(f"[plot] skip {sub} — no summaries")
            continue
        ceil = _ceiling(args.eval_dir, ceil_key)
        fig, ax = plt.subplots(figsize=(10, 6))
        _draw(ax, data, ceil, label)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=args.dpi)
        plt.close(fig)
        made.append(out / fname)
        panels.append((sub, label, ceil_key))
        print(f"[plot] -> {out / fname}")

    # side-by-side combined figure for direct comparison
    if len(panels) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(19, 6), sharey=True)
        for ax, (sub, label, ceil_key) in zip(axes, panels):
            _draw(ax, _series(args.eval_dir, sub), _ceiling(args.eval_dir, ceil_key), label)
        fig.suptitle("Does multi-layer AV input improve verbalization? — same conditions, two reconstruction targets",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        cp = out / "fve_vs_input__BOTH_targets.png"
        fig.savefig(cp, dpi=args.dpi)
        plt.close(fig)
        made.append(cp)
        print(f"[plot] -> {cp}")
    return made


if __name__ == "__main__":
    main()
