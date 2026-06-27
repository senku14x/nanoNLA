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
# Okabe-Ito colourblind-safe palette (the ML-paper standard): blue=highlight, grey=control,
# vermillion=ceiling reference.
C_MULTI, C_SINGLE, C_CEIL = "#0072B2", "#9E9E9E", "#D55E00"


def _set_pub_style():
    """ICLR/NeurIPS-style: serif fonts, despined axes, readable sizes, high-res export."""
    matplotlib.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12,
        "xtick.labelsize": 10.5, "ytick.labelsize": 11, "legend.fontsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9, "axes.axisbelow": True,
    })


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
    ax.bar(xs, fves, yerr=yerr, capsize=3, color=colors, edgecolor="black", linewidth=0.7,
           width=0.7, zorder=3, error_kw={"ecolor": "#222222", "elinewidth": 1.0})
    for x, f in zip(xs, fves):
        ax.text(x, f + 0.6, f"{f:.1f}", ha="center", va="bottom", fontsize=9.5)
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


# ---- grouped-by-target-layer figure (per-tap FVE + AR-gold skyline) + Δ-vs-control panel ----
G_ORDER = ["single", "local", "duplicate", "wide", "s2_19_21_23", "s2_20_22_24"]
G_LABEL = {"single": "single", "local": "local", "duplicate": "duplicate", "wide": "wide",
           "s2_19_21_23": "pre i.", "s2_20_22_24": "pre ii."}
G_COLOR = {"single": "#b3b3b3", "local": "#3b2f7a", "duplicate": "#2c6fa6", "wide": "#1aa39a",
           "s2_19_21_23": "#4fc16b", "s2_20_22_24": "#c2d92b"}
G_TAPS = [("fve_prev", "L23"), ("fve_centre", "L24"), ("fve_next", "L25"), ("fve_overall", "Average")]


def plot_by_target(eval_dir, out_path, with_delta=False, split_label="held-out TEST (1,000 docs)"):
    """Grouped bars: per-target-layer (L23/L24/L25/Average) FVE for each condition + AR-gold
    skyline. If with_delta, add a right panel of paired Δ-vs-`duplicate` (success-only, 95% CI)."""
    from multilayer_nla.analyze_sweep import _join, load_results, paired_bootstrap_diff
    _set_pub_style()
    res = load_results(eval_dir)  # test/ summaries + per-example rows
    conds = [c for c in G_ORDER if c in res]
    if with_delta:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2.4, 1.0]})
    else:
        fig, axL = plt.subplots(figsize=(10.5, 6)); axR = None

    ng, nc = len(G_TAPS), len(conds)
    w = 0.82 / nc
    for ci, c in enumerate(conds):
        s = res[c]["summary"]
        vals = [s[k] * 100 for k, _ in G_TAPS]
        xs = [g + (ci - (nc - 1) / 2) * w for g in range(ng)]
        axL.bar(xs, vals, width=w, color=G_COLOR[c], edgecolor="black", linewidth=0.6,
                label=G_LABEL[c], hatch=("////" if c == "single" else None), zorder=3)
    g = json.loads((Path(eval_dir) / "test" / "ar_gold_test.json").read_text())
    gold = [g["fve_prev"] * 100, g["fve_centre"] * 100, g["fve_next"] * 100, g["fve_overall"] * 100]
    axL.plot(range(ng), gold, "o--", color=C_CEIL, mfc="white", mec=C_CEIL, mew=1.6, lw=1.8, ms=8,
             label="AR-gold", zorder=5)
    axL.set_xticks(range(ng)); axL.set_xticklabels([lbl for _, lbl in G_TAPS])
    axL.set_xlabel("target layer"); axL.set_ylabel("FVE (%)")
    axL.set_ylim(0, 100); axL.set_yticks(range(0, 101, 10))
    axL.grid(axis="y", ls=":", alpha=0.4, zorder=0)
    axL.legend(ncol=2, fontsize=9, loc="upper right", framealpha=0.96)
    axL.set_title(f"End-to-end FVE by target layer  ·  {split_label}\n"
                  f"AV input varies; AR target fixed [L23,L24,L25]; shared frozen AR", fontsize=11)

    if axR is not None:
        dup = res["duplicate"]["rows"]
        items = []
        for c in ["single", "local", "s2_19_21_23", "wide", "s2_20_22_24"]:
            pairs, _, _ = _join(res[c]["rows"], dup)
            d = paired_bootstrap_diff(pairs, tap=None, penalized=False, n_boot=2000, seed=0)
            items.append((c, d["mean_diff"] * 100, d["ci_lo"] * 100, d["ci_hi"] * 100))
        items.sort(key=lambda t: t[1])
        for y, (c, m, lo, hi) in enumerate(items):
            axR.barh(y, m, color=G_COLOR[c], edgecolor="black", linewidth=0.6, zorder=3,
                     xerr=[[m - lo], [hi - m]], error_kw={"ecolor": "#222", "elinewidth": 1.0, "capsize": 3})
            axR.text(hi + 0.12, y, f"{m:+.1f}", va="center", ha="left", fontsize=9)
        axR.axvline(0, color="black", lw=1.0, zorder=4)
        axR.set_yticks(range(len(items))); axR.set_yticklabels([G_LABEL[c] for c, *_ in items])
        axR.set_xlabel("Δ FVE vs duplicate (pp)")
        axR.set_title("Effect size vs the k=3\nredundant control (paired, 95% CI)", fontsize=11)
        axR.grid(axis="x", ls=":", alpha=0.4, zorder=0)
        axR.margins(x=0.18)

    fig.suptitle("We tried to improve NLAs by giving the AV multi-layer input",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] -> {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True, help="dir with test/ and test_arL24/ summaries")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dpi", type=int, default=160)
    args = p.parse_args()
    _set_pub_style()
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

    # grouped-by-target-layer test figure (matches the per-target-layer paper layout) +
    # a Δ-vs-duplicate effect-size variant
    if (Path(args.eval_dir) / "test" / "test_local.json").exists():
        plot_by_target(args.eval_dir, out / "fve_test__by_target_layer.png", with_delta=False)
        made.append(out / "fve_test__by_target_layer.png")
        plot_by_target(args.eval_dir, out / "fve_test__by_target_layer_with_delta.png", with_delta=True)
        made.append(out / "fve_test__by_target_layer_with_delta.png")

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
