"""Experiment #3 — anisotropy / massive-activation audit (read-only, pure numpy).

THE QUESTION. The §7 sweep's loss/FVE/AR-gold all live in the √d-normalized space
    u = sqrt(d) * a / ||a||          (== nla.schema.normalize_activation(a, sqrt(d)),
                                        applied to BOTH pred and gold before MSE)
and the FVE denominator is, exactly,
    baseline = E_rows E_dims (u - mean_u)^2   == trace(Cov(u)) / d
    (this is compute_predict_mean_baselines(...)[1] = mse_rawvar, the value
     evaluate_e2e.py:368 and eval_ar_gold use). So
        FVE = 1 - MSE_model / (trace(Cov(u))/d).
If a HANDFUL of coordinates (Qwen attention-sink / "massive activation" dims) or a
handful of eigen-directions carry most of trace(Cov(u)), then both the FVE numerator
and denominator are dominated by those few giant coordinates, and the whole metric is
mostly reporting how well the AR reproduces semantically-inert giant coordinates. This
is UPSTREAM of experiments #1/#2/#4 — if it fires, those should be re-read in a whitened
/ dim-clipped space.

WHAT THIS DOES (no model, no GPU — numpy + pyarrow only):
  1. Loads activation_L{layer} (default 24 == activation_centre) from the L19-29 bank.
  2. Built-in sanity gates (catch a broken loader BEFORE any conclusion):
       - ||u|| == sqrt(d) for every row (the normalization identity);
       - E_dims E_rows u_i^2 == 1.0 EXACTLY (||u||^2 = d by construction);
       - reproduces the FVE denominator and (for L24) compares to the LOCKED
         published baseline 0.5630 — a hard check that the column == the L24 the
         sweep actually scored.
  3. Per-coordinate variance decomposition of Cov(u) (the denominator, coord basis):
     how many coords carry 50/90/99% of trace(Cov(u)); top dims & their share.
     Same in RAW a-space, to show what the row-norm does to the massive dims.
  4. Massive-activation ID: top dims of E[a_i^2] in RAW space, and where their
     variance goes after normalization (does row-norm neutralize them?).
  5. Eigenbasis of Cov(u): cumulative eigen-share, participation ratio (effective
     rank), top-direction share, and its alignment with the mean direction mu_u.
  6. Per-row and per-doc residual ||u-mu||^2/d distribution: do a few outlier
     rows/docs dominate the denominator (a different concentration axis than dims)?
  7. A PRE-REGISTERED decision rule (printed): whiten/clip-and-re-read vs proceed.

DECISION RULE (pre-registered, read off the eigenbasis of Cov(u), the loss space):
    Let f5 = share of trace(Cov(u)) in the top-5 eigen-directions, and PR the
    participation ratio (Sum lambda)^2 / Sum lambda^2.
      * CONCENTRATED  (f5 > 0.50, i.e. <=5 directions are a majority of the
        denominator)  -> the FVE numbers are dominated by a few directions. REFRAME:
        re-run #1/#2/#4 in a whitened (Sigma^{-1/2}) or top-dim-clipped space and
        re-read every FVE. Report which dims/directions before whitening.
      * DIFFUSE  (f5 < 0.25 and PR > d/8)  -> row-normalization has already largely
        neutralized the massive-activation concern; FVE is a broad measure; proceed
        to #1/#2/#4 as-is.
      * In between -> AMBIGUOUS; report both and prefer the whitened re-read for the
        headline contrasts (cheap insurance), but do not relabel the locked numbers.

This is a DIAGNOSTIC on the existing bank. It does NOT retrain, does NOT touch the
locked sweep numbers, and writes only its own JSON.

Run (H200, where the bank lives):
  python -m multilayer_nla.variance_audit \
      --bank $REGEN --layer 24 --out-json $DATA/variance_audit_L24.json
  # cross-check the denominator against the literal sweep eval split:
  python -m multilayer_nla.variance_audit \
      --bank $SWEEP/rl_test_local.parquet --centre-col --out-json /tmp/cal.json
  # preflight the math on synthetic data (no bank needed):
  python -m multilayer_nla.variance_audit --selfcheck
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# The published, LOCKED predict-the-mean baselines (mse_scale=64 units), from
# EXPERIMENT_REPORT.md §i / the rl_test summaries. Used ONLY as a calibration target
# (does our loaded L24 column reproduce the number the sweep actually scored?).
LOCKED_BASELINE = {"prev": 0.5970, "centre": 0.5630, "next": 0.5663}
LOCKED_LAYER = {23: "prev", 24: "centre", 25: "next"}

# Pre-registered decision-rule thresholds (eigenbasis of Cov(u)).
CONCENTRATED_F5 = 0.50   # top-5 eigendir share above this -> CONCENTRATED
DIFFUSE_F5 = 0.25        # and PR > d/8 -> DIFFUSE
DIFFUSE_PR_FRAC = 1.0 / 8.0


# ----------------------------------------------------------------------------
# Core math — identical to nla.schema.normalize_activation(a, sqrt(d)).
# ----------------------------------------------------------------------------

def normalize_sqrt_d(a: np.ndarray) -> np.ndarray:
    """u = sqrt(d) * a / ||a||  (row-wise; norm in fp64). The loss space."""
    a = np.asarray(a, dtype=np.float64)
    d = a.shape[1]
    norm = np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    return math.sqrt(d) * a / norm


def _cumulative_counts(shares_desc: np.ndarray, thresholds=(0.5, 0.9, 0.99)) -> dict:
    """#components (in descending-share order) to reach each cumulative threshold."""
    cum = np.cumsum(shares_desc)
    return {f"k_for_{int(t * 100)}pct": int(np.searchsorted(cum, t) + 1) for t in thresholds}


def participation_ratio(eigvals: np.ndarray) -> float:
    """(Sum l)^2 / Sum l^2 — effective number of dimensions (1=rank-1, d=isotropic)."""
    s1 = float(eigvals.sum())
    s2 = float((eigvals ** 2).sum())
    return (s1 * s1 / s2) if s2 > 0 else float("nan")


# ----------------------------------------------------------------------------
# The audit (pure given arrays — unit-testable, no IO).
# ----------------------------------------------------------------------------

def audit_activations(a: np.ndarray, doc_ids: list[str] | None, *,
                      layer: int | None, do_eigh: bool = True,
                      top_k: int = 15) -> dict[str, Any]:
    """Decompose Cov(u) (and raw a) and locate concentration. `a` is RAW [N, d]."""
    a = np.asarray(a, dtype=np.float64)
    N, d = a.shape
    sqrt_d = math.sqrt(d)
    res: dict[str, Any] = {"n_rows": int(N), "d_model": int(d), "mse_scale": sqrt_d,
                           "layer": layer}

    # --- raw-norm distribution (massive activations -> heavy right tail) ---
    raw_norm = np.linalg.norm(a, axis=1)
    res["raw_norm"] = {
        "mean": float(raw_norm.mean()), "std": float(raw_norm.std()),
        "p1": float(np.percentile(raw_norm, 1)), "p50": float(np.percentile(raw_norm, 50)),
        "p99": float(np.percentile(raw_norm, 99)), "max": float(raw_norm.max()),
    }

    # --- normalize + by-construction identities (broken-loader detectors) ---
    u = normalize_sqrt_d(a)
    u_norm = np.linalg.norm(u, axis=1)
    assert np.allclose(u_norm, sqrt_d, rtol=1e-4, atol=1e-3), (
        f"||u|| != sqrt(d): mean {u_norm.mean():.4f} vs {sqrt_d:.4f} — bad column/reshape/dtype")
    elem_ms = float((u ** 2).mean())            # == 1.0 EXACTLY (||u||^2 = d)
    assert abs(elem_ms - 1.0) < 1e-6, f"E[u_i^2]={elem_ms:.6f} != 1 — normalization/reshape wrong"

    # --- FVE denominator = trace(Cov(u))/d = mse_rawvar (the locked baseline) ---
    mu = u.mean(axis=0)                          # [d]
    centred = u - mu
    var_i = (centred ** 2).mean(axis=0)          # [d] per-coord variance; sum/d = baseline
    baseline_rawvar = float(var_i.mean())        # == compute_predict_mean_baselines[1]
    mu_normed = normalize_sqrt_d(mu[None, :])[0]
    baseline_meannorm = float(((u - mu_normed) ** 2).mean())   # the [0] variant (context)
    mean_dir_energy = float((mu ** 2).sum() / d)               # ||mu||^2/d = 1 - baseline_rawvar
    res["fve_denominator"] = {
        "baseline_rawvar": baseline_rawvar,            # the number FVE divides by
        "baseline_meannorm": baseline_meannorm,
        "mean_direction_energy_frac": mean_dir_energy, # how much "energy" is the shared mean
        "elementwise_meansq_u": elem_ms,               # == 1.0 (sanity)
        "identity_check_sum_var_over_d": float(var_i.sum() / d),  # == baseline_rawvar
    }
    # calibration vs the LOCKED published baseline (only meaningful for L23/24/25)
    if layer in LOCKED_LAYER:
        nm = LOCKED_LAYER[layer]
        locked = LOCKED_BASELINE[nm]
        res["fve_denominator"]["locked_baseline"] = locked
        res["fve_denominator"]["locked_name"] = nm
        res["fve_denominator"]["abs_gap_vs_locked"] = abs(baseline_rawvar - locked)

    # --- per-coordinate variance shares: Cov(u) (loss space) and raw a ---
    def coord_block(var_vec: np.ndarray, second_moment: np.ndarray) -> dict:
        order = np.argsort(var_vec)[::-1]
        shares = var_vec[order] / max(var_vec.sum(), 1e-30)
        out = {
            "total": float(var_vec.sum()),
            **_cumulative_counts(shares),
            "top_dims": [int(i) for i in order[:top_k]],
            "top_dim_var_share": [float(s) for s in shares[:top_k]],
            "top5_share": float(shares[:5].sum()),
            "top1_share": float(shares[0]),
        }
        # second moment (for raw: this is where attention-sink dims live)
        som_order = np.argsort(second_moment)[::-1]
        som_shares = second_moment[som_order] / max(second_moment.sum(), 1e-30)
        out["top_dims_by_2nd_moment"] = [int(i) for i in som_order[:top_k]]
        out["top_2nd_moment_share"] = [float(s) for s in som_shares[:top_k]]
        return out

    res["coord_var_normalized"] = coord_block(var_i, (u ** 2).mean(axis=0))
    a_var = a.var(axis=0)
    res["coord_var_raw"] = coord_block(a_var, (a ** 2).mean(axis=0))

    # --- massive-activation tracking: raw-massive dims -> their u-variance share ---
    raw_2nd = (a ** 2).mean(axis=0)
    massive = np.argsort(raw_2nd)[::-1][:top_k]
    u_var_share = var_i / max(var_i.sum(), 1e-30)
    res["massive_activation_dims"] = {
        "dims": [int(i) for i in massive],
        "raw_2nd_moment_share": [float(raw_2nd[i] / raw_2nd.sum()) for i in massive],
        "u_variance_share": [float(u_var_share[i]) for i in massive],
        "raw_massive_total_2nd_share": float(raw_2nd[massive].sum() / raw_2nd.sum()),
        "same_dims_u_variance_total_share": float(u_var_share[massive].sum()),
        "neutralized_by_rownorm": bool(u_var_share[massive].sum() < 0.5 * (raw_2nd[massive].sum() / raw_2nd.sum())),
    }

    # --- eigenbasis of Cov(u) (the loss-relevant rotation) ---
    if do_eigh:
        cov = (centred.T @ centred) / N            # [d, d] fp64
        eigvals = np.linalg.eigvalsh(cov)          # ascending
        eigvals = np.clip(eigvals[::-1], 0, None)  # descending, clip tiny negatives
        eig_share = eigvals / max(eigvals.sum(), 1e-30)
        pr = participation_ratio(eigvals)
        # alignment of the top eigenvector with the mean direction
        # (recompute the leading eigvec cheaply via power iteration on cov)
        v = mu / max(np.linalg.norm(mu), 1e-12)
        for _ in range(50):
            v = cov @ v
            v = v / max(np.linalg.norm(v), 1e-12)
        top_eigvec_mu_cos = float(abs(v @ (mu / max(np.linalg.norm(mu), 1e-12))))
        res["eigen"] = {
            "participation_ratio": pr,
            "participation_ratio_frac_of_d": float(pr / d),
            **{f"eig_{k}": v for k, v in _cumulative_counts(eig_share).items()},
            "top1_share": float(eig_share[0]),
            "top5_share": float(eig_share[:5].sum()),
            "top10_share": float(eig_share[:10].sum()),
            "top_eigvec_mean_dir_cos": top_eigvec_mu_cos,
            "trace": float(eigvals.sum()),
        }
    else:
        res["eigen"] = None

    # --- per-row & per-doc residual concentration (a different axis) ---
    row_res = (centred ** 2).mean(axis=1)          # [N]; mean == baseline_rawvar
    order = np.argsort(row_res)[::-1]
    res["row_residual"] = {
        "mean": float(row_res.mean()), "p50": float(np.percentile(row_res, 50)),
        "p99": float(np.percentile(row_res, 99)), "max": float(row_res.max()),
        "top1pct_share_of_total": float(row_res[order[:max(1, N // 100)]].sum() / row_res.sum()),
    }
    if doc_ids is not None and len(doc_ids) == N:
        by_doc: dict[str, list[float]] = {}
        for r, did in zip(row_res, doc_ids):
            by_doc.setdefault(did, []).append(float(r))
        doc_means = np.array([np.mean(v) for v in by_doc.values()])
        res["doc_residual"] = {
            "n_docs": int(len(by_doc)),
            "mean": float(doc_means.mean()), "p99": float(np.percentile(doc_means, 99)),
            "max": float(doc_means.max()),
            "top1pct_docs_share": float(
                np.sort(doc_means)[::-1][:max(1, len(doc_means) // 100)].sum() / doc_means.sum()),
        }

    # --- pre-registered verdict ---
    f5 = res["eigen"]["top5_share"] if res["eigen"] else res["coord_var_normalized"]["top5_share"]
    pr_frac = res["eigen"]["participation_ratio_frac_of_d"] if res["eigen"] else None
    if f5 > CONCENTRATED_F5:
        verdict = "CONCENTRATED"
    elif f5 < DIFFUSE_F5 and (pr_frac is not None and pr_frac > DIFFUSE_PR_FRAC):
        verdict = "DIFFUSE"
    else:
        verdict = "AMBIGUOUS"
    res["verdict"] = verdict
    res["verdict_inputs"] = {"top5_eigen_share": float(f5), "pr_frac_of_d": pr_frac,
                             "thresholds": {"concentrated_f5": CONCENTRATED_F5,
                                            "diffuse_f5": DIFFUSE_F5, "diffuse_pr_frac": DIFFUSE_PR_FRAC}}
    return res


# ----------------------------------------------------------------------------
# Bank loading (pyarrow, lazy) — supports a single parquet or a dir of shards.
# ----------------------------------------------------------------------------

def _list_parquets(path: str) -> list[str]:
    p = Path(path)
    if p.is_dir():
        files = sorted(str(x) for x in p.glob("*.parquet"))
        assert files, f"no *.parquet under {path}"
        return files
    return [str(p)]


def load_layer(bank: str, layer: int | None, *, centre_col: bool,
               max_rows: int | None) -> tuple[np.ndarray, list[str]]:
    """Load activation_L{layer} (or activation_centre if --centre-col) + doc_id.

    centre_col targets the SWEEP parquets ($SWEEP/rl_*_*.parquet) whose centre is
    activation_centre == L24 — used to reproduce the locked baseline exactly. The
    raw bank ($REGEN) stores activation_L{k}.
    """
    import pyarrow.parquet as pq
    col = "activation_centre" if centre_col else f"activation_L{layer}"
    acts, dids, n = [], [], 0
    for fp in _list_parquets(bank):
        pf = pq.ParquetFile(fp)
        names = pf.schema_arrow.names
        assert col in names, f"{fp}: column {col!r} not present (have {names[:8]}...)"
        has_doc = "doc_id" in names
        cols = [col] + (["doc_id"] if has_doc else [])
        for rg_idx in range(pf.num_row_groups):
            if max_rows is not None and n >= max_rows:
                break
            rg = pf.read_row_group(rg_idx, columns=cols)
            take = rg.num_rows if max_rows is None else min(max_rows - n, rg.num_rows)
            rg = rg.slice(0, take)
            c = rg.column(col).combine_chunks()
            acts.append(c.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(c), -1))
            dids.extend(rg.column("doc_id").to_pylist() if has_doc else [None] * take)
            n += take
        if max_rows is not None and n >= max_rows:
            break
    A = np.concatenate(acts, axis=0)
    doc_ids = dids if any(x is not None for x in dids) else None
    return A, doc_ids


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def _print_report(res: dict[str, Any]) -> None:
    d = res["d_model"]
    fd = res["fve_denominator"]
    print("=" * 74)
    print(f"VARIANCE AUDIT (#3)  layer={res['layer']}  N={res['n_rows']}  d={d}  "
          f"mse_scale=sqrt(d)={res['mse_scale']:.3f}")
    rn = res["raw_norm"]
    print(f"  raw ||a||: mean {rn['mean']:.2f}  p1/p50/p99 {rn['p1']:.1f}/{rn['p50']:.1f}/{rn['p99']:.1f}"
          f"  max {rn['max']:.1f}   (heavy right tail => massive activations)")
    print("-" * 74)
    print("FVE denominator (the number FVE divides by), loss space u=sqrt(d)a/||a||:")
    print(f"  baseline = trace(Cov(u))/d = {fd['baseline_rawvar']:.4f}   "
          f"(sum var_i/d check {fd['identity_check_sum_var_over_d']:.4f})")
    print(f"  mean-direction energy ||mu||^2/d = {fd['mean_direction_energy_frac']:.4f}  "
          f"(= 1 - baseline; the shared common component)")
    if "locked_baseline" in fd:
        print(f"  LOCKED published baseline ({fd['locked_name']}) = {fd['locked_baseline']:.4f}  "
              f"|gap| = {fd['abs_gap_vs_locked']:.4f}  "
              + ("OK" if fd["abs_gap_vs_locked"] < 0.03 else "** MISMATCH: wrong column/split? **"))
    print("-" * 74)
    cn, cr = res["coord_var_normalized"], res["coord_var_raw"]
    print("Per-coordinate variance concentration (how many of d coords = the denominator):")
    print(f"  normalized u : 50%<= {cn['k_for_50pct']}  90%<= {cn['k_for_90pct']}  "
          f"99%<= {cn['k_for_99pct']} coords   top5 {cn['top5_share']*100:.1f}%  top1 {cn['top1_share']*100:.1f}%")
    print(f"  raw a        : 50%<= {cr['k_for_50pct']}  90%<= {cr['k_for_90pct']}  "
          f"99%<= {cr['k_for_99pct']} coords   top5 {cr['top5_share']*100:.1f}%  top1 {cr['top1_share']*100:.1f}%")
    ma = res["massive_activation_dims"]
    print(f"  massive dims (top raw 2nd-moment): {ma['dims'][:8]}")
    print(f"     raw 2nd-moment share {ma['raw_massive_total_2nd_share']*100:.1f}%  ->  "
          f"u-variance share {ma['same_dims_u_variance_total_share']*100:.1f}%   "
          + ("(row-norm NEUTRALIZES them)" if ma["neutralized_by_rownorm"] else "(still dominant after norm)"))
    if res["eigen"]:
        e = res["eigen"]
        print("-" * 74)
        print("Eigenbasis of Cov(u) (the loss-space rotation):")
        print(f"  participation ratio {e['participation_ratio']:.1f} / {d}  "
              f"({e['participation_ratio_frac_of_d']*100:.1f}% of d)   "
              f"top1 {e['top1_share']*100:.1f}%  top5 {e['top5_share']*100:.1f}%  top10 {e['top10_share']*100:.1f}%")
        print(f"  dirs for 50/90/99%: {e['eig_k_for_50pct']}/{e['eig_k_for_90pct']}/{e['eig_k_for_99pct']}"
              f"   top-eigvec·mean-dir cos {e['top_eigvec_mean_dir_cos']:.3f}")
    rr = res["row_residual"]
    print("-" * 74)
    print(f"Row residual ||u-mu||^2/d: p50 {rr['p50']:.3f}  p99 {rr['p99']:.3f}  max {rr['max']:.3f}  "
          f"top-1% rows = {rr['top1pct_share_of_total']*100:.1f}% of denom")
    if "doc_residual" in res:
        dr = res["doc_residual"]
        print(f"Doc residual (n_docs {dr['n_docs']}): top-1% docs = {dr['top1pct_docs_share']*100:.1f}% of denom")
    print("=" * 74)
    v = res["verdict"]
    vi = res["verdict_inputs"]
    print(f"VERDICT: {v}   (top-5 eigendir share {vi['top5_eigen_share']*100:.1f}%, "
          f"PR/d {('%.1f%%' % (vi['pr_frac_of_d']*100)) if vi['pr_frac_of_d'] else 'n/a'})")
    if v == "CONCENTRATED":
        print("  -> FVE is dominated by a few directions. RE-READ #1/#2/#4 in a whitened")
        print("     (Sigma^-1/2) or top-dim-clipped space; the locked numbers stay as-is.")
    elif v == "DIFFUSE":
        print("  -> row-norm has neutralized the massive-activation concern; FVE is broad.")
        print("     Proceed to #1/#2/#4 as-is.")
    else:
        print("  -> AMBIGUOUS; prefer a whitened re-read for the headline contrasts (cheap),")
        print("     but do not relabel the locked numbers.")
    print("=" * 74)


# ----------------------------------------------------------------------------
# Synthetic self-check — validates the math + that the audit detects a planted
# massive dim and planted anisotropy. Pure numpy; no bank, no torch, no GPU.
# ----------------------------------------------------------------------------

def selfcheck() -> None:
    print("[selfcheck] identities + planted massive-dim/anisotropy detection")
    rng = np.random.default_rng(0)
    d, N = 256, 8000
    doc_ids = [f"doc{i // 10}" for i in range(N)]

    # (1) ISOTROPIC raw Gaussian -> after row-norm, ~isotropic on the sphere:
    #     no single coord/eigendir should dominate; verdict DIFFUSE.
    a_iso = rng.standard_normal((N, d)).astype(np.float32)
    r_iso = audit_activations(a_iso, doc_ids, layer=None, top_k=10)
    assert abs(r_iso["fve_denominator"]["elementwise_meansq_u"] - 1.0) < 1e-6
    assert abs(r_iso["fve_denominator"]["identity_check_sum_var_over_d"]
               - r_iso["fve_denominator"]["baseline_rawvar"]) < 1e-9
    assert r_iso["verdict"] == "DIFFUSE", r_iso["verdict"]

    # (2) Planted MASSIVE dim 0 (huge, near-constant) + a few high-variance dims:
    #     raw 2nd-moment is dominated by dim 0; after row-norm, dim 0 becomes nearly
    #     CONSTANT (low variance) -> a sharp illustration that row-norm can move where
    #     the variance lives. We assert the audit reports dim 0 as the top raw 2nd
    #     moment and that the by-construction identities still hold.
    a_mass = rng.standard_normal((N, d)).astype(np.float32)
    a_mass[:, 0] += 200.0  # attention-sink-like massive, near-constant across rows
    a_mass[:, 1:4] *= 8.0  # a few genuinely high-variance coords
    r_mass = audit_activations(a_mass, doc_ids, layer=None, top_k=10)
    assert r_mass["massive_activation_dims"]["dims"][0] == 0, "should flag dim 0 as massive"
    assert r_mass["massive_activation_dims"]["raw_massive_total_2nd_share"] > 0.5
    # row-norm sends the near-constant massive dim to ~0 variance:
    assert r_mass["coord_var_normalized"]["top_dims"][0] in (1, 2, 3), \
        "post-norm variance should be led by the genuinely-varying coords, not the sink"
    assert abs(r_mass["fve_denominator"]["elementwise_meansq_u"] - 1.0) < 1e-6

    # (3) Planted ANISOTROPY: signal in a low-rank subspace -> CONCENTRATED.
    k = 3
    basis = rng.standard_normal((k, d))
    coeff = rng.standard_normal((N, k)) * np.array([40.0, 30.0, 25.0])
    a_aniso = (coeff @ basis + 0.05 * rng.standard_normal((N, d))).astype(np.float32)
    r_aniso = audit_activations(a_aniso, doc_ids, layer=None, top_k=10)
    assert r_aniso["verdict"] == "CONCENTRATED", r_aniso["verdict"]
    assert r_aniso["eigen"]["participation_ratio"] < d / 4

    print(f"[selfcheck] iso PR/d={r_iso['eigen']['participation_ratio_frac_of_d']:.2f} ({r_iso['verdict']}) | "
          f"massive raw-share={r_mass['massive_activation_dims']['raw_massive_total_2nd_share']:.2f} | "
          f"aniso PR={r_aniso['eigen']['participation_ratio']:.1f} ({r_aniso['verdict']})")
    print("[selfcheck] PASS")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", help="raw L19-29 bank parquet or dir of shards ($REGEN), "
                                   "or a sweep parquet with --centre-col")
    p.add_argument("--layer", type=int, default=24, help="bank layer to audit (activation_L{layer})")
    p.add_argument("--centre-col", action="store_true",
                   help="read activation_centre (==L24) instead of activation_L{layer}; "
                        "use on $SWEEP/rl_*_*.parquet to reproduce the LOCKED baseline exactly")
    p.add_argument("--max-rows", type=int, default=None, help="cap rows (smoke / memory)")
    p.add_argument("--no-eigh", dest="do_eigh", action="store_false", default=True,
                   help="skip the O(d^3) eigendecomposition (coord-basis only)")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--out-json", default=None)
    p.add_argument("--selfcheck", action="store_true", help="synthetic validation, then exit")
    args = p.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    assert args.bank, "--bank required (or --selfcheck)"
    layer = 24 if args.centre_col else args.layer
    A, doc_ids = load_layer(args.bank, layer, centre_col=args.centre_col, max_rows=args.max_rows)
    res = audit_activations(A, doc_ids, layer=layer, do_eigh=args.do_eigh, top_k=args.top_k)
    res["bank"] = args.bank
    _print_report(res)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(res, indent=2))
        print(f"[results] -> {args.out_json}")


if __name__ == "__main__":
    main()
