"""Headroom probe — the pre-registered NLA-free Gate 0 (plan §5.1, §8, §13).

Decides, before any AV/AR/RL training, whether a coherent three-layer patch
COULD carry information about the local cross-layer update that the center
activation alone does not already determine. If it can't, the whole NLA phase
is dead on arrival and we stop here for ~probe-fitting cost instead of 25-50
GPU-hours/condition.

What it computes, on cached three-layer patches (from extract_multilayer.py):

  Normalized states (target/FVE space only — §4 Rev 2):  u^(j) = sqrt(d) a^(j)/||a^(j)||
  Local updates:   delta_lo = u^(l) - u^(l-1),   delta_hi = u^(l+1) - u^(l)
  Update target:   delta = [delta_lo, delta_hi]   (predicted from center u^(l))

  FVE_delta = 1 - E||pred - delta||^2 / E||delta - delta_bar||^2     (§8; floor = mean update)

  Four center-only predictors of delta from u^(l):
    (i)   constant delta_bar  -> FVE_delta = 0 EXACTLY (the floor; built-in check)
    (ii)  ridge (best linear)
    (iii) small MLP (best cheap nonlinear)   [torch; skipped w/ a warning if unavailable]
    (iv)  true neighbor       -> FVE_delta = 1 EXACTLY (the ceiling; built-in check)

  GATE 0 (Rev 2, §13):  H_unique = 1 - max(FVE_ridge, FVE_mlp)   on the DEV split.
    STOP/PIVOT iff H_unique < 0.05  (center already determines neighbors).
    The MLP-ridge gap is reported as a DIAGNOSTIC (is the recoverable part
    nonlinear?), NOT the gate.

If torch is missing, the probe falls back to ridge-only. Since the MLP can only
RAISE the best center-only FVE, ridge-only H_unique is an UPPER BOUND on the true
H_unique: a ridge-only STOP is decisive; a ridge-only GO is provisional (flagged).

Math core is numpy (unit-tested in tests/test_fve_math.py); only the MLP needs
torch (lazy-imported), so this module imports under numpy alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Pre-registered Gate-0 threshold on H_unique (plan §5.1.7, §13).
GATE0_HEADROOM_MIN = 0.05
# Tolerance for the built-in by-construction identity checks (ceiling=1, floor=0).
_IDENTITY_TOL = 1e-5


# ----------------------------------------------------------------------------
# Pure-numpy numerical core (importable without torch/pyarrow — see tests/)
# ----------------------------------------------------------------------------

def normalize_sqrt_d(a: np.ndarray) -> np.ndarray:
    """u = sqrt(d) * a / ||a||  (row-wise). Norm in fp64 for precision."""
    a = np.asarray(a, dtype=np.float64)
    d = a.shape[1]
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-12, None)
    return math.sqrt(d) * a / norm


def doc_split_mask(doc_ids: list[str], frac: float, seed: int) -> np.ndarray:
    """Deterministic document-level split. Returns a bool mask, True = selected.

    Hash each doc_id with the seed -> uniform in [0,1); select if < frac. All
    positions of a doc share its doc_id, so the split is doc-level (invariant:
    never split a doc's positions across buckets).
    """
    out = np.zeros(len(doc_ids), dtype=bool)
    for i, did in enumerate(doc_ids):
        h = hashlib.sha256(f"{seed}|{did}".encode()).digest()
        # First 8 bytes -> uint64 -> [0,1)
        u = int.from_bytes(h[:8], "big") / float(1 << 64)
        out[i] = u < frac
    return out


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge with intercept via centering (intercept is NOT regularized).

    Returns (W, x_mean, y_mean) with W solving (Xc^T Xc + lam I) W = Xc^T Yc.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    x_mean = X.mean(axis=0, keepdims=True)
    y_mean = Y.mean(axis=0, keepdims=True)
    Xc = X - x_mean
    Yc = Y - y_mean
    d = Xc.shape[1]
    A = Xc.T @ Xc + lam * np.eye(d)
    B = Xc.T @ Yc
    W = np.linalg.solve(A, B)
    return W, x_mean, y_mean


def ridge_predict(X: np.ndarray, W: np.ndarray, x_mean: np.ndarray, y_mean: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return (X - x_mean) @ W + y_mean


def fve(pred: np.ndarray, target: np.ndarray, floor: np.ndarray) -> float:
    """Fraction of variance explained, normalized against a `floor` predictor.

    FVE = 1 - E||pred - target||^2 / E||target - floor||^2,
    with E the mean over ALL elements (samples x dims). `floor` broadcasts over
    samples (shape [dims] or [1, dims]). With floor == E[target] this is the
    plan's §8 definition (denominator = variance around the mean).
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    floor = np.asarray(floor, dtype=np.float64)
    num = np.mean((pred - target) ** 2)
    den = np.mean((target - floor) ** 2)
    if den <= 0:
        return float("nan")
    return float(1.0 - num / den)


def ridge_fit_cv(X: np.ndarray, Y: np.ndarray, doc_ids: list[str],
                 lam_grid: tuple[float, ...], floor: np.ndarray, seed: int
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pick lam by an internal doc-level train/val split of the TRAIN set, then
    refit on all of TRAIN. Returns (W, x_mean, y_mean, chosen_lam).

    Honest model selection: lam is chosen on a held-out val FOLD of train, never
    on the dev split the gate is read from.
    """
    val_mask = doc_split_mask(doc_ids, frac=0.1, seed=seed + 7919)
    fit_mask = ~val_mask
    if val_mask.sum() == 0 or fit_mask.sum() == 0:
        # Degenerate (tiny/synthetic) — skip CV, use the middle of the grid.
        lam = lam_grid[len(lam_grid) // 2]
        W, xm, ym = ridge_fit(X, Y, lam)
        return W, xm, ym, lam
    best_lam, best_fve = lam_grid[0], -float("inf")
    for lam in lam_grid:
        W, xm, ym = ridge_fit(X[fit_mask], Y[fit_mask], lam)
        pred = ridge_predict(X[val_mask], W, xm, ym)
        f = fve(pred, Y[val_mask], floor)
        if f > best_fve:
            best_fve, best_lam = f, lam
    W, xm, ym = ridge_fit(X, Y, best_lam)
    return W, xm, ym, best_lam


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.clip(np.linalg.norm(a, axis=1), 1e-12, None)
    nb = np.clip(np.linalg.norm(b, axis=1), 1e-12, None)
    return np.sum(a * b, axis=1) / (na * nb)


def geometry_stats(a_prev: np.ndarray, a_centre: np.ndarray, a_next: np.ndarray) -> dict[str, float]:
    """Raw-activation geometry audit (plan §5): adjacent cosines, update norms,
    and per-layer norms. Computed on raw a (cosine/norm are what §5 specifies)."""
    cos_lo = cosine_rows(a_prev, a_centre)
    cos_hi = cosine_rows(a_centre, a_next)
    upd_lo = np.linalg.norm(a_centre - a_prev, axis=1)
    upd_hi = np.linalg.norm(a_next - a_centre, axis=1)
    return {
        "cos_prev_centre_mean": float(cos_lo.mean()), "cos_prev_centre_std": float(cos_lo.std()),
        "cos_centre_next_mean": float(cos_hi.mean()), "cos_centre_next_std": float(cos_hi.std()),
        "update_norm_lo_mean": float(upd_lo.mean()), "update_norm_hi_mean": float(upd_hi.mean()),
        "act_norm_prev_mean": float(np.linalg.norm(a_prev, axis=1).mean()),
        "act_norm_centre_mean": float(np.linalg.norm(a_centre, axis=1).mean()),
        "act_norm_next_mean": float(np.linalg.norm(a_next, axis=1).mean()),
    }


# ----------------------------------------------------------------------------
# MLP predictor (torch, lazy) — the only non-numpy piece.
# ----------------------------------------------------------------------------

def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def mlp_fit_predict(X_tr: np.ndarray, Y_tr: np.ndarray, X_dev: np.ndarray, *,
                    hidden: int = 2048, epochs: int = 30, lr: float = 1e-3,
                    batch_size: int = 4096, seed: int = 0,
                    val_frac: float = 0.05) -> np.ndarray:
    """Small 1-hidden-layer MLP, center u^(l) -> delta. Returns dev predictions
    in ORIGINAL delta space (un-standardized) as numpy [N_dev, 2d].

    Inputs and targets are z-scored on TRAIN for optimization stability; the
    prediction is mapped back so FVE is computed in delta space against
    delta_bar. A small internal val fold picks the best epoch (cheap early stop).
    """
    import torch
    import torch.nn as nn

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Xm, Xs = X_tr.mean(0), X_tr.std(0) + 1e-6
    Ym, Ys = Y_tr.mean(0), Y_tr.std(0) + 1e-6
    Xtr = ((X_tr - Xm) / Xs).astype(np.float32)
    Ytr = ((Y_tr - Ym) / Ys).astype(np.float32)
    Xdv = ((X_dev - Xm) / Xs).astype(np.float32)

    n = Xtr.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, fit_idx = perm[:n_val], perm[n_val:]

    d_in, d_out = Xtr.shape[1], Ytr.shape[1]
    model = nn.Sequential(
        nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.from_numpy(Xtr).to(device)
    Yt = torch.from_numpy(Ytr).to(device)
    fit_t = torch.from_numpy(fit_idx.astype(np.int64)).to(device)
    val_t = torch.from_numpy(val_idx.astype(np.int64)).to(device)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for _ep in range(epochs):
        model.train()
        ep_perm = fit_t[torch.randperm(fit_t.numel(), device=device)]
        for bs in range(0, ep_perm.numel(), batch_size):
            idx = ep_perm[bs:bs + batch_size]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = loss_fn(out, Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(Xt[val_t]), Yt[val_t]).item()
        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_std = model(torch.from_numpy(Xdv).to(device)).cpu().numpy()
    return pred_std.astype(np.float64) * Ys + Ym  # back to delta space


# ----------------------------------------------------------------------------
# Data loading (pyarrow, lazy) + orchestration
# ----------------------------------------------------------------------------

def load_patches(parquet_path: str, max_rows: int | None = None) -> dict[str, Any]:
    """Stream prev/centre/next activations + doc_id from the multi-layer parquet.

    FixedSizeList -> flatten -> numpy reshape (zero-copy-ish; avoids to_pylist on
    millions of floats — same pattern as nla/schema.py and train_sft.load_*)."""
    import pyarrow.parquet as pq

    cols = ["activation_prev", "activation_centre", "activation_next", "doc_id"]
    pf = pq.ParquetFile(parquet_path)
    prev_l, centre_l, next_l, dids = [], [], [], []
    n = 0
    for rg_idx in range(pf.num_row_groups):
        if max_rows is not None and n >= max_rows:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        take = rg.num_rows if max_rows is None else min(max_rows - n, rg.num_rows)
        rg = rg.slice(0, take)

        def to_np(name):
            col = rg.column(name).combine_chunks()
            return (col.flatten().to_numpy(zero_copy_only=False)
                    .astype(np.float32).reshape(len(col), -1))

        prev_l.append(to_np("activation_prev"))
        centre_l.append(to_np("activation_centre"))
        next_l.append(to_np("activation_next"))
        dids.extend(rg.column("doc_id").to_pylist())
        n += take
    return {
        "a_prev": np.concatenate(prev_l, axis=0),
        "a_centre": np.concatenate(centre_l, axis=0),
        "a_next": np.concatenate(next_l, axis=0),
        "doc_ids": dids,
    }


def compute_headroom(a_prev: np.ndarray, a_centre: np.ndarray, a_next: np.ndarray,
                     doc_ids: list[str], *, dev_frac: float = 0.2, seed: int = 0,
                     ridge_lam_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0),
                     use_mlp: bool = True, mlp_kwargs: dict | None = None,
                     geometry_sample: int = 20000) -> dict[str, Any]:
    """Run the full Gate-0 computation. Pure given arrays; no IO. Returns a
    results dict including the verdict. Raises if a by-construction identity
    (ceiling FVE=1, floor FVE=0, sqrt(d)-norm) fails — caught before it can
    poison downstream numbers (plan §5.1.6)."""
    d = a_centre.shape[1]
    sqrt_d = math.sqrt(d)

    u_prev = normalize_sqrt_d(a_prev)
    u_centre = normalize_sqrt_d(a_centre)
    u_next = normalize_sqrt_d(a_next)

    # Sanity: sqrt(d) normalization (plan §6.3 / §5.1.6).
    for nm, u in (("prev", u_prev), ("centre", u_centre), ("next", u_next)):
        norms = np.linalg.norm(u, axis=1)
        assert np.allclose(norms, sqrt_d, rtol=1e-4, atol=1e-3), (
            f"u_{nm} not sqrt(d)-normalized: mean ||u||={norms.mean():.4f}, sqrt(d)={sqrt_d:.4f}"
        )

    delta_lo = u_centre - u_prev
    delta_hi = u_next - u_centre
    delta = np.concatenate([delta_lo, delta_hi], axis=1)   # [N, 2d]
    X = u_centre                                            # center-only input

    dev_mask = doc_split_mask(doc_ids, frac=dev_frac, seed=seed)
    train_mask = ~dev_mask
    n_dev, n_train = int(dev_mask.sum()), int(train_mask.sum())
    assert n_dev > 0 and n_train > 0, (
        f"degenerate split: n_train={n_train}, n_dev={n_dev} (raise --max-rows or lower --dev-frac)"
    )

    delta_bar = delta[train_mask].mean(axis=0, keepdims=True)   # [1, 2d] — the floor

    # --- Built-in identity checks (catch FVE bookkeeping bugs) ---
    fve_constant = fve(np.broadcast_to(delta_bar, delta[dev_mask].shape), delta[dev_mask], delta_bar)
    fve_ceiling = fve(delta[dev_mask], delta[dev_mask], delta_bar)
    assert abs(fve_constant) < _IDENTITY_TOL, (
        f"constant-floor FVE_delta={fve_constant:.2e} != 0 — FVE normalization is wrong"
    )
    assert abs(fve_ceiling - 1.0) < _IDENTITY_TOL, (
        f"true-neighbor ceiling FVE_delta={fve_ceiling:.6f} != 1 — delta bookkeeping is wrong"
    )

    # --- (ii) ridge center-only predictor (best linear) ---
    train_dids = [d_ for d_, m in zip(doc_ids, train_mask) if m]
    Wr, xm, ym, lam = ridge_fit_cv(
        X[train_mask], delta[train_mask], train_dids, ridge_lam_grid, delta_bar, seed,
    )
    fve_ridge = fve(ridge_predict(X[dev_mask], Wr, xm, ym), delta[dev_mask], delta_bar)

    # --- (iii) MLP center-only predictor (best cheap nonlinear) ---
    fve_mlp: float | None = None
    mlp_skipped_reason = None
    if use_mlp and torch_available():
        pred_mlp = mlp_fit_predict(
            X[train_mask], delta[train_mask], X[dev_mask], seed=seed, **(mlp_kwargs or {}),
        )
        fve_mlp = fve(pred_mlp, delta[dev_mask], delta_bar)
    else:
        mlp_skipped_reason = "torch unavailable" if use_mlp else "disabled (--no-mlp)"

    # --- Gate 0 (Rev 2): H_unique = 1 - best center-only FVE_delta on dev ---
    candidates = {"ridge": fve_ridge}
    if fve_mlp is not None:
        candidates["mlp"] = fve_mlp
    best_name = max(candidates, key=candidates.get)
    best_fve = candidates[best_name]
    h_unique = 1.0 - best_fve
    mlp_ridge_gap = (fve_mlp - fve_ridge) if fve_mlp is not None else None

    go = h_unique >= GATE0_HEADROOM_MIN
    provisional = go and (fve_mlp is None)  # ridge-only GO can't see an MLP that might lower H

    # --- State-space context (per-layer center->neighbor, §5 layer selection) ---
    def state_fve_from_center(u_target):
        Ws, sxm, sym, _ = ridge_fit_cv(u_centre[train_mask], u_target[train_mask],
                                       train_dids, ridge_lam_grid,
                                       u_target[train_mask].mean(0, keepdims=True), seed)
        pred = ridge_predict(u_centre[dev_mask], Ws, sxm, sym)
        floor = u_target[train_mask].mean(0, keepdims=True)
        return fve(pred, u_target[dev_mask], floor)

    state_fve = {
        "prev": state_fve_from_center(u_prev),
        "centre": state_fve_from_center(u_centre),  # ~1 (identity); a self-consistency check
        "next": state_fve_from_center(u_next),
    }

    # --- Geometry audit (subsample for speed) ---
    gsel = slice(0, min(geometry_sample, a_centre.shape[0]))
    geometry = geometry_stats(a_prev[gsel], a_centre[gsel], a_next[gsel])

    return {
        "n_total": int(a_centre.shape[0]),
        "n_train": n_train, "n_dev": n_dev, "d_model": d,
        "dev_frac": dev_frac, "split_seed": seed, "ridge_lambda": lam,
        "fve_delta": {
            "constant_floor": fve_constant,    # == 0 by construction
            "ridge": fve_ridge,
            "mlp": fve_mlp,                     # None if skipped
            "true_neighbor_ceiling": fve_ceiling,  # == 1 by construction
        },
        "best_center_only": best_name,
        "best_center_only_fve_delta": best_fve,
        "H_unique": h_unique,
        "mlp_ridge_gap": mlp_ridge_gap,        # diagnostic only (Rev 2)
        "mlp_skipped_reason": mlp_skipped_reason,
        "gate0_threshold": GATE0_HEADROOM_MIN,
        "gate0_verdict": "GO" if go else "STOP",
        "gate0_provisional": provisional,
        "state_fve": state_fve,
        "geometry": geometry,
    }


def _print_report(res: dict[str, Any]) -> None:
    fv = res["fve_delta"]
    g = res["geometry"]
    print("=" * 70)
    print(f"HEADROOM PROBE — Gate 0 (plan §5.1, §13)   d_model={res['d_model']}")
    print(f"  rows: {res['n_total']}  (train {res['n_train']} / dev {res['n_dev']}, "
          f"dev_frac={res['dev_frac']}, seed={res['split_seed']})")
    print("-" * 70)
    print("Geometry (raw activations, plan §5):")
    print(f"  cos(prev,centre) = {g['cos_prev_centre_mean']:.4f} ± {g['cos_prev_centre_std']:.4f}")
    print(f"  cos(centre,next) = {g['cos_centre_next_mean']:.4f} ± {g['cos_centre_next_std']:.4f}")
    print(f"  ||update lo/hi|| = {g['update_norm_lo_mean']:.3f} / {g['update_norm_hi_mean']:.3f}"
          f"   ||a|| p/c/n = {g['act_norm_prev_mean']:.1f}/{g['act_norm_centre_mean']:.1f}/{g['act_norm_next_mean']:.1f}")
    print("-" * 70)
    print("FVE_delta ladder (predict update delta from center u^(l), eval on DEV):")
    print(f"  (i)   constant  delta_bar      = {fv['constant_floor']:+.4f}   (floor, == 0 by construction)")
    print(f"  (ii)  ridge     (best linear)  = {fv['ridge']:+.4f}   [lambda={res['ridge_lambda']:g}]")
    mlp_str = f"{fv['mlp']:+.4f}" if fv['mlp'] is not None else f"SKIPPED ({res['mlp_skipped_reason']})"
    print(f"  (iii) MLP       (nonlinear)    = {mlp_str}")
    print(f"  (iv)  true neighbor (ceiling)  = {fv['true_neighbor_ceiling']:+.4f}   (== 1 by construction)")
    print("-" * 70)
    print("State FVE (center->neighbor, §5 context):  "
          f"prev={res['state_fve']['prev']:+.4f}  centre={res['state_fve']['centre']:+.4f}  "
          f"next={res['state_fve']['next']:+.4f}")
    gap = res["mlp_ridge_gap"]
    print(f"Diagnostic — MLP-ridge gap (nonlinearity): "
          f"{gap:+.4f}" if gap is not None else "Diagnostic — MLP-ridge gap: n/a (MLP skipped)")
    print("=" * 70)
    print(f"  best center-only predictor : {res['best_center_only']} "
          f"(FVE_delta = {res['best_center_only_fve_delta']:+.4f})")
    print(f"  H_unique = 1 - best        : {res['H_unique']:+.4f}   "
          f"(threshold {res['gate0_threshold']})")
    print(f"  GATE 0 VERDICT             : {res['gate0_verdict']}"
          + ("  [PROVISIONAL — MLP skipped; ridge-only H_unique is an upper bound]"
             if res["gate0_provisional"] else ""))
    if res["gate0_verdict"] == "STOP":
        print("  -> center activation already determines the neighbors; nothing for the")
        print("     bottleneck to add. STOP or pivot (plan §13).")
    else:
        print("  -> unique cross-layer headroom exists; 'can language carry it' is live.")
        print("     Proceed to Stage 2 (Conditions A-D one-seed sweep).")
    print("=" * 70)


def run(parquet_path: str, *, out_json: str | None, dev_frac: float, seed: int,
        max_rows: int | None, use_mlp: bool, sidecar: str | None) -> dict[str, Any]:
    data = load_patches(parquet_path, max_rows=max_rows)
    res = compute_headroom(
        data["a_prev"], data["a_centre"], data["a_next"], data["doc_ids"],
        dev_frac=dev_frac, seed=seed, use_mlp=use_mlp,
    )
    # Provenance / manifest (plan §12.6).
    from multilayer_nla.manifest import build_manifest
    extra = {"parquet": parquet_path, "doc_split_seed": seed, "dev_frac": dev_frac}
    if sidecar:
        try:
            import yaml
            sc = yaml.safe_load(Path(sidecar).read_text())
            extra.update({
                "base_model": sc.get("base_model"),
                "layer_triplet": sc.get("layers"),
                "corpus": sc.get("corpus"),
                "corpus_slice": sc.get("corpus_slice"),
                "position_seed": sc.get("seed"),
            })
        except Exception as e:
            print(f"[warn] could not read sidecar {sidecar}: {e}")
    res["manifest"] = build_manifest(stage="stage1_headroom_probe", extra=extra)
    _print_report(res)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(res, indent=2))
        print(f"[results] -> {out_json}")
    return res


# ----------------------------------------------------------------------------
# Self-check on synthetic data — validates the gate end-to-end (ridge path runs
# under numpy alone; MLP asserted only if torch is present).
# ----------------------------------------------------------------------------

def _random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal d x d (Q from QR of a Gaussian). Norm-preserving, so
    R @ u stays on the sqrt(d) sphere -> the probe's internal re-normalization is
    a no-op and the synthetic update is EXACTLY linear in the center."""
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    # Fix sign ambiguity so Q is a proper, deterministic orthogonal matrix.
    return q * np.sign(np.diag(r))


def make_synthetic_patches(rng: np.random.Generator, d: int, N: int, *, independent_next: bool):
    """Build (a_prev, a_centre, a_next) on the sqrt(d) sphere.

    centre: random sphere point. prev: ALWAYS a fixed orthogonal map of centre
    (linear, ridge-recoverable). next: orthogonal map of centre (STOP, fully
    center-determined) OR an independent sphere point (GO, unrecoverable update).
    """
    u_centre = normalize_sqrt_d(rng.standard_normal((N, d)))
    R_lo = _random_orthogonal(d, rng)
    u_prev = u_centre @ R_lo.T                      # = R_lo u_centre, on the sphere
    if independent_next:
        u_next = normalize_sqrt_d(rng.standard_normal((N, d)))   # independent of centre
    else:
        R_hi = _random_orthogonal(d, rng)
        u_next = u_centre @ R_hi.T
    # Tiny noise so the noiseless-linear STOP case isn't exactly singular.
    eps = 1e-3
    return (u_prev + eps * rng.standard_normal((N, d)),
            u_centre,
            u_next + eps * rng.standard_normal((N, d)))


def selfcheck() -> None:
    print("[selfcheck] synthetic GO / STOP cases + by-construction identities")
    rng = np.random.default_rng(0)
    d, N = 64, 4000
    doc_ids = [f"doc{i // 10}" for i in range(N)]  # 10 positions/doc, doc-level split

    # STOP: both neighbors are orthogonal (linear) maps of the center -> the update
    # is exactly linear in the center -> ridge FVE ~ 1 -> H_unique ~ 0 -> STOP.
    a_prev, a_centre, a_next = make_synthetic_patches(rng, d, N, independent_next=False)
    res_stop = compute_headroom(a_prev, a_centre, a_next, doc_ids, seed=1, use_mlp=torch_available())
    assert res_stop["gate0_verdict"] == "STOP", f"expected STOP, got H_unique={res_stop['H_unique']}"
    assert res_stop["H_unique"] < GATE0_HEADROOM_MIN

    # GO: the upper update is an independent sphere point -> not center-recoverable
    # -> ridge FVE bounded well below 1 -> H_unique large -> GO.
    a_prev, a_centre, a_next = make_synthetic_patches(rng, d, N, independent_next=True)
    res_go = compute_headroom(a_prev, a_centre, a_next, doc_ids, seed=1, use_mlp=torch_available())
    assert res_go["gate0_verdict"] == "GO", f"expected GO, got H_unique={res_go['H_unique']}"
    assert res_go["H_unique"] >= GATE0_HEADROOM_MIN

    # Identities are asserted inside compute_headroom; surface them here too.
    assert abs(res_stop["fve_delta"]["constant_floor"]) < _IDENTITY_TOL
    assert abs(res_stop["fve_delta"]["true_neighbor_ceiling"] - 1.0) < _IDENTITY_TOL
    print(f"[selfcheck] STOP H_unique={res_stop['H_unique']:.3f} (ridge FVE "
          f"{res_stop['fve_delta']['ridge']:.3f})  |  GO H_unique={res_go['H_unique']:.3f} "
          f"(ridge FVE {res_go['fve_delta']['ridge']:.3f})  |  MLP "
          f"{'on' if torch_available() else 'off'}")
    print("[selfcheck] PASS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", help="multi-layer patch parquet (from extract_multilayer.py)")
    p.add_argument("--sidecar", default=None, help="*.mlnla_meta.yaml for manifest provenance "
                   "(defaults to {parquet}.mlnla_meta.yaml if present)")
    p.add_argument("--out-json", default=None, help="write the full results+verdict JSON here")
    p.add_argument("--dev-frac", type=float, default=0.2, help="doc-level dev fraction")
    p.add_argument("--seed", type=int, default=0, help="doc-split seed (pre-register & keep fixed)")
    p.add_argument("--max-rows", type=int, default=None, help="cap rows (smoke runs)")
    p.add_argument("--no-mlp", dest="use_mlp", action="store_false", default=True,
                   help="skip the MLP (ridge-only H_unique = upper bound; GO is provisional)")
    p.add_argument("--selfcheck", action="store_true", help="run synthetic GO/STOP validation and exit")
    args = p.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    assert args.parquet, "--parquet required (or use --selfcheck)"
    sidecar = args.sidecar
    if sidecar is None:
        cand = args.parquet + ".mlnla_meta.yaml"
        sidecar = cand if Path(cand).exists() else None
    run(args.parquet, out_json=args.out_json, dev_frac=args.dev_frac, seed=args.seed,
        max_rows=args.max_rows, use_mlp=args.use_mlp, sidecar=sidecar)


if __name__ == "__main__":
    main()
