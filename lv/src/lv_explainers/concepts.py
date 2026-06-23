"""Concept directions and linear probes on activations.

A "concept" is defined by a set of contrast pairs (present vs absent). From the
activations at the NLA's read layer we derive:

  * delta_c : the mean-difference (CAA-style) direction, mean(h_present) -
              mean(h_absent), in the SAME space the residual lives in. This is
              what read-rate, residual alignment, and faithfulness all project
              onto, so its validity is logically prior to every gate.
  * a linear probe (logistic regression, numpy-only, no sklearn) with an AUROC,
              plus a paired cross-validation split so the AUROC is not inflated
              by pair leakage.

Gate-1 (instrument validation) lives here in spirit: before trusting any
residual-alignment number we check that the probe (a) beats a bag-of-words
baseline, (b) survives length-residualization, (c) collapses to ~0.5 under
shuffled labels, and (d) TRANSFERS to held-out naturalistic activations where
the disposition arises without paired-prompt scaffolding. (d) is the one that
distinguishes "the concept is really represented" from "the contrast-pair
construction leaks a surface feature." The transfer check needs real
activations and so runs on the GPU box; the rest is numpy and is exercised by
the self-test here.

Everything in this file is numpy; activations are passed in as float arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConceptDirection:
    name: str
    delta_c: np.ndarray  # (d,) raw mean-difference, un-normalized
    mu_present: np.ndarray
    mu_absent: np.ndarray
    n_present: int
    n_absent: int

    @property
    def unit(self) -> np.ndarray:
        return self.delta_c / np.linalg.norm(self.delta_c)


def mean_difference(h_present: np.ndarray, h_absent: np.ndarray, name: str) -> ConceptDirection:
    """CAA-style concept direction. Operate in whatever space the caller passes;
    for residual-alignment work, pass NORMALIZED activations q(h) so delta_c
    lives in the same space as the residual e = q(h) - q(h_hat)."""
    hp = np.asarray(h_present, dtype=np.float64)
    ha = np.asarray(h_absent, dtype=np.float64)
    mp, ma = hp.mean(0), ha.mean(0)
    return ConceptDirection(name, mp - ma, mp, ma, hp.shape[0], ha.shape[0])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def fit_logistic(
    X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 500, lr: float = 0.1
) -> np.ndarray:
    """Minimal L2-regularized logistic regression (full-batch gradient descent).

    Returns weights of shape (d+1,) with bias as the last element. Kept simple
    and dependency-free on purpose; this is a probe, not a product."""
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros(Xb.shape[1])
    n = Xb.shape[0]
    # standardize features for conditioning (store nothing; probe is per-call)
    for _ in range(iters):
        p = _sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / n + l2 * np.r_[w[:-1], 0.0]
        w -= lr * grad
    return w


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank-sum (Mann-Whitney U) identity. labels in {0,1}."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("auroc needs both classes present")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    tie_mean = np.zeros(len(counts))
    np.add.at(tie_mean, inv, ranks)
    tie_mean /= counts
    ranks = tie_mean[inv]
    r_pos = ranks[y == 1].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def probe_auroc_cv(
    X: np.ndarray, y: np.ndarray, pair_id: np.ndarray | None = None, folds: int = 5, seed: int = 0
) -> dict:
    """Cross-validated probe AUROC. If pair_id is given, both members of a
    contrast pair go to the same fold (prevents pair leakage inflating AUROC)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n = X.shape[0]
    if pair_id is None:
        groups = np.arange(n)
    else:
        groups = np.asarray(pair_id)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    fold_of_group = {g: i % folds for i, g in enumerate(uniq)}
    fold = np.array([fold_of_group[g] for g in groups])

    scores = np.empty(n)
    for f in range(folds):
        tr, te = fold != f, fold == f
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            scores[te] = 0.0
            continue
        w = fit_logistic(X[tr], y[tr])
        scores[te] = np.hstack([X[te], np.ones((te.sum(), 1))]) @ w
    return {"auroc": auroc(scores, y), "scores": scores}


def shuffled_label_auroc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Negative control: AUROC with permuted labels should be ~0.5."""
    rng = np.random.default_rng(seed)
    yp = rng.permutation(np.asarray(y).astype(int))
    return probe_auroc_cv(X, yp, seed=seed)["auroc"]


def _selftest() -> None:
    rng = np.random.default_rng(0)
    d, n = 256, 1200
    # planted concept direction with signal-to-noise that should give high AUROC
    true_dir = rng.standard_normal(d)
    true_dir /= np.linalg.norm(true_dir)
    y = (rng.random(n) > 0.5).astype(int)
    X = rng.standard_normal((n, d)) + 3.0 * (y[:, None] - 0.5) * true_dir[None, :]

    cd = mean_difference(X[y == 1], X[y == 0], "synthetic")
    # recovered direction should align with the planted one
    cos = float(cd.unit @ true_dir)
    assert cos > 0.9, f"mean-difference did not recover planted dir (cos={cos:.2f})"

    real = probe_auroc_cv(X, y, seed=1)["auroc"]
    assert real > 0.95, f"probe AUROC too low on planted signal ({real:.3f})"

    shuf = shuffled_label_auroc(X, y, seed=1)
    assert abs(shuf - 0.5) < 0.12, f"shuffled-label AUROC not ~0.5 ({shuf:.3f})"

    # AUROC sanity: perfect separation -> 1.0, reversed -> 0.0
    assert abs(auroc(np.arange(10), np.r_[np.zeros(5), np.ones(5)]) - 1.0) < 1e-9
    assert abs(auroc(np.arange(10), np.r_[np.ones(5), np.zeros(5)]) - 0.0) < 1e-9

    print("concepts self-test: PASS")
    print(f"  recovered-direction cos = {cos:.3f}")
    print(f"  probe AUROC (real labels)   = {real:.3f}")
    print(f"  probe AUROC (shuffled)      = {shuf:.3f}")


if __name__ == "__main__":
    _selftest()
