"""Numpy-only proof of the headroom-probe math (plan §5.1.6, §6.3, §8).

Runs WITHOUT torch/pyarrow — exercises the pure numerical core of
multilayer_nla.headroom so the FVE_delta bookkeeping (ceiling==1, floor==0,
sqrt(d)-norm), the ridge baseline, and the Gate-0 decision are validated before
any GPU time is spent. The plan is emphatic that these identities must hold or
"you have caught it before it poisons everything downstream."

Run:  python -m multilayer_nla.tests.test_fve_math
(also discoverable by pytest as test_* functions).
"""

import math

import numpy as np

from multilayer_nla.headroom import (
    GATE0_HEADROOM_MIN,
    compute_headroom,
    doc_split_mask,
    fve,
    make_synthetic_patches,
    normalize_sqrt_d,
    ridge_fit,
    ridge_predict,
)


def test_normalize_sqrt_d():
    rng = np.random.default_rng(0)
    d = 128
    a = rng.standard_normal((500, d)) * rng.uniform(0.1, 50, size=(500, 1))  # varied norms
    u = normalize_sqrt_d(a)
    norms = np.linalg.norm(u, axis=1)
    assert np.allclose(norms, math.sqrt(d), rtol=1e-6), norms[:3]
    # direction preserved
    cos = np.sum(a * u, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(u, axis=1))
    assert np.allclose(cos, 1.0, atol=1e-6)


def test_fve_floor_and_ceiling():
    rng = np.random.default_rng(1)
    target = rng.standard_normal((1000, 32))
    floor = target.mean(axis=0, keepdims=True)
    # ceiling: perfect prediction -> FVE == 1
    assert abs(fve(target, target, floor) - 1.0) < 1e-12
    # floor: predict the mean -> FVE == 0 (denominator == numerator)
    pred_const = np.broadcast_to(floor, target.shape)
    assert abs(fve(pred_const, target, floor)) < 1e-12
    # a predictor worse than the mean -> FVE < 0
    assert fve(target + 5.0, target, floor) < 0.0


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(2)
    N, d, k = 4000, 40, 12
    X = rng.standard_normal((N, d))
    W_true = rng.standard_normal((d, k))
    b_true = rng.standard_normal((1, k))
    Y = X @ W_true + b_true  # noiseless linear
    W, xm, ym = ridge_fit(X, Y, lam=1e-6)
    pred = ridge_predict(X, W, xm, ym)
    floor = Y.mean(axis=0, keepdims=True)
    assert fve(pred, Y, floor) > 0.999
    # add noise -> FVE drops but stays high
    Yn = Y + rng.standard_normal((N, k)) * 0.1
    Wn, xmn, ymn = ridge_fit(X, Yn, lam=1.0)
    assert 0.8 < fve(ridge_predict(X, Wn, xmn, ymn), Yn, Yn.mean(0, keepdims=True)) < 1.0


def test_doc_split_is_document_level_and_deterministic():
    # 200 docs x 10 positions; all positions of a doc must share a bucket.
    doc_ids = [f"doc{i // 10}" for i in range(2000)]
    m1 = doc_split_mask(doc_ids, frac=0.2, seed=0)
    m2 = doc_split_mask(doc_ids, frac=0.2, seed=0)
    assert np.array_equal(m1, m2)  # deterministic
    # every doc's 10 rows are all-in or all-out
    for d0 in range(0, 2000, 10):
        block = m1[d0:d0 + 10]
        assert block.all() or (~block).all(), f"doc {d0//10} split across buckets"
    # frac roughly honored
    frac_docs = m1[::10].mean()
    assert 0.12 < frac_docs < 0.28
    # different seed -> different (in general) partition
    assert not np.array_equal(m1, doc_split_mask(doc_ids, frac=0.2, seed=99))


def test_compute_headroom_identities_and_verdicts():
    rng = np.random.default_rng(3)
    d, N = 64, 4000
    doc_ids = [f"doc{i // 10}" for i in range(N)]

    # STOP: neighbors are orthogonal (linear) maps of center -> H_unique ~ 0
    a_prev, a_centre, a_next = make_synthetic_patches(rng, d, N, independent_next=False)
    res_stop = compute_headroom(a_prev, a_centre, a_next, doc_ids, seed=1, use_mlp=False)
    # built-in identities
    assert abs(res_stop["fve_delta"]["constant_floor"]) < 1e-5
    assert abs(res_stop["fve_delta"]["true_neighbor_ceiling"] - 1.0) < 1e-5
    assert res_stop["fve_delta"]["ridge"] > 0.95
    assert res_stop["H_unique"] < GATE0_HEADROOM_MIN
    assert res_stop["gate0_verdict"] == "STOP"

    # GO: upper update independent of center -> H_unique well above threshold
    a_prev, a_centre, a_next = make_synthetic_patches(rng, d, N, independent_next=True)
    res_go = compute_headroom(a_prev, a_centre, a_next, doc_ids, seed=1, use_mlp=False)
    assert res_go["H_unique"] >= GATE0_HEADROOM_MIN
    assert res_go["gate0_verdict"] == "GO"
    # ridge-only GO is flagged provisional (an MLP could only lower H_unique)
    assert res_go["gate0_provisional"] is True


def test_h_unique_definition():
    # H_unique = 1 - best center-only FVE_delta; gate is >= 0.05
    rng = np.random.default_rng(4)
    d, N = 48, 3000
    doc_ids = [f"doc{i // 10}" for i in range(N)]
    a_prev, a_centre, a_next = make_synthetic_patches(rng, d, N, independent_next=True)
    res = compute_headroom(a_prev, a_centre, a_next, doc_ids, seed=2, use_mlp=False)
    assert abs(res["H_unique"] - (1.0 - res["best_center_only_fve_delta"])) < 1e-12
    assert res["best_center_only"] == "ridge"  # only predictor available without torch


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
