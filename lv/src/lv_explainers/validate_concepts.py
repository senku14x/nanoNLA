"""Gate -1 / Gate-1-decodability: TEST THE CONCEPT VECTORS on the target model.

This is the FIRST experimental step (Gate 0a in the Rev-3 doc): for each concept,
does a linearly-decodable direction Delta_c exist in the model's L24 activations,
and is it real (not a lexical shortcut or a shuffled-label artifact)?

It does NOT need the NLA — just the base model + contrast pairs + a probe. The
AV/AR enter later (Gate 0 read-rate / counterfactual mention).

Split:
  * `validate_concept(...)`  pure numpy given activations — UNIT-TESTED here.
  * `main()`                 CLI that extracts Qwen3-8B L24 activations via
                             nla_io.ActivationExtractor, then calls it. NEEDS-GPU.

Verdict per concept:
  DECODABLE      probe AUROC >> lexical floor AND >> shuffled(~0.5) -> real direction
  LEXICAL_LEAK   probe high but lexical baseline also high -> surface shortcut
  NOT_DECODABLE  probe ~ shuffled -> no linear direction at this layer
(Transfer-to-naturalistic + the causal steering screen (Gate -1b) are separate,
stronger checks; this is the decodability gate.)
"""

from __future__ import annotations

import numpy as np

from . import concepts, text_baselines


def validate_concept(
    name: str,
    present_acts: np.ndarray,   # (n_p, d) raw L24 activations, present-c
    absent_acts: np.ndarray,    # (n_a, d) raw L24 activations, absent-c
    present_texts: list[str],
    absent_texts: list[str],
    pair_ids: list[int] | None = None,
    lexical_gap: float = 0.15,
    shuffled_gap: float = 0.15,
    seed: int = 0,
) -> dict:
    """Compute Delta_c + probe AUROC + lexical-floor + shuffled control + verdict."""
    P = np.asarray(present_acts, dtype=np.float64)
    A = np.asarray(absent_acts, dtype=np.float64)
    assert P.shape[1] == A.shape[1], "present/absent activation dims differ"

    cd = concepts.mean_difference(P, A, name)
    X = np.vstack([P, A])
    y = np.array([1] * len(P) + [0] * len(A))
    groups = None
    if pair_ids is not None:
        nxt = (max(pair_ids) + 1) if pair_ids else 0
        groups = list(pair_ids) + [pair_ids[i] if i < len(pair_ids) else nxt + i
                                   for i in range(len(A))]

    probe = concepts.probe_auroc_cv(X, y, np.asarray(groups) if groups else None, seed=seed)["auroc"]
    shuffled = concepts.shuffled_label_auroc(X, y, seed=seed)
    lexical = text_baselines.lexical_auroc(present_texts + absent_texts, y.tolist(),
                                           groups, seed=seed)["auroc"]
    # sanity: probe direction should roughly align with the mean-difference
    cos_probe_delta = None  # filled by caller if it has the probe weights; optional

    if probe < shuffled + shuffled_gap:
        verdict = "NOT_DECODABLE"
    elif lexical > probe - lexical_gap:
        verdict = "LEXICAL_LEAK"
    else:
        verdict = "DECODABLE"

    return {
        "concept": name,
        "n_present": int(len(P)),
        "n_absent": int(len(A)),
        "probe_auroc": float(probe),
        "shuffled_auroc": float(shuffled),
        "lexical_auroc": float(lexical),
        "delta_c_norm": float(np.linalg.norm(cd.delta_c)),
        "verdict": verdict,
        "note": "decodability only; run Gate -1b (causal) + transfer before trusting.",
    }


def main(argv=None) -> None:  # pragma: no cover - NEEDS-GPU
    """CLI: extract L24 activations for a concept's pairs and validate. Example:
      python -m lv_explainers.validate_concepts --model Qwen/Qwen3-8B --layer 24 \
          --concept refusal --present harmful.txt --absent harmless.txt
      python -m lv_explainers.validate_concepts --model Qwen/Qwen3-8B --layer 24 \
          --concept corrigibility --ab corrigible-neutral-HHH.jsonl
    Activations are read at the LAST token of each rendered text (override the
    extraction position to match the NLA regime as needed)."""
    import argparse, json
    from . import data, nla_io

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--ab", help="A/B jsonl (Anthropic evals / CAA)")
    ap.add_argument("--present", help="present-text file (two-population, e.g. harmful)")
    ap.add_argument("--absent", help="absent-text file (two-population, e.g. harmless)")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args(argv)

    if args.ab:
        cs = data.load_ab_jsonl(args.ab, args.concept, limit=args.limit)
    elif args.present and args.absent:
        cs = data.load_two_population(data.load_lines(args.present)[: args.limit],
                                      data.load_lines(args.absent)[: args.limit],
                                      args.concept, f"{args.present}|{args.absent}")
    else:
        ap.error("provide --ab OR (--present and --absent)")

    ex = nla_io.ActivationExtractor(args.model, args.layer)
    try:
        def acts(texts):
            out = []
            for t in texts:
                _, v = ex.activations(t, positions=[-1])  # last token; see docstring
                out.append(v[-1])
            return np.vstack(out)
        P = acts(cs.present_texts)
        A = acts(cs.absent_texts)
    finally:
        ex.close()

    rep = validate_concept(args.concept, P, A, cs.present_texts, cs.absent_texts, cs.pair_ids)
    print(json.dumps(rep, indent=2))


def _selftest() -> None:
    rng = np.random.default_rng(0)
    d = 512

    # DECODABLE: a planted direction separates classes; texts share vocabulary
    # (so lexical floor ~ chance) -> probe high, lexical low -> DECODABLE.
    vdir = rng.standard_normal(d); vdir /= np.linalg.norm(vdir)
    P = rng.standard_normal((80, d)) + 3.0 * vdir
    A = rng.standard_normal((80, d)) - 3.0 * vdir
    pool = "user model goal task value system".split()
    txt = lambda: " ".join(rng.choice(pool, 5))
    r1 = validate_concept("planted", P, A, [txt() for _ in range(80)], [txt() for _ in range(80)])
    assert r1["verdict"] == "DECODABLE", r1

    # NOT_DECODABLE: no separation in activations
    P2 = rng.standard_normal((80, d)); A2 = rng.standard_normal((80, d))
    r2 = validate_concept("noise", P2, A2, [txt() for _ in range(80)], [txt() for _ in range(80)])
    assert r2["verdict"] == "NOT_DECODABLE", r2

    # LEXICAL_LEAK: activations separable AND texts lexically separable
    pres_t = [f"alpha keyword case {i}" for i in range(80)]
    abs_t = [f"beta keyword case {i}" for i in range(80)]
    r3 = validate_concept("leaky", P, A, pres_t, abs_t)
    assert r3["verdict"] == "LEXICAL_LEAK", r3

    print("validate_concepts self-test: PASS")
    print(f"  planted -> {r1['verdict']} (probe {r1['probe_auroc']:.2f} / lex {r1['lexical_auroc']:.2f})")
    print(f"  noise   -> {r2['verdict']} (probe {r2['probe_auroc']:.2f} / shuf {r2['shuffled_auroc']:.2f})")
    print(f"  leaky   -> {r3['verdict']} (probe {r3['probe_auroc']:.2f} / lex {r3['lexical_auroc']:.2f})")


if __name__ == "__main__":
    import sys
    # GPU CLI when --model is passed; otherwise run the numpy self-test.
    if any(a == "--model" or a.startswith("--model=") for a in sys.argv):
        main()
    else:
        _selftest()
