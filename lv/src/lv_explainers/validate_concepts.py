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
  DECODABLE      probe AUROC >> shuffled(~0.5) -> a linear direction exists at L24
  NOT_DECODABLE  probe ~ shuffled -> no linear direction at this layer
LEAKAGE/validity is NOT judged by a lexical classifier on the constructed prompts
(confounded — the rendered text encodes the label; see validate_concept / D8). It is
judged by TRANSFER to naturalistic activations (Gate -1 transfer) and the causal
steering screen (Gate -1b). Text baselines (lexical + semantic) live on the AV
EXPLANATION at Gate 0, where the text is generated, not constructed.
"""

from __future__ import annotations

import numpy as np

from . import concepts


def validate_concept(
    name: str,
    present_acts: np.ndarray,   # (n_p, d) raw L24 activations, present-c
    absent_acts: np.ndarray,    # (n_a, d) raw L24 activations, absent-c
    pair_ids: list[int] | None = None,
    shuffled_gap: float = 0.1,
    seed: int = 0,
) -> dict:
    """Delta_c + probe AUROC vs a shuffled-label control -> DECODABLE / NOT_DECODABLE.

    D8: we deliberately do NOT run a lexical/TF-IDF classifier on the contrast
    prompts here. For these constructions it is confounded — the rendered text
    encodes the label (the appended answer letter for A/B pairs; entirely different
    prompts for two-population sets like refusal/truth), so a text classifier scores
    ~1.0 and says nothing about whether the *activation* direction is the concept.
    Leakage/validity is tested by TRANSFER to held-out naturalistic activations
    (separate). Text baselines (lexical + semantic) belong on the AV EXPLANATION at
    Gate 0, where the text is generated, not constructed.
    """
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
    verdict = "DECODABLE" if probe > shuffled + shuffled_gap else "NOT_DECODABLE"

    return {
        "concept": name,
        "n_present": int(len(P)),
        "n_absent": int(len(A)),
        "probe_auroc": float(probe),
        "shuffled_auroc": float(shuffled),
        "delta_c_norm": float(np.linalg.norm(cd.delta_c)),
        "verdict": verdict,
        "note": "decodability vs shuffled only; LEAKAGE -> transfer to naturalistic "
                "activations (not a lexical check on constructed prompts).",
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

    rep = validate_concept(args.concept, P, A, cs.pair_ids)
    print(json.dumps(rep, indent=2))


def _selftest() -> None:
    rng = np.random.default_rng(0)
    d = 512
    # DECODABLE: a planted direction separates the classes
    vdir = rng.standard_normal(d); vdir /= np.linalg.norm(vdir)
    P = rng.standard_normal((80, d)) + 3.0 * vdir
    A = rng.standard_normal((80, d)) - 3.0 * vdir
    r1 = validate_concept("planted", P, A)
    assert r1["verdict"] == "DECODABLE", r1
    # NOT_DECODABLE: no separation in activations
    r2 = validate_concept("noise", rng.standard_normal((80, d)), rng.standard_normal((80, d)))
    assert r2["verdict"] == "NOT_DECODABLE", r2

    print("validate_concepts self-test: PASS")
    print(f"  planted -> {r1['verdict']} (probe {r1['probe_auroc']:.2f} / shuf {r1['shuffled_auroc']:.2f})")
    print(f"  noise   -> {r2['verdict']} (probe {r2['probe_auroc']:.2f} / shuf {r2['shuffled_auroc']:.2f})")


if __name__ == "__main__":
    import sys
    # GPU CLI when --model is passed; otherwise run the numpy self-test.
    if any(a == "--model" or a.startswith("--model=") for a in sys.argv):
        main()
    else:
        _selftest()
