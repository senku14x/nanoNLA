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


def transfer_report(
    a_present: np.ndarray, a_absent: np.ndarray,
    b_present: np.ndarray, b_absent: np.ndarray,
    name_a: str = "train", name_b: str = "test",
    a_groups: list[int] | None = None, b_groups: list[int] | None = None,
    b_len: np.ndarray | None = None,
    seed: int = 0, gap: float = 0.1,
) -> dict:
    """Gate -1 TRANSFER: does the direction learned on construction A separate the
    concept on a structurally-different construction B?

    This is the check that distinguishes a real concept direction from construction
    leakage. Within-set decodability (probe AUROC ~1.0) CANNOT make that distinction
    — two different text populations are trivially separable by topic/length/template
    — so a near-perfect in-set AUROC is necessary but not sufficient. Reports:

      within_A / within_B : pair-aware CV decodability in each set (the ceilings).
      dir_transfer        : project B onto UNIT Delta_c(A); AUROC vs B labels. The
                            headline number, and exactly the CAA direction we steer
                            at Gate -1b — so it doubles as a steering pre-check.
      probe_transfer      : logistic fit on A (A's standardization, frozen) applied
                            to B; corroborates dir_transfer with a full classifier.
      shuffled_B          : permuted-label floor on B (~0.5).
      len_auroc_B         : 1-feature length / last-token-position probe on B
                            (shortcut control). High => the classes differ by length,
                            so ANY probe AUROC is suspect, transfer or not.

    Verdict TRANSFERS iff dir_transfer AND probe_transfer beat the B floor by `gap`.
    Leakage signature (the failure we want to catch): within_A, within_B ~ 1.0 while
    dir_transfer ~ shuffled_B — decodable in each construction, shared nowhere.
    """
    Ap, Aa = np.asarray(a_present, float), np.asarray(a_absent, float)
    Bp, Ba = np.asarray(b_present, float), np.asarray(b_absent, float)
    XA = np.vstack([Ap, Aa]); yA = np.array([1] * len(Ap) + [0] * len(Aa))
    XB = np.vstack([Bp, Ba]); yB = np.array([1] * len(Bp) + [0] * len(Ba))

    within_A = concepts.probe_auroc_cv(
        XA, yA, np.asarray(a_groups) if a_groups else None, seed=seed)["auroc"]
    within_B = concepts.probe_auroc_cv(
        XB, yB, np.asarray(b_groups) if b_groups else None, seed=seed)["auroc"]
    shuffled_B = concepts.shuffled_label_auroc(XB, yB, seed=seed)

    # direction transfer: UNIT Delta_c from A, projected onto B (the CAA direction)
    dc = concepts.mean_difference(Ap, Aa, name_a).unit
    dir_transfer = concepts.auroc(XB @ dc, yB)

    # probe transfer: logistic on A with A's standardization, FROZEN, applied to B.
    # Standardizing B with A's stats (not B's) is the honest frozen-probe protocol —
    # the preprocessing is part of the classifier learned on A.
    muA = XA.mean(0); sdA = XA.std(0) + 1e-6
    w = concepts.fit_logistic((XA - muA) / sdA, yA)
    sB = np.hstack([(XB - muA) / sdA, np.ones((len(XB), 1))]) @ w
    probe_transfer = concepts.auroc(sB, yB)

    len_auroc_B = None
    if b_len is not None:
        raw = concepts.auroc(np.asarray(b_len, float), yB)
        len_auroc_B = float(max(raw, 1.0 - raw))  # orientation-agnostic shortcut score

    transfers = (dir_transfer > shuffled_B + gap) and (probe_transfer > shuffled_B + gap)
    return {
        "train_set": name_a, "test_set": name_b,
        "within_A": float(within_A), "within_B": float(within_B),
        "dir_transfer": float(dir_transfer), "probe_transfer": float(probe_transfer),
        "shuffled_B": float(shuffled_B), "len_auroc_B": len_auroc_B,
        "verdict": "TRANSFERS" if transfers else "NO_TRANSFER",
        "note": "TRANSFERS => Delta_c is a real concept direction, not construction "
                "leakage. NO_TRANSFER with high within_A/within_B => the in-set AUROC "
                "WAS leakage. len_auroc_B high => length confound; distrust all AUROC. "
                "dir_transfer is the CAA direction steered at Gate -1b.",
    }


def main(argv=None) -> None:  # pragma: no cover - NEEDS-GPU
    """CLI: extract L24 activations for a concept's pairs and validate. Example:
      python -m lv_explainers.validate_concepts --model Qwen/Qwen3-8B --layer 24 \
          --concept refusal --present harmful.txt --absent harmless.txt
      python -m lv_explainers.validate_concepts --model Qwen/Qwen3-8B --layer 24 \
          --concept corrigibility --ab corrigible-neutral-HHH.jsonl

    TRANSFER (Gate -1 validity): train Delta_c on the primary set, test whether it
    separates a structurally-different held-out set — the check that distinguishes a
    real direction from construction leakage (in-set AUROC ~1.0 cannot):
      python -m lv_explainers.validate_concepts --model Qwen/Qwen3-8B --layer 24 \
          --concept truth_value --present data/truth_value/true.txt \
          --absent data/truth_value/false.txt \
          --transfer-present data/truth_transfer/larger_than_true.txt \
          --transfer-absent  data/truth_transfer/larger_than_false.txt

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
    ap.add_argument("--transfer-ab", help="held-out A/B jsonl for the transfer test")
    ap.add_argument("--transfer-present", help="held-out present-text file (transfer test)")
    ap.add_argument("--transfer-absent", help="held-out absent-text file (transfer test)")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args(argv)

    def build(ab, present, absent, src):
        if ab:
            return data.load_ab_jsonl(ab, args.concept, limit=args.limit)
        if present and absent:
            return data.load_two_population(data.load_lines(present)[: args.limit],
                                            data.load_lines(absent)[: args.limit],
                                            args.concept, src)
        return None

    cs = build(args.ab, args.present, args.absent, f"{args.present}|{args.absent}")
    if cs is None:
        ap.error("provide --ab OR (--present and --absent)")
    cs_t = build(args.transfer_ab, args.transfer_present, args.transfer_absent,
                 f"{args.transfer_present}|{args.transfer_absent}")

    ex = nla_io.ActivationExtractor(args.model, args.layer)
    try:
        def acts(texts):
            """Return (vectors (n,d), seq_lengths (n,)) at the last token."""
            out, lens = [], []
            for t in texts:
                pos, v = ex.activations(t, positions=[-1])  # last token; see docstring
                out.append(v[-1]); lens.append(pos[-1] + 1)  # resolved -1 -> seq-1
            return np.vstack(out), np.asarray(lens, dtype=float)
        P, _ = acts(cs.present_texts)
        A, _ = acts(cs.absent_texts)
        bp = ba = b_len = None
        if cs_t is not None:
            bp, bp_len = acts(cs_t.present_texts)
            ba, ba_len = acts(cs_t.absent_texts)
            b_len = np.r_[bp_len, ba_len]
    finally:
        ex.close()

    rep = validate_concept(args.concept, P, A, cs.pair_ids)
    if cs_t is None:
        print(json.dumps(rep, indent=2))
        return
    trep = transfer_report(P, A, bp, ba, name_a=cs.source, name_b=cs_t.source,
                           a_groups=cs.cv_groups(), b_groups=cs_t.cv_groups(), b_len=b_len)
    print(json.dumps({"concept": args.concept, "decodability": rep, "transfer": trep}, indent=2))


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

    # --- TRANSFER: the harness must DISTINGUISH a shared direction from per-set
    # construction leakage. This is the failure-inducing control the contract wants:
    # both worlds are ~1.0 decodable in-set; only the shared one should transfer. ---
    rng = np.random.default_rng(1)
    dd = 256

    def unit():
        u = rng.standard_normal(dd); return u / np.linalg.norm(u)

    def pop(direction, n=120, s=3.0):
        return rng.standard_normal((n, dd)) + s * direction

    shared, cA, cB = unit(), unit(), unit()  # shared concept dir + per-set nuisances
    # World 1 (real): both constructions carry the SAME +/- shared signal (plus own nuisance)
    tr_share = transfer_report(pop(shared + cA), pop(-shared - cA),
                               pop(shared + cB), pop(-shared - cB), "A", "B")
    assert tr_share["within_A"] > 0.9 and tr_share["within_B"] > 0.9, tr_share
    assert tr_share["dir_transfer"] > 0.85, tr_share
    assert tr_share["verdict"] == "TRANSFERS", tr_share
    # World 2 (leakage): each set separated ONLY by its own orthogonal nuisance.
    # Decodable in-set (~1.0) but shares no direction -> transfer must collapse.
    tr_leak = transfer_report(pop(cA), pop(-cA), pop(cB), pop(-cB), "A", "B")
    assert tr_leak["within_A"] > 0.9 and tr_leak["within_B"] > 0.9, tr_leak
    assert tr_leak["dir_transfer"] < 0.65, tr_leak
    assert tr_leak["verdict"] == "NO_TRANSFER", tr_leak
    # length shortcut control: a planted length confound is detected
    b_len = np.r_[np.full(120, 90.0), np.full(120, 20.0)]  # present systematically longer
    tr_len = transfer_report(pop(shared), pop(-shared), pop(shared), pop(-shared),
                             "A", "B", b_len=b_len)
    assert tr_len["len_auroc_B"] > 0.9, tr_len

    print("validate_concepts self-test: PASS")
    print(f"  planted -> {r1['verdict']} (probe {r1['probe_auroc']:.2f} / shuf {r1['shuffled_auroc']:.2f})")
    print(f"  noise   -> {r2['verdict']} (probe {r2['probe_auroc']:.2f} / shuf {r2['shuffled_auroc']:.2f})")
    print(f"  transfer shared -> {tr_share['verdict']} "
          f"(dir {tr_share['dir_transfer']:.2f}, within_B {tr_share['within_B']:.2f})")
    print(f"  transfer leak   -> {tr_leak['verdict']} "
          f"(dir {tr_leak['dir_transfer']:.2f}, within_B {tr_leak['within_B']:.2f}) "
          f"[in-set decodable, does NOT transfer]")
    print(f"  length shortcut -> len_auroc_B {tr_len['len_auroc_B']:.2f} (confound caught)")


if __name__ == "__main__":
    import sys
    # GPU CLI when --model is passed; otherwise run the numpy self-test.
    if any(a == "--model" or a.startswith("--model=") for a in sys.argv):
        main()
    else:
        _selftest()
