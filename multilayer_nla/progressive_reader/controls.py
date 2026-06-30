"""Evaluation-only controls (spec §8.3 no-text, §8.4 shuffled-text) — stdlib only.

The shuffled-text control deranges teacher prefixes ACROSS DIFFERENT DOCUMENTS within an
eval split, preserving budget / effective length / target activations / row metadata, with
NO row mapping to itself or to a row from the same document. This tests whether a high real
score is genuine text dependence vs distributional luck — the same negative control the rest
of the project uses, lifted to the per-(doc,row) level here.
"""

from __future__ import annotations

import random


def doc_derangement(doc_ids, seed: int) -> list[int]:
    """Permutation p with doc_ids[p[i]] != doc_ids[i] for every i where possible (and hence
    p[i] != i). Deterministic in `seed`. Stdlib `random` (no numpy) so it's test-runnable.

    Greedy: shuffle, then for any position still landing on its own document, swap with a
    later position that fixes BOTH without creating a new collision. With many documents
    (the real case) this fully deranges; if a residual self/same-doc pair survives (tiny or
    near-degenerate inputs), `assert_deranged` will catch it and the caller should re-seed.
    """
    n = len(doc_ids)
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    for i in range(n):
        if doc_ids[perm[i]] == doc_ids[i]:
            for j in range(n):
                if (j != i
                        and doc_ids[perm[j]] != doc_ids[i]        # j's source ok for i
                        and doc_ids[perm[i]] != doc_ids[j]):      # i's source ok for j
                    perm[i], perm[j] = perm[j], perm[i]
                    break
    return perm


def assert_deranged(doc_ids, perm) -> None:
    """Spec §13.8: every shuffled row maps to a DIFFERENT document, none to itself."""
    assert len(perm) == len(doc_ids), "perm length mismatch"
    assert sorted(perm) == list(range(len(doc_ids))), "perm is not a permutation"
    for i, pi in enumerate(perm):
        assert pi != i, f"row {i} maps to itself"
        assert doc_ids[pi] != doc_ids[i], (
            f"row {i} (doc {doc_ids[i]}) maps to same-document row {pi} (doc {doc_ids[pi]})")
