"""Exact teacher-token-prefix construction (spec §1.1) — stdlib only.

The non-negotiable invariant: prefixes are token-ID SLICES of the teacher's tokenized
explanation, NEVER decode-then-retokenize. There is no pre-existing Qwen ID sequence for
the API teacher text, so the canonical source of truth is: tokenize the explanation text
ONCE, store the IDs + their SHA-256, then slice. This module owns the slice + the hash;
the one-time tokenization happens in data.py / audit.py with the real tokenizer.
"""

from __future__ import annotations

import hashlib


def teacher_ids_sha256(full_ids) -> str:
    """SHA-256 of the full teacher token-ID sequence (comma-joined). Persisted per row so
    prefix construction can be validated against the exact sequence (spec §1.1)."""
    return hashlib.sha256(",".join(str(int(t)) for t in full_ids).encode()).hexdigest()


def exact_prefix(full_ids, budget: int) -> list[int]:
    """p = e[:min(budget, len(e))] — a pure slice of the stored token IDs. No decode."""
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    return [int(t) for t in full_ids[:budget]]


def effective_prefix_length(full_ids, budget: int) -> int:
    return min(budget, len(full_ids))


def had_full_budget(full_ids, budget: int) -> bool:
    """True iff the explanation had at least `budget` tokens (so the prefix isn't a short
    explanation masquerading as a smaller budget — the strict-prefix concern, spec §3)."""
    return len(full_ids) >= budget


def validate_prefix(full_ids, budget: int, prefix_ids, full_sha256: str | None = None) -> None:
    """Assert prefix_ids == exact_prefix(full_ids, budget) (and, if given, that full_ids
    still hashes to full_sha256). Raises AssertionError on any drift (spec §1.1, §13.1)."""
    want = exact_prefix(full_ids, budget)
    got = [int(t) for t in prefix_ids]
    assert got == want, (
        f"prefix drift at budget {budget}: got {got[:8]}...(len {len(got)}) "
        f"!= full[:{budget}] {want[:8]}...(len {len(want)}) — decode/re-tokenize path?")
    if full_sha256 is not None:
        assert teacher_ids_sha256(full_ids) == full_sha256, (
            "teacher full-id SHA-256 mismatch — the stored token sequence changed")
