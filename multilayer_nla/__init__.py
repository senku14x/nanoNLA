"""Multi-layer NLA — FineWeb architecture-validation phase.

Self-contained experiment package for the "FineWeb Multi-Layer Natural Language
Autoencoders" research plan (Revision 2, design frozen). Everything here REUSES
the stable `nla/` core (keyed-RNG position sampling, HFExtractor, storage) and
never mutates it.

Scope of this phase (NLA-free, pre-training): extract a contiguous three-layer
activation patch [a^(l-1), a^(l), a^(l+1)] per token position, then decide the
pre-registered Gate 0 (unique cross-layer headroom H_unique) before spending any
GPU time on AV/AR/RL. See README.md for the frozen execution order (plan §12.5).

Key invariants carried over from nla/ (CLAUDE.md):
  - data-gen NEVER normalizes: parquets store RAW vectors (norm="none").
    The plan's §4 (Rev 2) confirms this — the AV injects raw a; the
    sqrt(d)-normalized u is used ONLY as AR target and in FVE.
  - document-level split (all positions of a doc in one bucket).
  - per-doc keyed RNG: same (seed, doc_id) -> same sampled positions, so a
    multi-layer run lands on the SAME positions a single-layer run would
    (Condition A comparability + center-tap parity vs legacy extraction).
"""
