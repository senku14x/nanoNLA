# Literature audit (verification status)

Purpose: the design's §9 leans on several references, some flagged "LLM-survey-
sourced, unverified — where fabricated citations hide." Per the research contract
(measurement tools and citations are hypotheses), each load-bearing ID was
checked for **existence + headline claim**.

**Verification channel & limit.** arXiv (abs/html/API) and the NLA publication
site all return HTTP 403 to the tooling available in this environment, so
verification is via web search of titles/abstracts, **not full-text PDF reads**.
Existence and headline claims are confirmed below; exact page-level quotes (e.g.
the "reconstruction wall" phrasing in 2604.20682) remain **unverified** and must
be read on the GPU box (which has open egress) before any external writeup.

| arXiv ID | Title (verified) | Status | Headline claim (verified via abstract) |
|---|---|---|---|
| 2604.20682 | Variance Is Not Importance: Structural Analysis of Transformer Compressibility Across Model Scales (Salfati) | REAL | High-variance directions ~96% uncorrelated (CCA) with predictive directions; GPT-2 124M + Mistral 7B, WikiText-2. Supports "variance ≠ importance." |
| 2508.16929 | Dimensional Collapse in Transformer Attention Outputs (Wang, Ge, Shu, He, Qiu) | REAL | Attention outputs ~60% effective rank; **residual streams & MLP ~90% (near full-rank)**. Undercuts "residual tail is noise." |
| 2605.14258 | Dynamics of the Transformer Residual Stream: Coupling Spectral Geometry to Network Topology (Fernando & Guitchounts) | REAL | Full **Jacobian** eigendecomposition; cumulative low-rank bottleneck through depth. About the propagation operator, **not** Σ_q. |
| 2605.01609 | Concepts Whisper While Syntax Shouts: Spectral Anti-Concentration and the Dual Geometry of Transformer Representations (Acharya, Rimal, Dhakal) | REAL | Concept vs syntax geometry; weak/supporting analogue (was flagged "unverified"). |
| 2601.16644 | Sycophancy Hides Linearly in the Attention Heads | REAL | Sycophancy most linearly separable in middle-layer attention heads. |
| 2509.21305 | Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs (OpenReview d24zTCznJu) | REAL | Sycophancy = distinct, independently steerable linear directions. Motivates subtype splitting. |

**Net:** no fabricated citations; the audit's *re-scoping* matches the abstracts.
The three load-bearing claims hold as the design's Rev-3 §9 states them:
1. 2604.20682 supports "variance ≠ importance" **and** (per its abstract's framing
   of structural-redundancy methods) cuts against a non-uniform Σ_q reweighting —
   so it is not unambiguous support for the fix.
2. 2508.16929 says the **residual stream is high-rank** → a Σ_q-spectrum "noise
   floor" kill-check is weakly motivated; a Σ_q-based whitening is *less* likely to
   be amplifying pure noise than Rev-2 assumed (recheck empirically in Gate 0b).
3. 2605.14258 is a **Jacobian** result; to invoke "the functional direction is
   unresolvable" you must project onto the Jacobian's active subspace, not Σ_q.

**Additional reading surfaced (not yet integrated):**
- 2605.08740 *Causal Dimensionality of Transformer Representations: Measurement,
  Scaling, and Layer Structure* — directly on the variance-vs-functional axis;
  likely the most relevant new pointer for distinguishing covariance dimensions
  from causally-effective ones. Read on the GPU box.

**To do before any external writeup:** read full PDFs of 2604.20682, 2508.16929,
2605.14258, 2605.08740 on the open-egress box; confirm the exact quotes; verify
the NLA paper's author list and the eval-awareness/hidden-motivation numbers
against the primary transformer-circuits page (currently 403 here).
