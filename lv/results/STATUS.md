# NLA whitened-reward study — status & findings (live)

Empirical record of what has actually run on hardware (Qwen3-8B L24, syvb NLA).
Claims are graded per the research contract: **supported** (survives baselines/
controls/replication) · **measured** (a confirmed instrument/substrate fact) ·
**observation** (narrow, not yet validated) · **retracted** · **untested**.
Read this before re-deriving anything.

## One-paragraph state
The substrate and the measurement instruments are validated; the hypothesis is
not yet touched. We have **one supported result** — truth_value is a real,
transferable concept direction at L24 — and **one well-characterized baseline** —
syvb's length-penalty sweep. The AR critic loads and reconstructs correctly
(cross-checked against the repo's own `NLACriticModel`). **Zero read/unread
numbers have been measured** (H1 is untouched). The next real problem is
on-manifold concept-bearing activations — our concept data lives off the NLA's
FineWeb manifold.

## Claims ledger

### Supported
- **truth_value is a real, transferable direction (Qwen3-8B L24).** Δ_c trained
  on GoT cities separates structurally-different held-out constructions:
  larger_than `dir_transfer 0.996`, neg_cities `0.999` (survives **negation**, the
  classic failure case), vs ~0.50 shuffled floor; length-shortcut control
  `len_auroc 0.62 / 0.51` (cannot drive it). Caveat: may track *factual
  plausibility* rather than a truth-disposition — fine for a read control.
  → `results/gate_minus1/truth_value_transfer*.json`, `RUN.md`.

### Measured (substrate / instrument facts)
- **Substrate is post-trained Qwen3-8B**, not `-Base`: sidecar
  `extraction.base_model: Qwen/Qwen3-8B` + behavioral (refuses harmful, resists
  flat-earth bait once Qwen3 thinking mode is handled). Arch `qwen3, 36 layers,
  d_model 4096`, read layer `L24 = (2/3)·36`.
- **Sidecar contract** (`nla_meta.yaml`): critic template
  `'Summary of the following text: <text>{explanation}</text> <summary>'`; marker
  `㈎`/`149705`; `critic_suffix_ids [1318,29,366,1708,29]`; `norm: none`;
  **no `mse_scale` key** → released default `sqrt(d_model)=64`.
- **AR critic loads + reconstructs.** Matches the repo's `NLACriticModel`
  (final LN→Identity; `value_head = Linear(4096,4096)` frozen at identity for
  Qwen3-8B; K+1 = 25 layers; suffix-anchored last token). syvb's own held-out
  FVE = `0.532` (base) / `0.585` (p0.0), baseline `mse_nrm 0.670`.
- **λ=0 base = `rl-lora/p0.0`** (the no-penalty RL run). The rl-lora repo is a
  length-penalty sweep `p0.0…p0.03`.
- **LoRA geometry: r=16, α=32, rsLoRA, attn-only** (q/k/v/o). Two RL runs exist:
  *overnight* (r16/α32, **frozen AR**) and *long* (r128/α16, **co-trained AR**) —
  corrects the doc's earlier "r=128/α=16" as the single recipe.
  → `results/model_check/qwen3_8b_substrate.md`.

### Observations (in-set only, NOT yet validated)
- **refusal, truth_value, corrigibility are decodable in-set** (probe AUROC
  ~0.99–1.0 vs shuffled ~0.5). Necessary, not sufficient — near-perfect in-set
  AUROC is the signature of two trivially-separable text populations. Only
  truth_value has passed transfer. **corrigibility cannot transfer** (MCQ-only) →
  stays the weakest, can't headline. refusal transfer not yet run.

### Baseline characterized — length-penalty sweep
- Small penalty (p0.001) is **Pareto-neutral-to-better** (−28% tokens, +0.8pp
  FVE). Reconstruction robust to −41% tokens (−1.4pp); cliffs below ~25 tokens
  (p0.03: 14 tok → FVE 0.22). **FVE ceiling ~0.59** → ~41% variance never
  reconstructed (headroom; decodability untested). RL adds +5.3pp over base.
  **Confabulation rampant and FVE-orthogonal** → faithfulness must be behavioral.
  **This does not test H1** (verbosity-in-explanation ≠ omission-from-explanation).
  → `results/length_penalty/ANALYSIS.md`, `sweep_metrics.csv`.

### Retracted
- **"refusal is not read"** (the off-manifold AR smoke test). Invalid by design:
  fed last-token AdvBench activations (off the NLA's FineWeb pos≥50 manifold) and
  a hand-written non-AV explanation. Withdrawn; superseded by the FVE validator.

### Untested (the actual phenomenon)
- **No read/unread number measured.** Every read/unread label (incl. "refusal is
  read") is still an assumption — the instrument that measures it (Gate 0
  counterfactual mention on on-manifold activations) has not been run.

## What we ran (work log)
1. Repo/branch hygiene: branch `experimentation`; box re-synced off a stale
   pre-D8 checkout; fixed `check_model` Qwen3 thinking-mode false-negative.
2. Substrate check (D2 prereq) → all pass; sidecar + checkpoint inventory dumped.
3. Gate −1 decodability on refusal/truth/corrigibility.
4. Built + self-tested the **transfer harness** (`validate_concepts.transfer_report`,
   proven on a planted leakage world) and the GoT transfer datasets; ran truth.
5. Built + repo-validated the **AR** path (`nla_io.ARScorer` ≡ `NLACriticModel`),
   retracted the bad smoke test, wrote the held-out FVE validator
   (`scripts/ar_fve_heldout.py`).
6. Persisted + analyzed syvb's length-penalty sweep.

## The real blocker & next experiments (ranked by info/effort)
1. **On-manifold concept instances** — the crux. Concept data (AdvBench/GoT/MCQ)
   is off the NLA's FineWeb manifold, so no read/unread number is yet trustworthy.
   Cheapest path: use the validated probe/Δ_c to **select concept-bearing
   positions from the released `-data-full` FineWeb activations**, then run the
   Gate-0 counterfactual-mention test there with the oracle explanation as
   baseline — on-manifold, **no AV/SGLang needed**.
2. Targets (sycophancy/deception): disposition-not-output fetchers, then transfer,
   then the Gate −1b causal screen — only on directions that transfer.
3. Optional: re-run the fixed AR FVE validator to cross-check our loader (~0.53
   expected); syvb's own number already validates the AR.

## Open questions still live
- Does any target go unread on this NLA on-manifold? (H1, unmeasured.)
- Can a GoT-trained truth direction select truth-bearing FineWeb positions
  (yet another transfer)?
- corrigibility in/out of the headline (MCQ-only → likely out).
