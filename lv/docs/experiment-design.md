# Experiment design (Rev 4, operative)

> Operative spec. Supersedes Rev 3 (`experiment-design-rev3.md`, preserved for
> provenance and for the full chain of reasoning behind each correction). Rev 4
> folds in a collaborator review and the now-resolved compute/substrate facts.
> Read with `nla-method-notes.md` (method), `literature.md` (citation audit),
> and `compute.md` (where things run).
>
> **Live decisions override recommendations below — see `decisions.md`.** Current
> state: substrate is **Qwen3-8B (syvb) for all gates** (D2); panel is **refusal,
> truth_value, sycophancy, deception/withholding** with corrigibility optional (D3).

## 0. Changes from Rev 3
1. **Instrument validation is now a gate that runs FIRST (Gate −1).** Rev 3 put
   the probe transfer/leakage check inside Gate 1, *after* the residual figure.
   But Gate 0 measures residual alignment *along* `Δ_c` and uses the read/unread
   labels — if `Δ_c` is a contrast-pair-construction artifact, every downstream
   number inherits the error coherently. Validate the probe/`Δ_c` before
   measuring with it.
2. **The decisive Gate 0 is a counterfactual-mention test, not the linearized
   `⟨e, Δ_c⟩`.** `⟨e, Δ_c⟩` is a first-order proxy for the marginal reward through
   a *nonlinear* AR, and at the FVE we can afford it is noisy (residual is large
   in nearly every direction). Instead, *causally* measure whether **naming `c`
   lowers the AR's reconstruction error**, with read-calibrator / irrelevant /
   absent controls. This collapses Rev-3 Gate-0a + Gate-0b into one text-space
   test and dodges both the linearization and the Σ_q eigendecomposition (itself
   a linear approximation of the nonlinear AR). The linearized `⟨e, Δ_c⟩` and the
   Σ_q overlap are retained as **corroborating** measures (Gate 0′), not the
   headline.
3. **Substrate.** The Rev-4 review argued Gate 0/1/2 should run on a released
   HIGH-FVE NLA (at FVE ~0.3 the residual *measurement itself* is dominated by
   global reconstruction failure). **SUPERSEDED by D2** (`decisions.md`): we chose
   syvb's Qwen3-8B (~0.3 FVE) for ALL gates — model-matching + frugality +
   runnability — knowingly accepting noisier measurement. This rationale survives
   as *why the high-FVE Qwen2.5-7B NLA is the fallback* if Qwen3-8B is too noisy,
   and as *why Gate 0 uses the difference-based counterfactual-mention test* rather
   than the raw residual.
4. **Faithfulness primacy moves from probe-consistency to a behavioral
   consequence.** Subspace-(i) whitening optimizes `Δ_c`-alignment, so certifying
   faithfulness by `Δ_c`/probe projection is partly circular — a confabulated
   gain passes by construction. Anchor the primary faithfulness check in
   something the reward did **not** see: does the activation the explanation now
   names as `c` actually correspond to `c`-consistent *behavior* (held-out
   continuations, or CAA-steering with `Δ_c`)? Probe-consistency demotes to
   corroboration.
5. **Scale control: the GRPO-relevant invariant is per-group reward std, not the
   trace.** `tr(MΣ_q)=tr(Σ_q)` fixes expected reward magnitude, but GRPO
   normalizes advantages by within-group std, which whitening can reshape
   independently. Keep `a_{α,τ}` but treat the *logged within-group reward std*
   as the real optimization-dynamics control.
6. **A frozen-AR arm is added** to cleanly separate "reward geometry moved the
   AV" from "the AR learned to read `c` from context" (the online-AR sign-test
   alone confounds these because the AR co-trains in both of its arms).
7. **Compute resolved:** Gate 0/1/2 on the **Vast H200** (single card, no
   sharding); Gate 3 RL on the **Lambda 2×H100** (vLLM rollout/trainable split).
8. **Citations verified** (`literature.md`): all cited arXiv IDs are real; the
   audit's re-scoping holds; full-text page-quotes still to be read on the GPU box.

Retained from Rev 3 (still load-bearing): the reward-geometry derivation;
covariance in the NLA's *normalized* space estimated on the *generic*
distribution (not the concept-pair set); shrinkage/power whitening (not top-k);
the spectrum-preserving shuffled-Σ control; subtype robustness for umbrella
labels; "meta ⇒ low-variance ⇒ unread" is falsified as a motivation and is not used.

## 1. One-paragraph summary
A released NLA verbalizes some concepts an activation contains and silently omits
others that are linearly decodable from the same activation (corrigibility: probe
AUROC ~0.97, BoW ~0.50, NLA read ~0, on real activations). **Hypothesis:** those
concepts have ~0 marginal reconstruction value — the AR already reconstructs them
from contextual prose, so the residual along the concept direction is ~0, so
naming them does not lower reconstruction error and the verbalizer is never
rewarded for it. **Decisive first test (Gate 0):** does *appending a sentence
that names `c`* lower the AR's reconstruction error on present-`c` activations,
beyond irrelevant-mention and absent-`c` controls, while a read concept's mention
demonstrably *does* help? If no → reconstruction-redundancy, whitening is
geometrically blocked, publishable null. If yes → there is reducible residual a
reweighted reward can exploit; **intervention:** a scale-normalized
shrinkage/power-whitened reward up-weighting low-variance directions, swept over
strength, with a behavioral faithfulness battery so a read-rate gain cannot be
confabulation. **Deliverable:** a panel result — read-rate vs ordinary directional
FVE as whitening strength varies, across several decodable-but-unread concepts —
or a clean negative from Gate 0/0′.

## 2. Background
See `nla-method-notes.md`. The one fact the whole design rests on: the released
reward compares *individually L2-normalized* vectors, so it is **direction-only**
(`reward = -2(1-cos)`), and the AR is the live reward model (truncated backbone +
`Linear(d,d)` head, read at the last token).

**Reward-relevant quantity (derivation, verified).** With `r = -‖e‖²`,
`e = q(h) - q(ĥ(z))`, a mention that nudges the reconstruction by `δ` changes the
reward by `Δr = 2⟨e,δ⟩ - ‖δ‖² ≈ 2⟨e,δ⟩`; along the concept direction,
`Δr ∝ ⟨e, Δ_c⟩`. The operative quantity is the **alignment of the baseline
reconstruction residual with `Δ_c`**, not the concept's variance or energy. A
direction the AR already reconstructs (from correlated context) has residual ≈ 0
and earns ~no marginal reward for being named.

## 3. Hypotheses (stated so they can be false)
- **H1 (mechanism).** Some decodable behavioral variables have ~0 residual on the
  NLA's distribution — the AR reconstructs `Δ_c` from contextual prose, so naming
  `c` does not lower MSE. Across the panel, **mention-induced MSE reduction**
  (Gate 0) predicts read-rate: redundant concepts are omitted, residual-bearing
  ones verbalized. **False if** mention-induced MSE reduction does not track
  read-rate across the panel.
- **H2 (intervention, with precondition).** A scale-normalized whitening `M`
  raises the marginal reward for naming `c` — `eᵀMΔ_c` rises more for targets than
  for read controls — and causes the AV to verbalize the targets at tolerable
  ordinary-FVE cost, with gains faithful to held-out *behavior*. **Geometrically
  blocked (null by construction)** where the residual and `Δ_c` do not co-occupy
  the low-`λ` eigendirections `M` up-weights (Gate 0′). **Empirically false** if
  the lever exists but no strength raises target read-rate before FVE collapses,
  or gains fail the behavioral faithfulness battery.

**Motivation wording (use this):** not "meta concepts are low-variance therefore
omitted" (falsified by the NLA paper's eval-awareness positives), but "these
specific omitted variables are reconstructable from context, so naming them earns
no reward; does reweighting the reward recover them?"

**Honest confound (restated).** "Low residual" and "not output-coupled" are
probably the *same* axis, not two separable causes — refusal is read because it is
output-coupled → AR cannot reconstruct the post-refusal trajectory from prose →
large residual → naming pays. So H1 does not need the intervention to disambiguate
two causes; the intervention's genuine job is narrower: whether reweighting the
*reward geometry* (vs. the AR simply learning to read `c`) moves read-rate (Gate 3
frozen-AR vs online-AR arms). Report residual/output-coupling collinearity as a
limitation, with an explicit output-coupled control class.

## 4. Compute & substrate
**Substrate (D2): syvb's Qwen3-8B L24 NLA for ALL gates** — one model end-to-end
(model-matching + frugality + runnability). This supersedes the earlier
recommendation to measure on the released high-FVE Qwen2.5-7B-Instruct NLA; the
accepted tradeoff is ~0.3 FVE (noisier Gate-0/1, an *unmeasured* read/unread gap →
Gate 1 is a real fork). The released high-FVE NLAs are kept as an optional
cross-check / fallback (`configs/models.yaml`).

Compute: Gate −1/0/1/2 are inference (8B AV+AR fits one card) → **Vast H200** or a
single **Lambda** card. Gate 3 (RL) → **Lambda 2×H100** (vLLM rollout/trainable
split), continuing from syvb λ=0 base. **Prerequisite:** `scripts/check_model.py`
must confirm the model is post-trained (not `-Base`) before probing behavioral
concepts; if base-like, the panel pivots to truth/topic/factual concepts (D2).

## 5. Gate structure (run strictly in order)

### Gate −1 — Validate the measurement instrument (no model training)
The probe/`Δ_c` is used to define the target, measure the residual, and check
faithfulness; validate it before any of that. Concretely: (a) probe AUROC ≫ BoW
(rules out lexical leakage); (b) length-residualize, shuffled-label ≈ 0.5,
pair-level CV (in `concepts.py`); (c) **transfer**: does the probe direction
survive on held-out *naturalistic* activations where the disposition arises
without paired-prompt scaffolding? **Decision:** if `Δ_c` fails transfer,
"decodable" is partly construction leakage and "unread" may be *correct behavior*
— re-pick the target before spending anything else.

### Gate −1b — Causal screen (targets must be *used*, not just decodable) [D4]
A decodable-but-non-causal Δ_c is an epiphenomenal correlate; an NLA omitting it is
*correct* behavior, not a failure — and whitening the reward to force its
verbalization would be optimizing for confabulation. So for each **target**, screen
that Δ_c is causally efficacious **at the read layer L24** before treating "unread"
as interesting. Method (CAA/Arditi/GoT, which our datasets come from): steer
±α·Δ_c into the residual stream at L24 over a magnitude sweep and confirm the
c-behavior shifts in the predicted direction (refusal: +Δ_c induces refusal on
harmless / −Δ_c suppresses on harmful; sycophancy: +Δ_c raises agreement-with-wrong;
truth: flips true/false treatment). **Controls (load-bearing):** a norm-matched
**random direction** must NOT reproduce the effect, the shift must be **specific**
(c-behavior moves more than unrelated behavior), and generations must stay
**coherent** (not broadly damaged). **Decision:** efficacious + specific + coherent →
valid interesting target (H1 becomes decodable-AND-causal-but-unread). Only broad
damage or no effect → Δ_c is not a usable causal direction at L24 → re-pick or
re-layer. Read controls (refusal/truth) are already causal in the literature; the
screen is primarily for targets. The *rigorous* behavioral causality check stays at
Gate 3b, validating the gains.

### Gate 0 — Counterfactual mention (THE decisive cheap test; Vast, inference only)
Implemented in `gate0_counterfactual.py`. For present-`c` real activations with
baseline AV explanation `z`, compare `MSE(AR(z), h)` to `MSE(AR(z + "names c"), h)`
and to `MSE(AR(z + "names irrelevant"), h)`; track the `Δ_c`-projection of the
reconstruction moving toward target. **Controls:** read-concept **calibrator**
(its mention must help, else setup broken), **irrelevant** mention (must not
help), **absent-`c`** (must not help). **Decision rule** (encoded in `decide()`):
- `LEVER_EXISTS`: present-`c` mention lowers MSE, beyond irrelevant & absent, and
  moves the projection toward target → proceed.
- `REDUNDANCY`: present-`c` mention ~0 while calibrator helps → STOP, publishable
  null (whitening geometrically blocked).
- `BROKEN_SETUP`: calibrator mention doesn't help → fix AR loading / last-token
  readout / normalization before interpreting.

**Why this is better than the linearized residual:** it asks the *actual nonlinear
AR* whether the *actual word* helps, differenced against baseline, so it is robust
to "the AR is bad everywhere so `⟨e, Δ_c⟩` is nonzero for everyone." Its own
sanity check (the calibrator) reveals a broken pipeline before any result is read.

### Gate 0′ — Linearized residual + Σ_q overlap (corroboration, only if Gate 0 = LEVER)
Compute `⟨e, v_c⟩` with a z-score against random directions (`metrics.residual_
alignment_z`; random ~0, concept ≫0). Estimate `Σ_q` on the **generic**
distribution (normalized space), eigendecompose, plot per-concept
`(eᵀu_i)(Δ_cᵀu_i)` vs `λ_i`: the lever needs co-occupancy in **low-`λ`**
directions. Also report static proxies (projected variance, present-energy,
separability, CCA-against-continuation) as confounded context, and do the
subtype-stability check for umbrella labels (corrigibility, sycophancy). Jacobian
caveat (2605.14258): if ambiguous, the sharper resolvability test projects onto
the propagation Jacobian's active subspace, not Σ_q — note as a limitation, do not
claim the Σ_q spectrum settles it.

### Gate 1 — Panel benchmark on the chosen model
**Decided panel (D3):** read controls `refusal` (also the Gate-0 calibrator) +
`truth_value`; targets `sycophancy/false_agreement` + `deception/withholding`;
`corrigibility` optional (asks only whether the Qwen2.5 finding generalizes to
Qwen3). For each: probe AUROC, BoW, and judge-scored read-rate on present-polarity
real activations (validate the judge on obvious ± first). Report the matching
variables in `concepts.yaml`. The panel is the unit of analysis; one concept is an
anecdote. **Probe the disposition, not the output** (D3): sycophantic-vs-genuine
agreement; knowing-falsehood-vs-honest — else the direction is output-coupled and
H1 predicts it is read, not a valid target.

### Gate 2 — Baseline NLA
Score the released `base` AV+AR on the Gate-1 panel: controls read, targets ~0 (the
released pattern). Verify one FVE number against the release. **FVE-headroom
check:** confirm syvb's baseline FVE (~0.3) is high enough that target read-rate
isn't already in the metric noise floor; if Gate-3 deltas turn out ambiguous, this
is the trigger to fall back to the high-FVE NLA (D2).

### Gate 3 — Whitened-reward RL (Lambda)
Continue RL from `base` with the whitened reward (§6). **First** run the
disambiguation arms at one `α`: (a) reward-only + **frozen AR**, (b) reward-only +
online AR, (c) reward + whitened AR. (a) isolates pure reward-geometry effect on
the AV; (b) vs (c) is the sign-test on AR whitening (Rev-3 change 3 predicts
(b)≥(c) on target read-rate). Then sweep `α ∈ {0,0.25,0.5,0.75}`, `τ` fixed,
scale-preserved. **Mandatory per-arm logging:** mean reward, **within-GRPO-group
reward std**, critic loss under standard & under `M`, response length (cap fixed
across arms), extraction-failure rate.

### Gate 3b — Faithfulness battery (required for any read-rate gain)
1. **(PRIMARY) Behavioral consequence** — grounded in something the reward did not
   optimize: when an explanation names `c`, does the underlying activation drive
   `c`-consistent behavior on held-out continuations (or under CAA-steering with
   `Δ_c`)? A naming that doesn't predict behavior is decorative.
2. Probe consistency (corroboration; circular with subspace-(i) whitening, so not
   primary): conditional on naming `c`, is the `Δ_c` projection higher than on
   matched non-naming examples?
3. Context-hallucination audit: do modified explanations invent prompt facts more
   than baseline?
4. Paraphrase / reorder / translation invariance (anti-steganography).

## 6. Intervention: scale-normalized shrinkage/power whitening
`M_{α,τ} = a_{α,τ} · U diag((λ_i+τ)^{-α}) Uᵀ`, `Σ_q = U diag(λ_i) Uᵀ` (normalized
space, generic distribution). `α=0` → identity (continuity check); `α∈(0,1]` →
partial→shrinkage whitening; `τ` = a **predeclared** spectral floor (median λ, or
elbow, or shuffled-split noise floor — pick one and preregister); `a_{α,τ}` set so
`tr(M_{α,τ}Σ_q)=tr(Σ_q)` (necessary, not sufficient — see §0 change 5). Whitened
reward `r_M = -eᵀMe`. **Concrete swap point (D5):** replace the
`F.mse_loss(pred_n, gold_n)` call in `score_with_critic()`
(nanoNLA `train_rl_self_contained.py` ~L550-580) with `((pred_n-gold_n) @ M_half).pow(2).mean()`
i.e. `r_M = -‖M^{1/2}(pred_n-gold_n)‖²`; mirror the same swap in the `--train-critic`
co-training loop (~L1250-1310). No separate reward.py exists.

**Closed-form precondition.** `Δr ∝ eᵀMΔ_c = Σ_i (λ_i+τ)^{-α}(eᵀu_i)(Δ_cᵀu_i)`.
`M` only reweights existing per-eigendirection products; it cannot create one that
is zero. So whitening helps `c` only where the residual and `Δ_c` co-occupy
low-`λ` directions (Gate 0′ precondition). Where the residual is ~0 along `Δ_c`
(redundancy, Gate 0 = REDUNDANCY), no `M` at any `α` creates naming reward.

**AR whitening may be the wrong sign.** Whitening the AR pushes it to reconstruct
`Δ_c` from context better → shrinks the residual the AV would be paid to name. The
Gate-3 arms test the sign; default to reward-only unless (c) beats (b).

**Targeted up-weighting & the unsupervised/supervised honesty.** Blanket `Σ_q^{-1}`
is the upper-aggressiveness ablation, not the headline. Up-weight a subspace:
(i) span of concept/CAA directions — uses labels → barely-disguised supervision →
the honest baseline is a *direct supervised naming reward*, and whitening-within-
the-subspace must beat it; (ii) CCA-against-the-model's-continuation (2604.20682's
method) — unsupervised, but these are output-coupled directions, already read, so
likely null for the targets. Preregister which claim you are making; the
distinctive "unsupervised" contribution survives only if (ii) moves targets or (i)
beats direct supervision.

## 7. Pre-registered success criteria
Decide before looking at Gate-3 numbers. Two fidelity metrics, but only **ordinary
directional `FVE_std`** is the cost budget (`FVE_M` mostly shows the model
optimized the new objective). **Precondition:** Gate 0 = LEVER_EXISTS and Gate 0′
shows low-`λ` co-occupancy; else the project stops with a negative finding.
- **Scale ("promising"):** target read-rate +≥0.25 absolute on ≥2 targets;
  `eᵀMΔ_c/‖e‖‖Δ_c‖` rises more for targets than controls; read-control read-rate
  drops ≤0.10; `FVE_std` drops ≤0.08 (or stays above warm-start); gains pass Gate-3b
  **behavioral** primary; spectrum-preserving shuffled-Σ control does *not*
  reproduce the gain; across ≥2 seeds.
- **Pivot:** gains only past the `FVE_std` knee → "fix exists, costs FVE"; try other
  `τ`/`α` or the reward-only arm.
- **Stop:** no read-rate movement before FVE collapse and shuffled-Σ behaves the
  same → whitening isn't the lever (Gate 0/0′ should have predicted this).

Hard rule: "explanations look more interpretable" is **not** a signal. The signal
is the baseline-vs-modified read-rate delta that passes Gate-3b behavioral
consistency, or it is nothing.

## 8. Risks & honest failure modes
- **No residual to amplify** (modal): Gate 0 kills it for ~half a day on the GPU
  box before any training. Ending here cheaply is a *good* outcome.
- **AR whitening pushes read-rate the wrong way:** Gate-3 arms detect; default
  reward-only.
- **Unsupervised framing collapses to supervision:** benchmark subspace-(i) vs a
  direct supervised naming reward, or the selling point is gone.
- **Faithfulness circularity:** primary check is behavioral, not probe (§0/4).
- **Low-FVE substrate corrupts the *measurement*:** accepted risk under D2 (Qwen3-8B
  ~0.3 FVE). Mitigations: the difference-based Gate-0 test; Gate 1 must *establish*
  the gap; fall back to the released high-FVE NLA if too noisy.
- **Confabulation** (worst): a whitening "success" that emits false mentions — the
  behavioral battery is mandatory.
- **Optimization-dynamics confound:** logged within-group reward std + scale-
  preservation; trace alone is insufficient.
- **Probe leakage:** Gate −1 transfer check; if `Δ_c` doesn't transfer, "unread" may
  be correct.
- **Citation scope:** see `literature.md`; read full PDFs before any writeup.
- **syvb provenance:** tail-driven ~0.3 FVE, some hard failures; use λ=0; reproduce
  one results number.
- **Heterogeneous umbrella labels:** Gate-0′ subtype-stability check.
- **Scope creep:** this intervenes on the reward and reads the explanation; it does
  not open the AV's internal circuit. Keep it there.

## 9. Decision tree
```
Gate -1  validate Δ_c (AUROC≫BoW, shuffled≈0.5, length-resid, TRANSFER to naturalistic)
  ├─ fails transfer ─► RE-PICK TARGET ("unread" may be correct behavior)
  └─ valid ─► Gate 0  counterfactual mention (Qwen3-8B syvb NLA [D2], inference)
       ├─ calibrator mention doesn't help ─► BROKEN_SETUP: fix AR/readout/normalization
       ├─ present-c mention ~0 (calibrator helps) ─► STOP: REDUNDANCY, publishable null
       └─ present-c mention helps, specific, grounded ─► Gate 0' (residual z + Σ_q overlap)
            ├─ co-occupancy HIGH-λ / absent ─► STOP/rethink (whitening can't reach it)
            └─ co-occupancy LOW-λ (lever) ─► Gate 1 panel ─► Gate 2 baseline (check FVE headroom)
                 └─► Gate 3 (Lambda): arms {frozen-AR, online-AR, whitened-AR} at one α,
                       then scale-preserved sweep; per-arm logging; Gate 3b BEHAVIORAL on any gain
                       ├─ read-rate↑≥0.25 on ≥2 targets, selective, controls ok, FVE_std drop≤0.08,
                       │   passes behavioral faithfulness, shuffled-Σ does NOT reproduce ─► SCALE
                       ├─ gains only past FVE knee ─► PIVOT (τ/α; reward-only)
                       ├─ gains fail Gate 3b ─► confabulation, not a win; rethink reward
                       └─ flat until FVE collapse + shuffled-Σ same ─► STOP, report negative
```

## 10. References
See `literature.md` for verification status. Load-bearing: **2604.20682**
(variance≠importance; half-warns against non-uniform reweighting), **2508.16929**
(residual stream ~90% rank — undercuts a Σ_q noise-floor), **2605.14258** (Jacobian,
not Σ_q), **2605.08740** (causal dimensionality — newly surfaced, to read). Method:
the NLA paper + released code. Supporting (verify before citing): 2605.01609,
2601.16644, 2509.21305 (sycophancy subtypes), CAA (Rimsky et al.).

## 11. What this design deliberately does NOT claim
- Not that meta/behavioral concepts are low-variance (falsified prior).
- Not that reward-relevance equals variance/energy (it is the residual `⟨e,Δ_c⟩`).
- Not that the intervention can work (geometrically blocked unless the residual
  co-occupies low-`λ` directions with `Δ_c`).
- Not that whitening the AR helps (may shrink the residual; arms decide).
- Not that the unsupervised framing survives if the working version uses
  label-defined subspaces (then the baseline is direct supervision).
- Not that higher FVE (or `FVE_M`) is the goal (`FVE_std` is a bounded cost).
- Not that a read-rate gain is a win unless it passes **behavioral** faithfulness.
- A faithful small reproduction is not a result; the result is the gap between two
  NLAs differing only in reward geometry, on a panel, that survives behavioral
  faithfulness — or a clean Gate-0/0′ negative, which is itself publishable.
```
