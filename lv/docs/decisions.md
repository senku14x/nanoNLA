# Decision log

Live decisions that may override specific recommendations in the design doc.
Newest first. Each entry records the decision, the rationale, and the tradeoff we
are knowingly accepting (per the research contract — record the cost, don't bury it).

## D8 — No lexical check on contrast prompts at Gate -1; verdict = probe vs shuffled (2026-06-23)
**Decision.** `validate_concepts` (Gate -1) judges decodability from **probe AUROC vs
a shuffled-label control** only → `DECODABLE` / `NOT_DECODABLE`. Removed the
lexical-on-prompts gate (and the `LEXICAL_LEAK` verdict).
**Why.** Running a lexical/TF-IDF classifier on the *rendered contrast text* is
confounded by construction — the text encodes the label: two-population sets
(refusal, truth) use entirely different prompts (lexical AUROC ≈ 1.0), and A/B pairs
differ only by the appended answer letter (also ≈ 1.0). So it flagged *every* concept
as `LEXICAL_LEAK`, mislabeling genuine directions (e.g. refusal). It measures the
construction marker, not leakage.
**Where the checks actually go.** Leakage/validity at Gate -1 = **transfer** to
held-out naturalistic activations (separate data, to build). The lexical+semantic
text baselines (D7) keep their home on the **AV explanation** at Gate 0 (redundancy),
where the text is generated, not constructed. `text_baselines.py` is unchanged and
still correct — it was just applied in the wrong place.

## D7 — Replace bare BoW with a tiered text baseline; split the two jobs (2026-06-23)
**Decision.** A bag-of-words baseline is too weak and was doing two different jobs.
Use a **tiered text classifier** and apply the right tier to the right question:
- **lexical floor** = TF-IDF unigram+bigram + logistic (`text_baselines.lexical_auroc`,
  numpy, tested). Replaces raw BoW. Only rules out *trivial lexical* leakage.
- **semantic** = frozen sentence-encoder probe (`embedding_auroc`) and/or an **LLM
  judge** (`llm_judge_decodability`) — the baselines that actually threaten the
  interpretation (catch paraphrase/meaning). NEEDS-GPU/API.
**Two jobs, kept distinct:**
1. *Leakage control* (Gate −1) — does the **activation probe** exploit a surface
   confound? Real tests = shortcut-feature probe (length/position/token-id) +
   transfer-to-naturalistic. The lexical baseline is only a cheap floor here.
2. *Redundancy / decodability-from-text* (H1, Gate 0) — can a **strong reader**
   recover c from the **AV explanation**? If yes, the AR reconstructs c from prose
   and naming earns no reward (the redundancy mechanism). Use the SEMANTIC tier,
   NOT BoW: a concept can saturate meaning with none of its surface words.
**Crucial split (LLM judge, two prompts):** "does the explanation **name** c?"
(read-rate) vs "can you **infer** c from the explanation?" (decodability).
Inferable-but-not-named *is* decodable-but-unread — BoW cannot see this at all.
**Caveat:** validate the judge on obvious ± and check it is not keying on
affirmation language; report both judge modes + the gap, not a single number.

## D5 — Fork ceselder/nanoNLA; repo roles fixed; nanoNLA reality corrections (2026-06-23)
**Decision.** Work on a **fork of `ceselder/nanoNLA`** — confirmed to be the exact
code that trained syvb's Qwen3-8B L24 checkpoints (its RL script loads
`celeste/.../qwen3_8b_L24_{av,ar}_sft_lora_fixed`; "celeste"=ceselder, syvb=HF
mirror). The Gate-3 intervention is a ~5-line change in that code.
**Repo roles.**
- **nanoNLA fork** = the trainer + our whitened-reward patch (Gate 3).
- **Making-LV-Explainers** = experiment harness: measurement/analysis (`metrics`,
  `concepts`, gate drivers, dataset loaders) + design/decisions docs. Calls the fork.
- **`kitft/natural_language_autoencoders`** = reference spec + high-FVE fallback
  checkpoints. NOT forked (heavy Miles/Megatron/SGLang production stack).
**Why a fork (not vendor):** continuing RL from syvb's checkpoints requires the
code's injection/AR/normalization conventions to MATCH the checkpoints; a fork
guarantees that and keeps the reward diff reviewable against upstream.

**nanoNLA reality corrections (override Rev-3/Rev-4 assumptions):**
- **Trainer:** `nla.train_rl_self_contained` (the validated "fixed" path) —
  single-process, **4-bit base + LoRA** (r=128 rsLoRA α=16), **NO vLLM/SGLang**
  (`train_rl_vllm.py` exists as an alt). → **Gate 3 runs on ONE H100/H200, not a
  2×H100 vLLM split.**
- **Reward-swap point:** the `F.mse_loss(pred_n, gold_n)` call inside
  `score_with_critic()` (train_rl_self_contained.py ~L550-580), mirrored in the
  `--train-critic` co-training loop (~L1250-1310). Replace with
  `-‖M^{1/2}(pred_n-gold_n)‖²`. There is **no separate reward.py**.
- **Injection:** nanoNLA default is **Karvonen residual-ADD** at an early layer
  (~layer 2), norm-matched `h' = h + ‖h‖·v/‖v‖`, positioned at the marker token
  (㊗) via 3-token neighbor match — **NOT embedding replacement** (which also exists
  as `inject_at_marked_positions`). → `nla_io.py` injection must be rewritten to
  match whichever syvb's `nla_meta.yaml` specifies. **[CODE TODO]**
- **AR:** `NLACriticModel` = backbone blocks 0..K (num_hidden_layers=K+1), final
  LN→Identity, `value_head=Linear(d,d,bias=False)`, read at last real token —
  matches `nla_io.ARScorer`. ✓
- **normalize_activation** = `v·target_scale/‖v‖` (sqrt_d_model default) — matches
  `metrics.normalize`. ✓ GRPO advantage = per-group `(r-μ)/(σ+1e-6)`. ✓
- **RL flags:** `--av-ckpt`/`--ar-ckpt` (NOT `--resume-from-lora`),
  `--train-critic --critic-lr 5e-5`, `--batch-prompts 16 --group-size 16`,
  `--kl-beta 0.01 --clip-eps 0.2 --lr 1e-5`, 500 steps.
**Open:** which injection method syvb's `nla_meta.yaml` specifies; match the 4-bit
base for checkpoint-faithful continuation.

## D4 — Targets must be causally efficacious at the read layer, not just decodable (2026-06-23)
**Decision.** For *target* concepts, Δ_c must be causally driving behavior at L24,
verified by a lightweight steering screen (new **Gate −1b**) before "unread" counts
as interesting. H1's target definition tightens to **decodable-AND-causal-but-unread**.
**Rationale.** A decodable-but-non-causal direction is an epiphenomenal correlate;
an NLA omitting it is *correct*, not a failure — and whitening to force its
verbalization optimizes for confabulation. The miss is only interesting if the model
*uses* the direction.
**Method.** Steer ±α·Δ_c at L24 (magnitude sweep); require the c-behavior to shift
specifically, beyond a **norm-matched random direction**, with generations staying
**coherent** (a direction that changes behavior is not automatically the concept).
**Head start.** Datasets were chosen from papers that already establish these
directions causally (Arditi refusal necessary+sufficient; CAA sycophancy/
corrigibility steerable; GoT truth causally intervenable) — Gate −1b re-confirms on
Qwen3-8B L24, not from scratch.
**Scope.** Primarily targets; read controls (refusal/truth) are already causal in the
literature. Heavy behavioral causality stays at Gate 3b for the gains.

## D3 — Concept panel: 4 core, corrigibility optional (2026-06-23)
**Decision.** First-run panel = read controls `refusal` (also the Gate-0
calibrator) + `truth_value`; targets `sycophancy/false_agreement` + `deception/
withholding`. `corrigibility` is optional/secondary on this substrate.
**Rationale.** refusal has the strongest cross-family prior (Arditi, incl. Qwen);
sycophancy + deception have BOTH paired data (for Δ_c) AND naturalistic data
(SycophancyEval, MASK) so the Gate−1 transfer check can pass/fail and read-rate
is on-manifold; sycophancy ships released subtypes for the Gate-0′ stability check.
**Corrigibility demoted because** its special status was the *measured* 0.97-decodable
/ ~0-read result on **Qwen2.5-7B L20**, which does NOT transfer to Qwen3-8B; here it
is just another MCQ-only target with the weakest Qwen prior and no naturalistic data.
Keep it only to ask "does the Qwen2.5 finding generalize to Qwen3?" — a fine
secondary question, not the anchor.
**Methods rider (load-bearing).** Probe the *disposition, not the output*: use
sycophantic-agreement vs **genuine-agreement** (both agree) and knowing-falsehood vs
**honest-statement** (MASK's belief/statement split), else the direction is
output-coupled and H1 predicts it is *read* — ruining it as a target.
**Open.** corrigibility in or out of the headline (pending user confirm).

## D2 — Substrate: Qwen3-8B (syvb) for ALL gates, not just RL (2026-06-23)
**Decision.** Use syvb's Qwen3-8B L24 NLA for Gate −1/0/1/2/3 — one model
end-to-end. Supersedes the Rev-4 recommendation to measure on the released
high-FVE Qwen2.5-7B-Instruct NLA.
**Rationale.** Model-matching (one model through the whole pipeline removes the
cross-model hazard), frugality (syvb released AV+AR+RL-LoRA+data → Gate 3 is one
continue-RL run), and runnability (it's what we can stand up on the boxes).
**Tradeoff accepted (recorded, not hidden).** syvb base FVE ~0.3 → Gate-0/1
numbers are noisier and the read/unread gap is **unmeasured** on this model (all
prior observations were Qwen2.5-7B). Consequences: (a) Gate 1 is a genuine fork —
it must *establish* a gap exists here and may return "no gap / reads nothing";
(b) the Gate-0 counterfactual-mention test is preferred precisely because it is a
*difference* (ΔMSE from a mention), which is more robust to low absolute FVE than
the raw residual; (c) the released high-FVE NLAs (Qwen2.5-7B etc.) are retained in
`models.yaml` as an optional cross-check / fallback if Qwen3-8B proves too noisy.
**Prerequisite (gating).** Confirm syvb reads the **post-trained** Qwen/Qwen3-8B,
not Qwen3-8B-Base, before probing behavioral concepts — `scripts/check_model.py`.
If base-like, refusal/sycophancy/corrigibility collapse and the panel pivots to
truth/topic/factual concepts.

## D1 — Decisive first test: counterfactual mention, instrument-validation first (2026-06-23)
**Decision.** Gate −1 (validate Δ_c/probe transfer) runs before any residual
measurement; Gate 0 is the counterfactual-mention test (does naming c lower AR
MSE?), not the linearized ⟨e, Δ_c⟩. Faithfulness primacy is a *behavioral*
consequence, not probe-consistency (which is circular with subspace-(i) whitening).
**Rationale.** See `experiment-design.md` §0; the counterfactual test is causal,
dodges the linearization and the Σ_q eigendecomposition, and self-checks via the
refusal calibrator.

## Open questions (resolve before/early on the GPU box)
- **Validating-NLAs exp2 SGLang reference (optional):** repo is private / out of
  this session's scope and no `add_repo` tool is available, so I can't read it. If
  you want the exp2 SGLang setup used, add it to scope or paste the script. NOTE:
  nanoNLA's validated path uses in-process generation (no SGLang), so this is a
  nice-to-have for AV-serving scale, not a blocker.
- **STEP 0: confirm the syvb repo IDs resolve / are public** — `syvb/nanonla-qwen3-8b-L24-{av,ar,rl-lora}`
  + datasets. Recorded from the Rev-3 spec but NOT independently verifiable from
  this container (HF 403 + not search-indexed). If an ID is off or gated, everything
  stalls. (`huggingface-cli download syvb/nanonla-qwen3-8b-L24-ar` dry-run.)
- Which artifact is the **λ=0 `base`** AV/AR (not a length-penalized run — that
  strips low-variance content and inflates the apparent gap). Separate checkpoint/
  branch, or only a column in `-results`?
- Is the AR shipped as the truncated `NLACriticModel` (K+1 backbone + Linear(d,d)
  value head) that `nla_io.ARScorer` expects?
- corrigibility in or out of the headline panel (D3).
- Does syvb's checkpoint ship an `nla_meta.yaml` sidecar, or do we read `config.json`?
- Is syvb's Qwen3-8B post-trained or `-Base`? (D2 prerequisite; `check_model.py`.)
- Does any target actually go unread on syvb's NLA? (Gate 1 free pre-check.)
