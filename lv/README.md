# Making-LV-Explainers

Research code for studying **why Natural Language Autoencoders (NLAs) verbalize
some activation content and silently omit other linearly-decodable content**, and
whether a **variance-equalized (whitened) reconstruction reward** can recover the
omitted concepts.

LV-Explainers = the latent-vector → natural-language *explainers* (NLA AVs). The
original README stub's question — *"Variance isn't important and can I make
targeted fixes?"* — is the project thesis: NLA optimizes/reports reconstruction
**variance explained (FVE)**, but FVE measures whether the AR can rebuild the
activation's *direction*, not whether the explanation names the *causally
relevant* concept. We test whether reweighting the reward away from variance
toward the reducible **residual** makes the explainer name decodable-but-unread
concepts — and we try hard to falsify that before believing it.

## Status
Design + verified pure-math core. **No model has been trained.** The decisive
first step (Gate 0) has not been run — it needs a GPU box (see below).

## Operating contract
`CLAUDE.md` is the research contract (skeptical-by-default, falsification-first,
claims graded observation→pattern→supported→interpretation→speculation, baselines
before mechanism, measurement tools — including NLA explanations — treated as
unvalidated instruments). It is auto-loaded every session.

## Repo layout
```
CLAUDE.md                        research operating contract (auto-loaded)
docs/
  experiment-design.md           Rev 4 — OPERATIVE spec (read this)
  experiment-design-rev3.md      Rev 3 — preserved verbatim (provenance)
  nla-method-notes.md            NLA method & released-code reference
  literature.md                  citation audit (all cited arXiv IDs verified real)
  compute.md                     Vast/Lambda plan + environment setup
configs/
  concepts.yaml                  the read/unread concept panel
  models.yaml                    released checkpoints + which gate runs where
src/lv_explainers/
  metrics.py                     normalize, MSE=2(1-cos), FVE, residual alignment (+self-test)
  concepts.py                    delta_c (CAA), numpy probe, AUROC, CV (+self-test)
  nla_io.py                      sidecar loader (CPU) + AR/AV/extractor (GPU, lazy torch)
  gate0_counterfactual.py        THE decisive test: does naming c lower AR MSE? (+self-test)
scripts/
  run_tests.sh                   pure-math self-tests (no GPU)
  setup_env.sh                   install deps (--gpu for model deps)
```

## Quickstart (here, CPU-only)
```bash
pip install -r requirements.txt
bash scripts/run_tests.sh        # asserts the measurement instruments are correct
```
The self-tests verify `MSE == 2(1-cos)`, the FVE geometry under sphere-
normalization (`FVE ≈ 2cos-1`), residual-alignment z-scores (random ~0, aligned
≫0), the probe (real ≫ shuffled≈0.5), and the Gate-0 decision logic on three
constructed worlds. They pass locally without a GPU.

## Substrate & compute (decisions in `docs/decisions.md`)
**Substrate: syvb's Qwen3-8B L24 NLA for all gates** (one model end-to-end;
accepted tradeoff ~0.3 FVE → noisier, gap unmeasured → Gate 1 is a real fork).
This container is CPU-only with restricted egress (model work is impossible here).
Real runs go to two GPU boxes (`docs/compute.md`):
- **Vast H200** — Gate −1/0/1/2 (inference & scoring); 8B AV+AR fits one card.
- **Lambda 2×H100** — Gate 3 (whitened-reward RL), vLLM rollout/trainable split.

First on a box: `python scripts/check_model.py …` — confirm post-trained, not `-Base`.

## Panel (decided)
Read controls `refusal` (Gate-0 calibrator) + `truth_value`; targets
`sycophancy/false_agreement` + `deception/withholding`; corrigibility optional.
**Probe the disposition, not the output** (sycophantic-vs-genuine agreement;
knowing-falsehood-vs-honest).

## The plan in one screen
```
Gate -1  validate Δ_c (AUROC≫BoW, shuffled≈0.5, TRANSFER to naturalistic) — instrument first
Gate 0   counterfactual mention: does naming c lower AR MSE? (controls: calibrator/irrelevant/absent)
           REDUNDANCY → STOP (publishable null)   LEVER_EXISTS → continue   BROKEN_SETUP → fix
Gate 0'  corroborate: residual z-score + Σ_q low-λ co-occupancy
Gate 1   panel benchmark (targets vs read controls) on Qwen3-8B — establishes the gap exists here
Gate 2   baseline NLA scores (check FVE headroom)
Gate 3   whitened-reward RL sweep; arms {frozen-AR, online-AR, whitened-AR}; per-arm logging
Gate 3b  faithfulness — BEHAVIORAL consequence primary (not probe; probe is in the loop)
```
Full rationale, hypotheses (stated falsifiably), the whitening family `M_{α,τ}`,
its closed-form precondition, success criteria, and risks: `docs/experiment-design.md`.
