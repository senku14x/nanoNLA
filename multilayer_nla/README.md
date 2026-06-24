# Multi-layer NLA — FineWeb architecture-validation phase

Self-contained experiment package for the **FineWeb Multi-Layer Natural Language
Autoencoders** plan (Revision 2, design frozen). Everything here reuses the
stable `nla/` core (keyed-RNG sampling, `HFExtractor`, storage) and never mutates
it.

**Primary question.** Does a coherent contiguous activation patch
`[a^(l-1), a^(l), a^(l+1)]` preserve local cross-layer information through a
natural-language bottleneck better than matched controls?

**Non-goals for this phase (plan §15).** No safety / deception / refusal /
corrigibility / agentic claims, no faithfulness claim, no claim that this
reproduces the official NLA. The only intended claim is the architectural one
above, on passive FineWeb activations.

---

## What's built here (and what isn't)

This is the **NLA-free pre-training** slice — plan execution-order steps 1–2
(§12.5). It exists to answer the pre-registered **Gate 0** for ~probe-fitting
cost *before* any AV/AR/RL GPU time.

| Script | Plan ref | Runs on |
|---|---|---|
| `extract_multilayer.py` | §4, §6.1, §12.5.1 | H200 (loads Qwen3-8B) |
| `verify_center_parity.py` | §12.5.1 | H200 |
| `headroom.py` | §5.1, §8, §13 | CPU/H200 (cached patches; MLP wants a GPU but runs on CPU) |
| `manifest.py` | §12.6 | — (provenance helper) |
| `tests/test_fve_math.py` | §5.1.6, §6.3 | anywhere (numpy only) |

**Not built yet** — Stage 2+, gated behind a GO verdict: the AV 3-slot
injection, the multi-tap AR (heads tapping their own depth, §6.2 Rev 2),
Conditions A–D (§7), and RL. Do not start these until Gate 0 returns GO.

---

## Frozen execution order (plan §12.5)

```
1. extract three contiguous layers  ── extract_multilayer.py
   └─ verify center tap == legacy L24 extraction (bitwise)  ── verify_center_parity.py
2. geometry + ridge + mean-update + center-only headroom  ── headroom.py  → GATE 0
   ─────────────────────────────  (this is where the NLA-free phase ends)  ─────────────
3. three-slot AV injection; coherent vs duplicate, 20 SFT steps        [Stage 2 — not built]
4. verify marker count / slot order / no marker leakage / loss down    [Stage 2 — not built]
5. multi-tap AR; 5k-doc full-chain smoke                               [Stage 2 — not built]
6. 100k corpus + 500-step RL                                           [Stage 3 — not built]
```

---

## Run recipe (steps 1–2, on the H200)

Set `PYTHONPATH` to the repo root so `import nla` / `import multilayer_nla` resolve.

```bash
export PYTHONPATH=/path/to/nanoNLA
OUT=/data/mlnla/qwen3_8b_L24

# --- Step 1a: extract the 3-layer patch (RAW vectors; center=24 -> {23,24,25}) ---
python -m multilayer_nla.extract_multilayer \
    --base-model Qwen/Qwen3-8B \
    --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
    --corpus-length 100000 --positions-per-doc 10 \
    --center-layer 24 --seed 42 \
    --extractor-kwargs '{"batch_size": 12, "max_length": 4096}' \
    --output $OUT/base_multilayer.parquet
#  smoke: --corpus-length 2000

# --- Step 1b: DECISIVE plumbing gate — center tap must match legacy L24 (bitwise) ---
python -m multilayer_nla.verify_center_parity \
    --base-model Qwen/Qwen3-8B \
    --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
    --center-layer 24 --n-docs 16 --atol 0
#  -> "[parity] PASS" required before trusting any FVE number.

# --- Step 2: headroom probe -> GATE 0 verdict ---
python -m multilayer_nla.headroom \
    --parquet $OUT/base_multilayer.parquet \
    --out-json $OUT/headroom_L24.json \
    --dev-frac 0.2 --seed 0
```

The sidecar (`$OUT/base_multilayer.parquet.mlnla_meta.yaml`) is auto-discovered by
the probe and folded into the results manifest (plan §12.6).

---

## Gate 0 — the decision (plan §5.1, §13, Rev 2)

The probe predicts the local update `delta = [u^(l)-u^(l-1), u^(l+1)-u^(l)]` from
the center `u^(l)` alone, with four predictors, and reports update FVE
(`FVE_delta = 1 - E||pred-delta||^2 / E||delta - delta_bar||^2`, §8):

| predictor | meaning | by construction |
|---|---|---|
| constant `delta_bar` | the floor | `FVE_delta = 0` |
| ridge | best **linear** center-only | — |
| MLP | best cheap **nonlinear** center-only | — |
| true neighbor | the ceiling | `FVE_delta = 1` |

**Decision variable (Rev 2):** the *unique headroom*
`H_unique = 1 - max(FVE_ridge, FVE_MLP)` on the **dev** split — the update
variance the center cannot recover, i.e. the ceiling on what a coherent patch
could add through the bottleneck.

- **STOP / pivot iff `H_unique < 0.05`** — the center already determines the
  neighbors; there is nothing for the bottleneck to carry.
- **Otherwise GO** — proceed to Stage 2 (Conditions A–D).
- The **MLP–ridge gap** is reported as a *diagnostic* (is the recoverable part
  nonlinear?), **not** the gate.

If torch is unavailable the probe runs ridge-only; since the MLP can only *raise*
the best center-only FVE, ridge-only `H_unique` is an **upper bound** — a
ridge-only STOP is decisive, a ridge-only GO is flagged provisional.

The probe self-asserts the floor (`==0`) and ceiling (`==1`) identities and
√d-normalization on every run; a failure raises before any number is trusted
(plan §5.1.6).

---

## Invariants preserved (CLAUDE.md / plan Rev 2)

- **RAW vectors stored** (`norm="none"`). The AV injects raw `a`; the
  √d-normalized `u = √d·a/‖a‖` is used **only** as AR target and in FVE (§4 Rev 2).
- **Document-level split** — all positions of a doc share one bucket
  (`doc_split_mask`), in both extraction provenance and the probe's dev split.
- **Per-doc keyed RNG** — position sampling reuses
  `nla.datagen.stage0_extract._sample_positions` verbatim, so the multi-layer
  center tap lands on the same positions a single-layer run would (Condition A
  comparability + the parity check).
- **No mutation of `nla/`** — `MultiLayerHFExtractor` subclasses `HFExtractor`;
  the single-layer `extract()` path is untouched so parity can compare both.

---

## Local validation already run (pre-H200)

- `tests/test_fve_math.py` — 6 numpy tests: floor=0, ceiling=1, √d-norm, ridge
  recovers a linear map, doc-level split, `H_unique` definition. **PASS.**
- `headroom.py --selfcheck` — synthetic GO/STOP cases (orthogonal-map neighbors
  vs an independent neighbor); verdicts and identities. **PASS** (incl. torch MLP).
- Probe on a synthetic parquet via the real CLI — streaming load, MLP path,
  JSON + manifest output. **PASS.**
- 3-hook capture on a tiny in-memory Llama — center tap matches a legacy
  single-hook capture **bitwise (diff 0.0)**, three layers distinct (closure
  binding correct). **PASS.**

The extraction and parity scripts themselves run on the H200 (need the real model
+ pinned `transformers==4.57.1`); the parity check is their designed validation.

```bash
# reproduce the local checks anywhere with numpy (+ torch for the MLP path):
python -m multilayer_nla.tests.test_fve_math
python -m multilayer_nla.headroom --selfcheck
```
