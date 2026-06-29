# Read-only diagnostics on the locked §7 sweep (no training, no re-extraction)

Two "free" diagnostics that run on the **existing** L19–L29 bank / AR checkpoints. They
do **not** retrain, do **not** touch the locked sweep numbers, and write only their own
JSON/PNG. Each states its hypothesis, the main alternative, the broken-setup sanity check,
the verified assumptions, and the decision rule **up front**.

Both reproduce a **locked published number** as a built-in correctness gate, so a wrong
column / template / normalization fails loudly instead of producing a plausible-but-wrong
curve.

---

## Verified assumptions (read from the code, not assumed)

| assumption | value | source |
| --- | --- | --- |
| loss/FVE space | `u = normalize_activation(a, mse_scale) = mse_scale · a/‖a‖` — **row-wise** projection to the sphere of radius `mse_scale` | `nla/schema.py:135` |
| `mse_scale` | `ar_meta["mse_scale"]` or `√d_model` = **√4096 = 64** | `evaluate_e2e.py:121`, `eval_ar_gold`, `results/stage0_L24_30k.json` |
| FVE denominator | `compute_predict_mean_baselines(...)[1] = mse_rawvar = E_rows E_dims (u − mean_u)²  = trace(Cov(u))/d` | `evaluate_e2e.py:368`, `schema.py:155` |
| locked baselines (rl_test) | prev **0.5970** · centre **0.5630** · next **0.5663** | `EXPERIMENT_REPORT.md §i` |
| raw bank columns | `activation_L{19..29}`; `activation_centre == activation_L24` | `build_sweep.py:59`, `regenerate_*:62` |
| AR gold text | embedded in the AR `prompt` column: `"Summary of the following text: <text>{expl}</text> <summary>"`, tokenized `add_special_tokens=False`, **last-token** tap | `datasets.py:313`, `evaluate_e2e.ar_sqerr` |
| AR-gold ceiling (full expl) | ar_test overall **62.4** · ar_dev **62.8** (3-tap AR) | `EXPERIMENT_REPORT.md §C` |

Key consequence: the √d normalization is **row-wise**, not a global scalar — so it can
both suppress (near-constant massive dims → low variance after norm) and concentrate
(everything else compressed) variance. Which one happens is the empirical question in #3.

---

## #3 — `variance_audit.py` (anisotropy / massive-activation audit; pure numpy, no GPU)

**Hypothesis:** a few outlier dims (Qwen attention-sink / massive activations) or a few
eigen-directions dominate `trace(Cov(u))`, so FVE / AR-gold mostly measure semantically-inert
giant coordinates. **This is upstream of #1/#2/#4.**

**Decomposes**, in the √d space the loss actually sees: per-coordinate variance shares,
the **eigenbasis of Cov(u)** (participation ratio, top-k share), where the raw "massive"
dims' variance goes after row-norm, and per-row/per-doc residual concentration.

**Broken-setup gates (built in):** `‖u‖ == √d` every row; `E[u_i²] == 1.0` exactly;
`Σ var_i / d == baseline`; and the reproduced baseline for L24 compared to the **locked
0.5630** (run `--centre-col` on `$SWEEP/rl_test_local.parquet` to hit it exactly).

**Decision rule (pre-registered, off the eigenbasis of Cov(u)):** let `f5` = top-5
eigen-direction share, `PR` = participation ratio.
- `f5 > 0.50` → **CONCENTRATED**: re-read #1/#2/#4 in a whitened (Σ^−1/2) / dim-clipped
  space; the locked numbers stay as-is.
- `f5 < 0.25` and `PR > d/8` → **DIFFUSE**: row-norm neutralized the concern; proceed.
- otherwise **AMBIGUOUS**: prefer a whitened re-read for the headline contrasts.

```bash
python -m multilayer_nla.variance_audit --selfcheck                       # preflight (numpy only)
python -m multilayer_nla.variance_audit --bank $SWEEP/rl_test_local.parquet \
    --centre-col --out-json /tmp/cal_L24.json                             # calibrate -> 0.5630
python -m multilayer_nla.variance_audit --bank $REGEN --layer 24 \
    --out-json $DATA/variance_audit_L24.json                              # the audit
```
Memory: full bank (~300k×4096) is ~30 GB fp64 peak; use `--max-rows` / `--no-eigh` to cap.

---

## #1 — `gold_rd_curve.py` (gold-explanation rate–distortion FVE(L); GPU/H200)

**Hypothesis:** the text channel is rate-limited (more/denser tokens would raise FVE).
**Alternative:** model-limited (the AR saturates; widening the channel is wasted).

**Test:** truncate the **gold** explanation to a token budget, reconstruct with the **same
shared AR**, read FVE(L). Sweeping *gold* (not AV output) isolates text→activation R–D from
verbalizer quality. Two modes run together: `sentence` (whole-sentence prefix ≤ L —
grammatical, in-distribution; the clean curve) and `hard` (exactly L tokens — may dangle;
the OOD confound control). The x-axis is **realized mean tokens** (a budget proxy).

**Broken-setup gates (built in):** `fill_ar_prompt(parse(prompt)) == prompt` per row
(byte-identity of the parse/template path, no model needed → use `--dry-run`); and the
**L=full point must reproduce the locked 62.4** (ar_test) / 62.8 (ar_dev) — it reuses
`evaluate_ar`/`_per_tap_baselines` verbatim, so a mismatch means the harness diverged.

**Decision rule (pre-registered, off the `sentence` curve):** `slope_tail = FVE(full) −
FVE(L≤128)`.
- `slope_tail > 0.5pp` (CIs separate) → **RATE-LIMITED**: favor objective/denser-label/
  longer-budget levers.
- flat (`≈0`, CIs overlap) before full → **MODEL-LIMITED**: bottleneck is elsewhere
  (verbalizer / AR capacity); widening the channel is wasted.
- **confound:** a `hard`-only early flatten the `sentence` curve does not share = OOD
  mis-parsing, not saturation → read `sentence`.

```bash
python -m multilayer_nla.gold_rd_curve --selfcheck                        # preflight (no deps)
python -m multilayer_nla.gold_rd_curve --base-ckpt Qwen/Qwen3-8B \
    --eval-parquet $SWEEP/ar_test.parquet --dry-run                       # tokenizer-only preflight
python -m multilayer_nla.gold_rd_curve --base-ckpt Qwen/Qwen3-8B \
    --ar-ckpt $CKPT/ar_3tap_bs256e_3k/iter_0003000 \
    --eval-parquet $SWEEP/ar_test.parquet \
    --out-json $DATA/rd_gold_test.json --plot $DATA/rd_gold_test.png      # the curve
```
Run `#3` first — it is upstream of the metric `#1`/`#2`/`#4` all read.
