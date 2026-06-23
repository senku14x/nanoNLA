# Length-penalty sweep — analysis

Source: syvb `-results` bundle (7×1000 held-out FineWeb completions, matched by
`idx`), distilled to `sweep_metrics.csv`. Recomputed here from the 7000 rows;
matches syvb's `RESULTS.md`. FVE = 1 − mse/baseline, predict-mean baseline
`mse_nrm = 0.670`. **This sweep does NOT test H1** — read the "Relevance" section.

## Per-model (recomputed, 1000 rows each)

| tag (λ) | mean tok | FVE mean | FVE median | FVE std | note |
|---|--:|--:|--:|--:|---|
| base (no RL) | 126 | 0.532 | 0.618 | 0.283 | AV-SFT+AR-SFT, no RL |
| p0.0 | 127 | 0.585 | 0.661 | 0.258 | RL, zero penalty |
| p0.001 | 92 | 0.592 | 0.664 | 0.242 | **Pareto point** |
| p0.002 | 74 | 0.570 | 0.640 | 0.250 | |
| p0.006 | 32 | 0.482 | 0.555 | 0.261 | |
| p0.015 | 25 | 0.450 | 0.524 | 0.271 | |
| p0.03 | 14 | 0.225 | 0.275 | 0.289 | collapse |

## Findings (supported by the paired-by-idx analysis)

1. **A small length penalty is Pareto-neutral-to-better.** p0.0→p0.001: median
   −35 tok (−28%), mean ΔFVE **+0.008**, and on **50%** of prompts the shorter
   model reconstructs as well or better. So ~25–30% of the no-penalty AV's tokens
   are reconstruction-neutral filler. The direction-only −MSE reward carries no
   length term, so extra tokens don't hurt reward → the AV pads.

2. **Reconstruction is robust to large cuts.** p0.0→p0.002: median −52 tok
   (−41%) costs only mean **−0.014** FVE (43% of prompts as-good-or-better). Even
   p0.006 (32 tok, −75%) retains 91% of base FVE. The reconstruction-bearing
   content concentrates in ~30–75 tokens.

3. **Cliff at extreme compression.** base→p0.03: −90% tokens costs **−0.306** FVE
   and only **10%** of prompts retain quality. There is an irreducible core
   (~25–30 tokens) of reconstruction-bearing content.

4. **FVE ceiling ≈ 0.59.** Even full-length + RL caps at 0.592 → **~41% of
   normalized variance is never reconstructed**. This is the headroom where
   decodable-but-unverbalized content *could* live — but the sweep cannot show it
   is decodable (that needs probes on the activations).

5. **RL improves reconstruction:** base 0.532 → p0.0 0.585 (+5.3 pp), consistent
   with the run-guide RL gain.

6. **Confabulation is rampant and FVE-orthogonal.** In `comparison_base_vs_penalty.md`
   the explanations invent source details (a "patient review / unanswered call",
   names like "Angelos / Family Hope") for a source that is a senior-care
   employee-award notice — yet FVE stays ~0.59. FVE rewards directional encoding,
   **not** source-faithfulness. Confirms the design's insistence that Gate-3b
   faithfulness be **behavioral**, not FVE/probe.

7. **Distribution caveat (plot, don't mean).** median FVE (0.62) ≫ mean (0.53),
   std ~0.28 → a heavy low-FVE tail drags the mean; ~a quarter of prompts
   reconstruct poorly. Aggregate FVE masks heterogeneity → any read/unread work
   must be per-prompt, not on aggregate FVE.

8. **Injection path healthy:** extraction 99–100%, **cjk 0%** across all models —
   no injection failures on this checkpoint (the marker/embedding path works).

## Relevance to H1 (decodable-but-unread) — honest

The sweep measures *reconstruction vs explanation length*, **not** omission of
decodable content, so it neither supports nor refutes H1. What it does establish
as context: (a) ~41% of variance is unreconstructed = real headroom; (b) the
−MSE reward leaves the AV **underconstrained** (verbosity slack), so a different
reward (e.g. whitening) could redirect output capacity at little FVE cost — but
could equally produce different filler; (c) confabulation, not omission, is the
dominant faithfulness failure here → behavioral checks are mandatory. It is a
useful **baseline and template** for how our whitened-reward sweep would look
(same intervention axis, same eval), and it bounds the opportunity — nothing more.

## What this does NOT let us claim
- Not that the unreconstructed 41% is decodable (untested — needs probes).
- Not that length-penalty redundancy = the "unread" content of H1 (different axis:
  verbosity in the explanation vs omission from the explanation).
- Not that a whitened reward would recover anything (the sweep is a different
  mechanism — brevity pressure, not reconstruction-geometry reweighting).
