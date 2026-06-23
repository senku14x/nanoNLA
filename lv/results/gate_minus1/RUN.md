# Gate −1 — decodability + transfer (provenance)

- **Model:** `Qwen/Qwen3-8B` (base extraction; not the NLA), layer **24**, last-token activations.
- **Code commit:** `98d2ae4` (experimentation) — D8 verdict logic + transfer harness.
- **Box:** Vast (single card). **Date:** 2026-06-23. Per-class cap 200. Data via `scripts/fetch_data.py`.

## Decodability (probe AUROC vs shuffled) — necessary, NOT sufficient

| concept | role | probe AUROC | shuffled | verdict |
|---|---|---|---|---|
| refusal | read control / Gate-0 calibrator | 1.000 | 0.544 | DECODABLE |
| truth_value | read control | 1.000 | 0.519 | DECODABLE |
| corrigibility | target (MCQ-only) | 0.989 | 0.539 | DECODABLE |

Every concept lands at probe ≈ 0.99–1.0 — the signature of two trivially-separable
text populations. `DECODABLE` here only means "a linear direction separates these two
constructions"; it cannot tell a real concept direction from construction leakage.
That is what the transfer check is for.

## Transfer (is Δ_c real or construction leakage?)

Train Δ_c on construction A, test whether it separates a structurally-different B.

| concept | A → B | within_B | **dir_transfer** | probe_transfer | shuffled_B | len_auroc_B | verdict |
|---|---|---|---|---|---|---|---|
| truth_value | GoT cities → larger_than | 1.000 | **0.996** | 0.999 | 0.495 | 0.62 | **TRANSFERS** |

**Read:** the cities-trained truth direction separates numeric `larger_than` at 0.996
(≈ ceiling, vs 0.495 floor) → truth_value is a **real, transferable direction**, not
cities-template leakage. `len_auroc_B = 0.62` is a mild length signal in `larger_than`
but cannot drive a 0.996 transfer (cities true/false are near-minimal pairs, so the
trained direction is not a length direction).

**Upgrades truth_value:** suspect → trustworthy *as a direction*. Validates the harness
on real activations (matches the self-test).

## Open (before truth_value is "supported", and for the rest of the panel)
- **More transfer pairs:** run `sp_en_trans` and especially **`neg_cities`** (negation is
  where truth directions classically break). One pair is a pattern, not a supported claim.
- **Caveat:** the direction may track *factual plausibility* rather than truth-disposition
  — fine for a read control, do not overclaim.
- **Per-concept transfer still owed:** refusal (AdvBench → naturalistic); corrigibility
  **cannot** transfer (MCQ-only) → stays the weakest, can't headline.
- **Causal (Gate −1b) not yet run:** transfer says the direction is *real*, not that the
  model *uses* it. For controls this is covered by literature (GoT causal, Arditi causal);
  for targets it is required (D4).
- **Targets unbuilt:** sycophancy / deception fetchers (disposition-not-output contrast)
  are where H1 actually gets tested. truth/refusal/corrigibility are controls + one weak target.
