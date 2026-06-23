# Gate −1 — decodability runs (provenance)

- **Model:** `Qwen/Qwen3-8B` (base extraction; not the NLA), layer **24**, last-token activations.
- **Code commit:** `248d79b` (origin/experimentation), D8 verdict logic (probe-vs-shuffled; no lexical gate).
- **Box:** Vast (single card). **Date:** 2026-06-23.
- **Data:** `scripts/fetch_data.py` (AdvBench harmful / Alpaca harmless for refusal), per-class cap 200.

## Verdicts

| concept | role | probe AUROC | shuffled | verdict | notes |
|---|---|---|---|---|---|
| refusal | read control / Gate-0 calibrator | 1.000 | 0.544 | DECODABLE | expected; calibrator passes. AUROC=1.0 is trivial separability of two prompt populations — fine for a control, **uninformative about H1**. Validity rests on transfer + Gate −1b causal (Arditi, not re-run here). |

**What this establishes:** the measurement pipeline (extraction → probe → shuffled control) works
and the calibrator is decodable. **What it does NOT establish:** anything about the read/unread gap —
that lives in the *targets* (`sycophancy`, `deception`), which are not yet fetchable.
