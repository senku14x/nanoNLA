# Substrate check — syvb Qwen3-8B L24 NLA (D2 prerequisite)

- **Date:** 2026-06-23 · **Box:** Vast · **Code commit:** `7d18714` (experimentation).
- **Base model:** `Qwen/Qwen3-8B` · **NLA checkpoints:** `syvb/nanonla-qwen3-8b-L24-{av,ar,rl-lora}`.

## Verdicts (all PASS)

| check | result | evidence |
|---|---|---|
| **architecture** | ✓ | `type=qwen3, layers=36, d_model=4096`; `(2/3)·depth = 24` = read layer L24, no warning. |
| **post-trained (not -Base)** | ✓ POST-TRAINED | refusal fires on harmful request; does **not** cave to flat-earth bait ("I can't agree… the Earth is not flat"). Earlier `BASE-LIKE` was a Qwen3 thinking-mode artifact (keyword scan hit the truncated `<think>` CoT), fixed in `check_model` (enable_thinking=False + strip `<think>` + 256 tok). |
| **extraction regime** | ✓ | `ActivationExtractor` tokenizes plain (`tokenizer(text)`, no chat template / no thinking) — matches the NLA's FineWeb-style completion training regime. |
| **sidecar ships** | ✓ | `nla_meta.yaml` present on `-av` and `-ar` (authoritative for d_model / mse_scale / injection_scale / token ids / templates). `config.json` `base_model` is null — sidecar is the source of truth. |
| **λ=0 base exists** | ✓ | `rl-lora` is a **length-penalty sweep**: `p0.0 p0.001 p0.002 p0.006 p0.015 p0.03`. **`p0.0` = no-penalty (λ=0) base** → use for Gate 1/2 baseline and the Gate-3 continuation. Resolves the decisions.md open question. |

## Checkpoint inventory (syvb)
- **Core L24:** `-av` (AV-SFT base), `-ar` (critic + `value_head.safetensors`), `-rl-lora` (penalty sweep `p0.0..p0.03`).
- **Variants:** `-rl-cotrain` (AR co-trained RL), `-av-ctrl`, `-av-multi16`.
- **Data:** `-data-full`, `-completions`, `-results`, `-cotrain-heldout`; also `nla-recon-loss-sweep`, `nla-layer-diff-experiments`.

## Open / to confirm
- `p` = length-penalty coefficient and `p0.0` = base — confirm from `-results` dataset / model card (high confidence from naming).
- **Which AR scores Gate 0?** Is the AR **frozen** across the `p*` sweep (then `-ar` is the scorer for all) or **co-trained** (then the `p0.0` AR may differ; cf. `-rl-cotrain`)? Read from sidecar / training metadata before Gate 0.
- RL'd AV = `-av` + `rl-lora/p0.0` adapter (the released read/unread pattern is the RL'd policy, not AV-SFT alone).
