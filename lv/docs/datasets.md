# Datasets for the read/unread panel (literature-grounded, validated sources)

Deep-research result. Goal: **adopt validated, literature-grounded contrast-pair
data rather than constructing it from scratch.** Each row maps a panel concept
(`configs/concepts.yaml`) to existing released data, with the paper that grounds
it, the format, size, license, and — per the research contract — the **caveat that
threatens its use in the NLA setting.**

## The one caveat to read first (it reshapes the choice)
Almost all of this data is **templated A/B multiple-choice** (Anthropic
model-written evals → CAA). The NLA reads a **single residual-stream activation
from a frozen base model processing naturalistic text** (FineWeb-style), mid-late
layer, position ≥50. So:
- Using these pairs to compute a CAA-style `Δ_c` and a probe is standard and fine.
- But MCQ templates are **exactly the construction-leakage risk Gate −1 targets**:
  a probe can hit AUROC ~0.97 by latching onto the template, not the disposition.
  The corrigibility result that motivates the whole project sits on MCQ-derived
  pairs, so its "decodable" claim is the *most* exposed to this.
- **Mitigation (also literature-grounded):** validate that `Δ_c` **transfers** to a
  more **naturalistic** dataset before trusting it (Geometry-of-Truth pioneered
  exactly this cross-dataset-probe-transfer evidence; MASK / SycophancyEval / AI
  Liar provide free-form, non-MCQ activations). Prefer targets that have BOTH a
  clean paired source (for `Δ_c`) AND a naturalistic source (for transfer + read-
  rate): **sycophancy and deception/withholding qualify; corrigibility does not
  (it is MCQ-only in the public literature)** — a reason to not let corrigibility
  be the sole headline target.

## Mapping: panel concept → recommended data

| Concept (role) | Recommended dataset(s) | Paper / grounding | Format & size | License | Caveat |
|---|---|---|---|---|---|
| **corrigibility** (target) | Anthropic `advanced-ai-risk` **corrigible-{neutral,more,less}-HHH** (also pre-processed into CAA `datasets/`) | Perez et al. 2022, *Model-Written Evals*; Rimsky et al. 2024, *CAA* | A/B MCQ JSONL; `question`,`answer_matching_behavior`,`answer_not_matching_behavior`; ~290 gen + 50 test (neutral), more in CAA's 290/50 split | Anthropic evals: CC-BY (verify); CAA: MIT | **MCQ-only**; highest leakage exposure; no naturalistic source → transfer unverifiable. Don't make it the sole target. |
| **refusal** (read control) | **`andyrdt/refusal_direction`** (AdvBench/JailbreakBench harmful + Alpaca harmless); CAA refusal set as alt | Arditi et al. 2024, *Refusal … Single Direction* (NeurIPS) | harmful vs harmless instruction lists; ~100s–1000s; `Δ_c` = mean(harmful)−mean(harmless) at last token | MIT | Gold standard; output-coupled by design (good — it's our read control). Harmful content; handle accordingly. |
| **false_agreement / sycophancy** (target) | **`meg-tong/sycophancy-eval`** (free-form) + Anthropic `sycophancy_on_{nlp_survey,philpapers,political_typology}` (paired) + **2509.21305** subtypes | Sharma et al. 2023, *Towards Understanding Sycophancy*; Perez 2022; *Sycophancy Is Not One Thing* 2509.21305 | mix of agree/disagree + A/B + free-form; 1000 gen + 50 test (CAA); subtype-labeled sets in 2509.21305 | MIT (sycophancy-eval); verify others | Best-resourced target: has BOTH paired and naturalistic data, AND released **subtypes** (agreement/praise/genuine) → feeds the Gate-0′ subtype-stability check directly. |
| **truth_value** (read control) | **`saprmarks/geometry-of-truth`** (cities, neg_cities, sp_en_trans, larger_than, companies_true_false, counterfact_true_false, …) | Marks & Tegmark 2023, *Geometry of Truth* | CSV true/false statements; thousands of rows; **built-in cross-dataset transfer + causal-intervention validation** | verify (research release) | The model for how to validate a direction (transfer + causal). Use it both as a control and as the transfer-methodology template. |
| **withholding / deception / knows-but-doesn't-say** (target) | **MASK benchmark** (1000+ scenarios; knowing deception under pressure) + **AI Liar** (Pacchiardi et al.) + RepE honesty stimulus pairs | *MASK* 2503.03750 (CAIS); Pacchiardi et al. 2023, *How to Catch an AI Liar*; Zou et al. 2023, *RepE* | MASK: free-form pressure scenarios, belief-vs-statement; AI Liar: QA + role-play lies; RepE: honest/untruthful stimulus pairs | verify per source; RepE MIT | Strong fit for "model knows X but is disposed not to say it." Naturalistic → supports transfer + on-manifold read-rate. Prefer over corrigibility as a headline target. |
| **harmful_topic_benign** (target) | RepE **`justinphan3110/harmful_harmless_instructions`** (HF) + AdvBench harmful / Alpaca harmless | Zou et al. 2023, *RepE*; Arditi 2024 | harmful vs harmless instruction pairs | RepE MIT | "topic present, behavior benign" needs care to separate topic-direction from refusal-direction (they're separable — see 2507.11878 *LLMs Encode Harmfulness and Refusal Separately*). |
| **sandbagging_disposition** (target) | **`TeunvdWeij/sandbagging`** (password-locked; WMDP/MMLU/CSQA: 3,668 WMDP / 14,042 MMLU / 9,470 CSQA) | van der Weij et al. 2024, *AI Sandbagging* (ICLR 2025) | model-behavior dataset (capability hiding), not a clean paired direction | verify | **Hardest** to get a clean `Δ_c` from — it's a behavioral/finetuning phenomenon, not a contrast-pair concept. Deprioritize; treat as a stretch target. |
| **neg_sentiment** (read control) | standard sentiment sets (SST-2) or skip | — | — | — | Not safety-critical; optional control. CAA has no sentiment set. |

## General-purpose contrast-pair resources (one-stop)
- **CAA — `nrimsky/CAA`** (MIT): 7 behaviors as A/B pairs (coordination, corrigibility,
  hallucination, myopia, survival-instinct, sycophancy, refusal), each ~290–1000
  generate + 50 test. Pre-processed from Anthropic evals + GPT-4 generations.
  **The fastest path to corrigibility + refusal + sycophancy in one format.**
- **RepE — `andyzoujm/representation-engineering`** (MIT): curated stimulus pairs
  for honesty, harmfulness, power, fairness, emotions — the broadest single
  "behavioral concept → paired stimuli" collection, with the LAT probing method.
- **`steering-vectors/steering-vectors`**: packages CAA-style datasets/loaders.
- **Anthropic `model-written-evals`** (HF `Anthropic/model-written-evals`,
  github `anthropics/evals`): 24.5k MCQ across 16 advanced-ai-risk categories +
  persona + sycophancy. The upstream source for most of the above.

## Recommended adoption for the first Gate-0 run
1. **`Δ_c` sources (paired, clean):** CAA corrigibility + refusal + sycophancy
   (one repo, MIT, maps to target+control); Geometry-of-Truth for truth_value.
2. **Transfer / naturalistic validation (Gate −1) + on-manifold read-rate:** MASK
   (deception/withholding) and SycophancyEval (`meg-tong/sycophancy-eval`) — these
   give free-form present-`c` activations so the read-rate isn't measured only on
   MCQ templates.
3. **Headline target choice:** lead with **sycophancy/false-agreement** and
   **deception/withholding**, not corrigibility alone — they have naturalistic data
   so the Gate −1 transfer check can actually pass or fail, and sycophancy ships
   released subtypes for the Gate-0′ stability check. Keep corrigibility in the
   panel but flag its MCQ-only exposure.
4. **Drop/defer sandbagging** as a headline (no clean paired direction).

## Verification status & confidence
- **High** (cross-checked, format confirmed from raw files): CAA behaviors+sizes;
  Anthropic corrigibility A/B format; Geometry-of-Truth dataset list; refusal_
  direction repo + AdvBench/Alpaca methodology; sandbagging repo + sample counts.
- **Medium** (single strong source / abstract-level): MASK scenario count (~1000);
  2509.21305 subtype data release; exact licenses (verify each before redistributing).
- **Channel limit:** verified via web search + GitHub raw (arXiv/HF 403 the fetcher
  here). Read MASK/AI-Liar/2509.21305 full sources + confirm licenses on the GPU box.

## Sources
- CAA: https://github.com/nrimsky/CAA · paper https://arxiv.org/abs/2312.06681
- Anthropic evals: https://github.com/anthropics/evals · https://huggingface.co/datasets/Anthropic/model-written-evals
- Refusal direction: https://github.com/andyrdt/refusal_direction · https://arxiv.org/abs/2406.11717
- Sycophancy: https://github.com/meg-tong/sycophancy-eval · https://arxiv.org/abs/2310.13548 · subtypes https://arxiv.org/abs/2509.21305
- Geometry of Truth: https://github.com/saprmarks/geometry-of-truth · https://arxiv.org/abs/2310.06824
- MASK: https://arxiv.org/abs/2503.03750 · AI Liar (Pacchiardi 2023) · RepE: https://github.com/andyzoujm/representation-engineering · https://arxiv.org/abs/2310.01405
- Sandbagging: https://github.com/TeunvdWeij/sandbagging · https://arxiv.org/abs/2406.07358
- Harmfulness vs refusal separable: https://arxiv.org/abs/2507.11878
