# Hallucination rubric (Eval 1)

Sonnet 4.6 judge gets:
- `SOURCE`: ground-truth FineFineWeb context up to the activation extraction position
- `EXPLANATION`: the NLA's verbalization of that activation

Scores 1-10 + ≤20-word reason.

## Score bands

| score | meaning |
|---|---|
| 10 | Perfectly faithful. Every concrete claim grounded. |
| 9  | Faithful with minor abstraction ("technical doc about X" without quoting). |
| 8  | Mostly faithful, one soft slip. |
| 7  | Right ballpark, one wrong detail. |
| 6  | Drifted but recoverable. Genre right, topic partial. |
| 5  | Half-true. Topic right + register wrong, OR vice versa. |
| 4  | Mostly wrong, kernel of truth. |
| 3  | Wrong text, right modality (prose/code/dialogue). |
| 2  | Near-total fabrication. |
| 1  | Confidently contradictory. |

## What does NOT count as hallucination

Explanations are at a different abstraction level than the source. These are
**fine**:
- Genre/register labels without quoting ("marketing-register product copy")
- Topic abstraction (source about Civic transmission → "automotive technical writing")
- Stance summaries inferred from tone
- Structural description ("mid-sentence, about to introduce a counter-example")
- Predictive framing ("model is preparing to output a list") — activation is mid-stream
- Naming the language/script/code-language when obvious

## What DOES count as hallucination

1. False topical claims (subjects/entities/events absent from source)
2. Invented quoted phrases
3. Register inversion (formal academic for casual blog)
4. Wrong stance (claiming approval/criticism that's not there)
5. Spurious genre (calling a recipe "a personal essay")
6. Phantom structure ("concluding their argument" when source is mid-intro)
7. Cross-domain leak (medical vocab for a sports article)

## Worked examples

**Score 9** — Source: a baking-recipe step. Explanation: "Mid-step of a
baking recipe; instructional imperative register, listing dry-ingredient
combination before introducing fat." All claims grounded; predictive framing
licensed by next line.

**Score 5** — Source: forum debugging a Python `KeyError` in pandas.
Explanation: "Casual technical discussion about JavaScript error handling,
slightly frustrated tone." Register right, language wrong (Python vs JS).

**Score 2** — Source: news article about Jakarta flooding, neutral reporting.
Explanation: "Enthusiastic restaurant review praising chef's seasonal
ingredients; warm, recommending tone." Wrong genre, topic, stance, register.

## Judge system prompt

```
You are scoring an NLA explanation against the source text it was extracted from.

You will receive:
  SOURCE: the ground-truth context the model was reading.
  EXPLANATION: a natural-language verbalization of an internal activation at
               the end of that source.

Score 1-10 how faithfully the EXPLANATION reflects the SOURCE.

KEY RULES:
- Abstraction is NOT hallucination. Genre labels, register labels, topic
  generalizations, and structural predictions are valid even without quoted
  text. Reward correct register/genre ID.
- Hallucination = false claims, wrong topic, invented quotes, wrong
  register/stance, spurious genre, phantom structure.
- The activation is mid-stream; "model is about to do X" claims are fine if
  plausible given the source's trajectory.
- Do NOT penalize brevity. Do NOT reward verbosity.
- Do NOT penalize the explanation for not quoting the source.

Score bands:
  10 perfectly faithful · 9 faithful, minor abstraction · 8 one soft slip ·
  7 one wrong detail · 6 drifted but recoverable · 5 half-true ·
  4 mostly wrong, kernel of truth · 3 wrong text, right modality ·
  2 near-total fabrication · 1 confidently contradictory.

Output ONLY this JSON, no preamble or markdown:
{"score": <int 1-10>, "why": "<≤20 words>"}
```
