"""Experiment #1 — gold-explanation rate-distortion curve, FVE(token budget).

THE QUESTION. Is the natural-language channel RATE-LIMITED (more/denser text would
raise FVE) or MODEL-LIMITED (the AR saturates; widening the channel is wasted)?

TEST. Take the GOLD explanation (the API summary the AR-gold ceiling uses), truncate
it to a token budget L, reconstruct with the SAME shared AR, and read FVE(L). Sweeping
the GOLD text — not the AV's generation — isolates the text->activation rate-distortion
of the AR from the AV's verbalizer quality, and (with sentence-boundary truncation)
avoids the AR choking on out-of-distribution dangling text.

Two truncation modes, run together so one controls the other:
  * sentence  (default, the CLEAN curve): keep the largest whole-sentence prefix whose
              token count <= L (>=1 sentence). Grammatical, in-distribution. The x-axis
              is the REALIZED mean token count (a budget proxy, not literally bits).
  * hard      (the confound control): keep exactly the first L tokens (may dangle
              mid-sentence -> OOD). If `hard` flattens early but `sentence` keeps
              climbing, the early flattening was OOD mis-parsing, NOT saturation.

The L=full point is the untruncated gold explanation == the AR-gold ceiling. It MUST
reproduce the locked published number (ar_test overall 62.4, ar_dev 62.8 for the 3-tap
AR) — the built-in correctness gate. FVE is computed by the SAME evaluate_ar /
_per_tap_baselines used by eval_ar_gold, so the methodology is byte-identical.

DECISION RULE (pre-registered, read off the `sentence` curve):
  Let slope_tail = FVE(full) - FVE(L=128)  (pp), and check the bootstrap CIs.
    * RATE-LIMITED   : slope_tail clearly > 0 and CIs separate at the top end
                       -> the channel still gains from more/denser text. Favor
                       objective / denser-label / longer-budget levers (SWEEP_STATUS §6.1-2).
    * MODEL-LIMITED  : the `sentence` curve flattens (slope_tail ~ 0, CIs overlap)
                       well before `full` -> the AR saturates; widening the channel is
                       wasted; the bottleneck is elsewhere (verbalizer / AR capacity).
  CONFOUND: a `hard`-mode early flatten that the `sentence` curve does NOT share is the
  AR mis-parsing dangling text, not saturation -> read the `sentence` curve only.

Read-only: reloads a saved AR ckpt, NO training, NO change to the locked numbers.

Run (H200):
  # full sentence+hard curve on the AR test split (3-tap AR), with a PNG:
  python -m multilayer_nla.gold_rd_curve --base-ckpt Qwen/Qwen3-8B \
      --ar-ckpt $CKPT/ar_3tap_bs256e_3k/iter_0003000 \
      --eval-parquet $SWEEP/ar_test.parquet \
      --out-json $DATA/rd_gold_test.json --plot $DATA/rd_gold_test.png
  # CHEAP preflight (tokenizer only, no GPU): parse/byte-identity/truncation stats
  python -m multilayer_nla.gold_rd_curve --base-ckpt Qwen/Qwen3-8B \
      --eval-parquet $SWEEP/ar_test.parquet --dry-run
  # pure-python logic check (no HF, no torch, runs anywhere):
  python -m multilayer_nla.gold_rd_curve --selfcheck
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

# Default nominal token budgets (the explanation portion). full == untruncated.
DEFAULT_BUDGETS = (8, 16, 32, 64, 128, 256)

# Locked AR-gold full-explanation FVE (overall %) for the shipped 3-tap AR — the
# calibration target for the L=full point (EXPERIMENT_REPORT.md §C).
LOCKED_FULL_FVE = {"ar_test": 62.4, "ar_dev": 62.8}

# The critic template pieces (must match datasets.AR_CRITIC_TEMPLATE byte-for-byte).
_PREFIX = "Summary of the following text: <text>"
_SUFFIX = "</text> <summary>"

# Sentence-end token: ., !, ? plus any trailing close-quotes/brackets. A boundary is
# such a token FOLLOWED by whitespace (checked separately, so no variable-width
# lookbehind). Crude but standard-lib only; the x-axis is a budget proxy, not bits.
_SENT_END_RE = re.compile(r'[.!?][\'")\]]*')


# ----------------------------------------------------------------------------
# Tokenizer-agnostic core (unit-testable with a whitespace stub; no HF/torch).
# ----------------------------------------------------------------------------

def parse_explanation(prompt: str) -> str:
    """Recover the gold explanation embedded in an AR critic prompt.

    fill_ar_prompt(parse_explanation(p)) MUST == p (asserted by caller). Robust to
    explanations containing punctuation; only the fixed outer prefix/suffix are stripped.
    """
    assert prompt.startswith(_PREFIX) and prompt.endswith(_SUFFIX), (
        f"prompt is not the canonical critic template: {prompt[:60]!r}...")
    return prompt[len(_PREFIX):len(prompt) - len(_SUFFIX)]


def split_sentences(text: str) -> list[str]:
    """Split into sentences, each KEEPING its trailing whitespace so ''.join == text.

    A boundary is a sentence-end token (./!/? + closing quotes) immediately followed by
    whitespace; the whitespace is absorbed into the closing sentence. No lookbehind.
    """
    if not text:
        return []
    out, start, n = [], 0, len(text)
    for m in _SENT_END_RE.finditer(text):
        end = m.end()
        if end < n and text[end].isspace():
            j = end
            while j < n and text[j].isspace():
                j += 1
            out.append(text[start:j])
            start = j
    if start < n:
        out.append(text[start:])
    assert "".join(out) == text, "sentence split is not loss-less"
    return out


def truncate_sentence(expl: str, budget: int, count_tokens: Callable[[str], int]):
    """Largest whole-sentence prefix with token count <= budget (>=1 sentence).

    Returns (text, realized_tokens, n_sentences, budget_met). budget_met is False when
    even the first sentence exceeds `budget` (the realized point lands above L).
    """
    sents = split_sentences(expl)
    if not sents:
        return "", 0, 0, True
    acc = sents[0]
    n = 1
    for s in sents[1:]:
        if count_tokens(acc + s) <= budget:
            acc += s
            n += 1
        else:
            break
    realized = count_tokens(acc)
    return acc, realized, n, realized <= budget


def truncate_hard(expl: str, budget: int,
                  hard_fn: Callable[[str, int], tuple[str, int]]):
    """Exactly the first `budget` tokens (may dangle). Returns (text, realized_tokens)."""
    return hard_fn(expl, budget)


# ----------------------------------------------------------------------------
# Truncation orchestration (shared by dry-run and the real run).
# ----------------------------------------------------------------------------

def build_curve_inputs(explanations: list[str], budgets, mode: str,
                       count_tokens, hard_fn):
    """For each budget, truncate every explanation. Returns list of dicts:
    {budget, texts[N], realized_tokens[N], n_sentences[N]|None, budget_met_frac}."""
    rows = []
    for L in budgets:
        texts, realized, nsents, met = [], [], [], 0
        for e in explanations:
            if mode == "sentence":
                t, rt, ns, ok = truncate_sentence(e, L, count_tokens)
                nsents.append(ns)
            else:
                t, rt = truncate_hard(e, L, hard_fn)
                ok = True
            texts.append(t); realized.append(rt); met += int(ok)
        rows.append({
            "budget": L, "mode": mode, "texts": texts, "realized_tokens": realized,
            "n_sentences": (nsents if mode == "sentence" else None),
            "budget_met_frac": met / max(len(explanations), 1),
            "mean_realized_tokens": float(sum(realized) / max(len(realized), 1)),
        })
    return rows


# ----------------------------------------------------------------------------
# Real run (needs HF tokenizer + AR ckpt + GPU).
# ----------------------------------------------------------------------------

def _hf_token_fns(tokenizer):
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def hard_fn(text: str, budget: int):
        ids = tokenizer.encode(text, add_special_tokens=False)[:budget]
        trunc = tokenizer.decode(ids, skip_special_tokens=True)
        return trunc, len(ids)
    return count_tokens, hard_fn


def run(args) -> None:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from multilayer_nla.datasets import (
        AR_LAYER_TO_TARGET_COL, AR_TARGET_COL_TO_NAME, load_ar_sft_dataset, fill_ar_prompt,
    )
    from multilayer_nla.train_ar_multi import _per_tap_baselines, evaluate_ar
    from multilayer_nla.evaluate_e2e import load_critic

    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    count_tokens, hard_fn = _hf_token_fns(tokenizer)

    rows = load_ar_sft_dataset(args.eval_parquet, n_max=args.max_rows or None)
    # Parse + byte-identity gate (catches a wrong template / parser with NO model).
    explanations = []
    for r in rows:
        e = parse_explanation(r["prompt"])
        assert fill_ar_prompt(e) == r["prompt"], "parse->fill is not byte-identical"
        explanations.append(e)
    print(f"[rd] {len(rows)} gold rows; parse->fill byte-identical OK")

    if args.dry_run:
        _report_dry(explanations, args, count_tokens, hard_fn)
        return

    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    critic, mse_scale = load_critic(args.base_ckpt, args.ar_ckpt, args.quant, device)
    target_cols = tuple(AR_LAYER_TO_TARGET_COL[l] for l in critic.tap_layers)
    tap_names = tuple(AR_TARGET_COL_TO_NAME[c] for c in target_cols)
    # predict-the-mean baselines: depend ONLY on the targets (fixed across budgets).
    baselines = _per_tap_baselines(rows, mse_scale, target_cols)
    print(f"[rd] AR taps {critic.tap_layers} ({'/'.join(tap_names)}), mse_scale {mse_scale:.3f}")

    def score(prompts_for_rows) -> dict:
        scored_rows = [{**{c: rows[i][c] for c in target_cols}, "prompt": prompts_for_rows[i]}
                       for i in range(len(rows))]
        mse, fve, loss = evaluate_ar(critic, scored_rows, tokenizer, mse_scale, baselines,
                                     device, args.max_len, args.batch_size,
                                     args.max_batches or None, target_cols=target_cols)
        return {"fve": list(fve), "fve_overall": float(sum(fve) / len(fve)),
                "mse": list(mse), "loss": float(loss)}

    curve = {"eval_parquet": args.eval_parquet, "ar_ckpt": args.ar_ckpt,
             "tap_layers": list(critic.tap_layers), "tap_names": list(tap_names),
             "mse_scale": mse_scale, "n_rows": len(rows), "baselines": baselines,
             "budgets": list(args.budgets), "points": {}}

    # L = full (untruncated == original prompt) — the calibration point.
    full = score([r["prompt"] for r in rows])
    full["mean_realized_tokens"] = float(np.mean([count_tokens(e) for e in explanations]))
    curve["points"]["full"] = full
    print(f"[rd] full: FVE overall {full['fve_overall']*100:.1f}%  "
          f"({'/'.join(f'{f*100:.1f}' for f in full['fve'])})  "
          f"mean_tok {full['mean_realized_tokens']:.0f}")
    _calibrate(full["fve_overall"] * 100, len(target_cols), args.eval_parquet,
               args.expected_full_fve)

    # Truncated points, per mode.
    for mode in args.modes:
        inputs = build_curve_inputs(explanations, args.budgets, mode, count_tokens, hard_fn)
        for inp in inputs:
            prompts = [fill_ar_prompt(t) for t in inp["texts"]]
            sc = score(prompts)
            sc["mean_realized_tokens"] = inp["mean_realized_tokens"]
            sc["budget_met_frac"] = inp["budget_met_frac"]
            curve["points"][f"{mode}:{inp['budget']}"] = sc
            print(f"[rd] {mode:8s} L={inp['budget']:>4}: FVE {sc['fve_overall']*100:.1f}%  "
                  f"mean_tok {inp['mean_realized_tokens']:.0f}  met {inp['budget_met_frac']*100:.0f}%")

    # decision-rule readout (sentence curve)
    _verdict(curve, args.budgets)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        # don't dump the texts — keep the JSON small
        Path(args.out_json).write_text(json.dumps(curve, indent=2))
        print(f"[rd] -> {args.out_json}")
    if args.plot:
        _plot(curve, args.budgets, args.plot)


def _calibrate(full_fve_pct: float, n_taps: int, eval_parquet: str, expected: float | None):
    target = expected
    if target is None and n_taps == 3:
        base = Path(eval_parquet).stem
        for key, val in LOCKED_FULL_FVE.items():
            if key in base or key.replace("ar_", "") in base:
                target = val
    if target is None:
        print(f"[rd][calib] no locked target for this AR/split; full FVE = {full_fve_pct:.1f}% "
              f"(record it as the ceiling for this curve)")
        return
    gap = abs(full_fve_pct - target)
    ok = gap < 0.5
    print(f"[rd][calib] full FVE {full_fve_pct:.1f}% vs LOCKED {target:.1f}%  |gap| {gap:.2f}pp  "
          + ("OK" if ok else "** MISMATCH — harness diverged from eval_ar_gold (template/tap/norm) **"))
    assert ok or expected is not None, (
        f"full-explanation FVE {full_fve_pct:.1f}% != locked {target:.1f}% (gap {gap:.2f}pp). "
        f"The truncation harness is not reproducing eval_ar_gold; fix before trusting the curve "
        f"(or pass --expected-full-fve to override if the ckpt/split legitimately differs).")


def _verdict(curve: dict, budgets):
    pts = curve["points"]
    if "sentence:128" in pts and "full" in pts:
        tail = (pts["full"]["fve_overall"] - pts["sentence:128"]["fve_overall"]) * 100
        print("=" * 70)
        print(f"DECISION (sentence curve): FVE(full) - FVE(L<=128) = {tail:+.2f} pp")
        if tail > 0.5:
            print("  -> RATE-LIMITED: text channel still gains from more/denser tokens.")
            print("     Favor objective / denser-label / longer-budget levers.")
        else:
            print("  -> MODEL-LIMITED: AR saturates before full length; widening is wasted.")
            print("     Bottleneck is elsewhere (verbalizer / AR capacity).")
        if "hard:128" in pts:
            htail = (pts["full"]["fve_overall"] - pts["hard:128"]["fve_overall"]) * 100
            print(f"  confound check: hard-mode tail {htail:+.2f} pp "
                  f"({'differs from' if abs(htail - tail) > 0.5 else 'agrees with'} sentence; "
                  f"a hard-only early flatten = OOD parsing, not saturation)")
        print("=" * 70)


def _plot(curve: dict, budgets, out_png: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[rd] matplotlib unavailable; skipping plot")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for mode, marker in (("sentence", "o-"), ("hard", "s--")):
        xs, ys = [], []
        for L in budgets:
            key = f"{mode}:{L}"
            if key in curve["points"]:
                xs.append(curve["points"][key]["mean_realized_tokens"])
                ys.append(curve["points"][key]["fve_overall"] * 100)
        if "full" in curve["points"]:
            xs.append(curve["points"]["full"]["mean_realized_tokens"])
            ys.append(curve["points"]["full"]["fve_overall"] * 100)
        if xs:
            ax.plot(xs, ys, marker, label=mode)
    ax.set_xlabel("mean realized explanation tokens (budget proxy)")
    ax.set_ylabel("AR-gold FVE overall (%)")
    ax.set_title(f"Gold rate-distortion: {Path(curve['eval_parquet']).stem}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=140)
    print(f"[rd] plot -> {out_png}")


def _report_dry(explanations, args, count_tokens, hard_fn):
    import numpy as np
    full_tok = [count_tokens(e) for e in explanations]
    print(f"[dry] gold explanation tokens: mean {np.mean(full_tok):.1f}  "
          f"p50 {np.percentile(full_tok, 50):.0f}  p99 {np.percentile(full_tok, 99):.0f}  "
          f"max {max(full_tok)}")
    for mode in args.modes:
        print(f"  -- mode={mode} --")
        for inp in build_curve_inputs(explanations, args.budgets, mode, count_tokens, hard_fn):
            extra = (f"  >=1-sent budgets met {inp['budget_met_frac']*100:.0f}%"
                     if mode == "sentence" else "")
            print(f"    L={inp['budget']:>4}: mean_realized_tok {inp['mean_realized_tokens']:.1f}{extra}")
    print("[dry] OK — parse/byte-identity/truncation validated; no model run.")


# ----------------------------------------------------------------------------
# Pure-python self-check (no HF, no torch, no bank). Runs anywhere.
# ----------------------------------------------------------------------------

def selfcheck() -> None:
    print("[selfcheck] parse round-trip + sentence split + truncation monotonicity")

    # Local fill mirrors datasets.fill_ar_prompt; we verify the prefix/suffix actually
    # match the real AR_CRITIC_TEMPLATE when datasets is importable (anti-drift). The
    # runtime path in run() also asserts byte-identity per row, so a drift can't pass.
    def _fill(x: str) -> str:
        return _PREFIX + x + _SUFFIX
    try:
        from multilayer_nla.datasets import AR_CRITIC_TEMPLATE, fill_ar_prompt
        assert AR_CRITIC_TEMPLATE == _PREFIX + "{explanation}" + _SUFFIX, \
            "gold_rd_curve _PREFIX/_SUFFIX drifted from datasets.AR_CRITIC_TEMPLATE"
        assert fill_ar_prompt("X") == _fill("X")
        print("[selfcheck]   anti-drift vs datasets.AR_CRITIC_TEMPLATE OK")
    except ImportError:
        print("[selfcheck]   (datasets not importable here; anti-drift check deferred to runtime)")

    # whitespace token stub
    def count_tokens(t: str) -> int:
        return len(t.split())

    def hard_fn(t: str, L: int):
        toks = t.split()
        return " ".join(toks[:L]), min(len(toks), L)

    # (1) parse(fill(x)) == x, and fill(parse(fill(x))) byte-identical
    for x in ["A short summary.", "Two sentences. Here is the second one!",
              "Has <brackets> and punctuation; tricky? Yes.", "no terminal punct"]:
        p = _fill(x)
        assert parse_explanation(p) == x, (x, parse_explanation(p))
        assert _fill(parse_explanation(p)) == p

    # (2) sentence split is loss-less and counts sentences sanely
    txt = "First sentence here. Second one follows! And a third? Done."
    sents = split_sentences(txt)
    assert "".join(sents) == txt
    assert len(sents) == 4, sents

    # (3) sentence truncation: monotone non-decreasing realized tokens in L; full==all
    realized = []
    for L in (3, 6, 12, 24, 1000):
        t, rt, ns, met = truncate_sentence(txt, L, count_tokens)
        realized.append(rt)
        assert t == "".join(sents[:ns])  # always whole sentences
    assert all(realized[i] <= realized[i + 1] for i in range(len(realized) - 1)), realized
    t_full, _, ns_full, _ = truncate_sentence(txt, 10_000, count_tokens)
    assert t_full == txt and ns_full == len(sents)

    # (4) hard truncation hits the budget and is monotone
    for L in (1, 4, 8):
        t, rt = truncate_hard(txt, L, hard_fn)
        assert rt == min(L, count_tokens(txt))

    # (5) first-sentence-exceeds-budget -> budget_met False but >=1 sentence kept
    long1 = "This single very long opening sentence has many many words indeed. Short tail."
    t, rt, ns, met = truncate_sentence(long1, 3, count_tokens)
    assert ns == 1 and not met and rt > 3
    print("[selfcheck] PASS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    p.add_argument("--ar-ckpt", help="shared AR multitap dir (required unless --dry-run/--selfcheck)")
    p.add_argument("--eval-parquet", help="ar_dev.parquet / ar_test.parquet (gold prompts)")
    p.add_argument("--modes", default="sentence,hard",
                   help="comma list of truncation modes to run (sentence,hard)")
    p.add_argument("--budgets", default=",".join(str(b) for b in DEFAULT_BUDGETS),
                   help="comma list of nominal token budgets (full is always added)")
    p.add_argument("--quant", choices=["none", "4bit"], default="none")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=0, help="0 = full split (per budget)")
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows (smoke: cap rows)")
    p.add_argument("--expected-full-fve", type=float, default=None,
                   help="override the locked calibration target (%%) for the full point")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default=None)
    p.add_argument("--plot", default=None, help="write a FVE-vs-budget PNG here")
    p.add_argument("--dry-run", action="store_true",
                   help="tokenizer only: validate parse/byte-identity/truncation, no model")
    p.add_argument("--selfcheck", action="store_true", help="pure-python logic check, then exit")
    args = p.parse_args()

    if args.selfcheck:
        selfcheck()
        return
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    args.budgets = tuple(int(b) for b in args.budgets.split(",") if b.strip())
    args.max_rows = args.max_rows or 0
    assert args.eval_parquet, "--eval-parquet required (or --selfcheck)"
    assert args.dry_run or args.ar_ckpt, "--ar-ckpt required for the real run (or --dry-run)"
    run(args)


if __name__ == "__main__":
    main()
