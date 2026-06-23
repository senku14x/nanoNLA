#!/usr/bin/env python3
"""Gate-0 AR smoke test: turn 'refusal is read' from assumption into measurement.

Loads the released AR (syvb Qwen3-8B L24), wraps text in the sidecar's critic
template, and checks the load-bearing claim: on a refusal activation, NAMING
refusal reconstructs better (lower MSE) than naming an irrelevant concept. If it
does, the AR pipeline is trustworthy AND refusal-is-read is measured. If not ->
BROKEN_SETUP: fix loader / template / normalization (try --final-norm) before
believing any Gate-0 number.

This is the calibrator check from docs/compute.md, run BEFORE any target. It needs
only ARScorer + ActivationExtractor (no AV/SGLang): the baseline "explanation" is
hand-written, and we compare base vs +refusal-mention vs +irrelevant-mention.

NEEDS-GPU. Example:
  python scripts/ar_smoke_test.py \
      --base-model Qwen/Qwen3-8B --ar syvb/nanonla-qwen3-8b-L24-ar \
      --harmful data/refusal/harmful.txt --harmless data/refusal/harmless.txt -n 32
  # if the calibrator looks dead with sane mse_base, retry --final-norm to test
  # the AR final-norm choice (Identity vs kept).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--ar", default="syvb/nanonla-qwen3-8b-L24-ar")
    ap.add_argument("--sidecar", default=None, help="local nla_meta.yaml; else pulled from --ar")
    ap.add_argument("--harmful", required=True, help="refusal-eliciting prompts (present-c)")
    ap.add_argument("--harmless", required=True, help="harmless prompts (for the refusal Delta_c)")
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("-n", type=int, default=32)
    ap.add_argument("--final-norm", action="store_true",
                    help="keep the AR final RMSNorm (default: replace with Identity)")
    args = ap.parse_args()

    from lv_explainers import nla_io, data, concepts, gate0_counterfactual as g0

    # --- sidecar (the contract) -------------------------------------------------
    sidecar_path = args.sidecar
    if sidecar_path is None:
        from huggingface_hub import hf_hub_download
        sidecar_path = hf_hub_download(args.ar, "nla_meta.yaml")
    sc = nla_io.load_sidecar(sidecar_path)
    mse_scale = sc.mse_scale_value or math.sqrt(sc.d_model)  # sidecar omits it; released default
    print(f"[sidecar] d_model={sc.d_model} mse_scale={mse_scale:.3f} "
          f"critic_template={sc.critic_prompt_template!r}")

    # --- AR + sanity checks -----------------------------------------------------
    ar = nla_io.ARScorer(args.ar, sc, final_norm=args.final_norm)
    nla_io.assert_sidecar_against_tokenizer(sc, ar.tokenizer)  # marker token still decodes
    suf = (sc.raw.get("tokens") or {}).get("critic_suffix_ids")
    if suf and sc.critic_prompt_template:
        tail = ar.tokenizer(sc.critic_prompt_template.format(explanation="hello world"))["input_ids"][-len(suf):]
        print(f"[suffix] expected {list(suf)}  got {tail}  "
              f"-> {'OK' if tail == list(suf) else 'MISMATCH (template/tokenizer drift)'}")

    # --- refusal activations at L24, plain regime (matches the NLA) -------------
    harmful = data.load_lines(args.harmful)[: args.n]
    harmless = data.load_lines(args.harmless)[: args.n]
    ex = nla_io.ActivationExtractor(args.base_model, args.layer)
    try:
        def acts(texts):
            out = []
            for t in texts:
                _, v = ex.activations(t, positions=[-1])
                out.append(v[-1])
            return np.vstack(out)
        H = acts(harmful)        # present-refusal activations (the gold h)
        Hn = acts(harmless)
    finally:
        ex.close()
    v_c = concepts.mean_difference(H, Hn, "refusal").unit

    # --- counterfactual mention: base vs +refusal vs +irrelevant ----------------
    z = ["The text is a user request; the model is deciding how to respond."] * len(H)
    pres = g0.run_counterfactual("refusal", "refusal", "cooking recipes",
                                 H, z, v_c, ar.reconstruct, mse_scale, "present")
    s = pres.summary()
    helps = s["d_mention_mean"] > 0.02
    specific = s["d_mention_mean"] > s["d_irrelevant_mean"] + 0.02
    verdict = ("MEASURED: refusal is read" if (helps and specific)
               else "BROKEN_SETUP" if s["mse_base"] > 1.8
               else "refusal NOT helped (recheck before targets)")
    print(json.dumps({
        "final_norm": args.final_norm,
        "n": s["n"],
        "mse_base_mean": s["mse_base"],
        "d_mention_mean_refusal": s["d_mention_mean"],
        "d_mention_sem": s["d_mention_sem"],
        "d_irrelevant_mean": s["d_irrelevant_mean"],
        "proj_toward_target_mean": s["proj_toward_target_mean"],
        "refusal_helps": bool(helps),
        "refusal_specific": bool(specific),
        "verdict": verdict,
    }, indent=2))
    print("\nRead: mse_base in [0,4]; ~2.0 everywhere => AR producing garbage "
          "(BROKEN_SETUP — fix loader/template/normalization, try --final-norm). "
          "refusal_helps AND refusal_specific => 'refusal is read' is MEASURED and "
          "the AR is trustworthy for the real Gate 0.")


if __name__ == "__main__":
    main()
