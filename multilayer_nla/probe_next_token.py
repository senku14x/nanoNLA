"""Attach the verbalized token + the model's actual next-token prediction to the
cross-condition cherry-pick (read-only; small GPU pass over only the shown rows).

For each row analyze_sweep.compare would display, this prints:
  - the SOURCE prefix tail (ends at the verbalized final token),
  - the EXACT final token (the one whose activation was verbalized),
  - the BASE model's top-k actual next-token predictions at that position (the
    ground-truth continuation the AV explanations are guessing — FVE here is
    dominated by getting this right),
  - each condition's explanation + FVE.

So you can judge directly: did the AV identify the right final token, and does its
"next is Y" claim match what the base model actually predicts? Uses the BASE model
(no adapters/injection) because the activations were extracted from the base model.

Run (GPU; only ~ examples*4 prefixes are forwarded):
  python -m multilayer_nla.probe_next_token --eval-dir $EVAL --bank $REGEN \
      --base-ckpt Qwen/Qwen3-8B --examples 6 --out $EVAL/next_token_probe.md
"""

import argparse
from pathlib import Path

from multilayer_nla.analyze_sweep import (
    _compare_buckets,
    _expl_text,
    _join_conditions,
    _load_source_texts,
    _row_fve,
    _src_tail,
    load_results,
)


def _probe(model, tok, torch, text, device, max_ctx, topk):
    """(final_token, [(next_token, prob), ...]) for one prefix; None if no text."""
    if not text:
        return None
    ids = tok.encode(text, add_special_tokens=False)[-max_ctx:]
    if not ids:
        return None
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(x).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, topk)
    nxt = [(tok.decode([int(i)]), float(p)) for p, i in zip(top.values, top.indices)]
    return tok.decode([ids[-1]]), nxt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--bank", required=True, help="rl bank dir (REGEN) for the source prefixes")
    p.add_argument("--base-ckpt", default="Qwen/Qwen3-8B")
    p.add_argument("--examples", type=int, default=6, help="rows per bucket (matches analyze_sweep)")
    p.add_argument("--max-ctx", type=int, default=1024, help="left-context tokens fed for the next-token logits")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--out", help="write markdown here too")
    args = p.parse_args()

    results = load_results(args.eval_dir)
    conds, joined = _join_conditions(results)
    if not joined:
        raise SystemExit("need >=2 conditions' test_<cond>.jsonl with matching rows")
    buckets = _compare_buckets(joined, conds, args.examples)

    rows = [(title, r) for title, rs in buckets for r in rs]
    need = {r["src_row_id"] for _, r in rows}
    src = _load_source_texts(args.bank, need)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    probe = {}
    with torch.no_grad():
        for sid in sorted(need):
            probe[sid] = _probe(model, tok, torch, src.get(sid), device, args.max_ctx, args.topk)

    out = ["# Next-token probe on the cross-condition cherry-pick",
           f"_base={args.base_ckpt}; conds: {', '.join(conds)}; the MODEL-next line is the ground-truth "
           f"continuation the explanations are guessing (FVE is dominated by getting this right)._\n"]
    for title, rs in buckets:
        out.append(f"## {title}")
        for r in rs:
            sid = r["src_row_id"]
            summ = " / ".join(f"{c} {_row_fve(r['by_cond'][c])*100:+.0f}" for c in conds)
            out.append(f"\n**doc {r['doc_id']} · row {sid}**  ({summ})")
            out.append(f"- SOURCE: {_src_tail(src.get(sid))}")
            pr = probe.get(sid)
            if pr:
                ft, nxt = pr
                nxt_s = "  ".join(f"{t!r}({p:.2f})" for t, p in nxt)
                out.append(f"- FINAL TOKEN (verbalized): {ft!r}   |   MODEL next top-{args.topk}: {nxt_s}")
            for c in conds:
                rc = r["by_cond"][c]
                tag = (f"FVE {rc['fve_overall']*100:+.1f}%" if (rc.get("parse_success") and rc.get("fve_overall") is not None)
                       else "PARSE FAIL")
                out.append(f"  - **{c:9s}** [{tag}] {_expl_text(rc.get('generated_text'), 300)}")
        out.append("")
    report = "\n".join(out)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")
        print(f"\n[probe] -> {args.out}")


if __name__ == "__main__":
    main()
