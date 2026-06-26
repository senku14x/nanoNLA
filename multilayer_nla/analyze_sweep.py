"""Post-hoc analysis of the §7 SFT control-sweep held-out results.

READ-ONLY. This does NOT re-run training/eval or touch the evaluator; it only loads
the frozen outputs produced by `evaluate_e2e.py` / `eval_ar_gold.py` and reports:

  1. The four-condition TEST table with each condition's bootstrap 95% CI
     (read straight from test_<cond>.json: fve_overall_ci95 / pen_fve_overall_ci95).
  2. HEADLINE local-vs-duplicate gap, computed TWO ways:
       (a) PRE-REGISTERED: compare the two marginal bootstrap CIs (separated vs overlap).
       (b) PAIRED (more sensitive, clearly secondary): local and duplicate are evaluated
           on the SAME rl_test documents (only av_in_* differs), so join per
           (doc_id, src_row_id) and bootstrap the per-document mean difference. Two
           overlapping marginal CIs can still yield a paired-difference CI that excludes
           0; the paired test is the right power for a within-document contrast.
       Both reported for overall and per tap (prev/centre/next).
  3. Per-example DISTRIBUTION inspection from the jsonl (quantiles, parse-fail share,
     FVE split by parse success and by generation-length bucket) — look before trusting
     the means.
  4. LEAKAGE/SANITY: shuffled_pen_fve_overall must be strongly negative; test doc_ids
     must be disjoint from dev; and (if --split-seed given) every test doc must hash to
     the 'test' bucket under the same doc_bucket formula used to build the split.
  5. BOTTLENECK: end-to-end overall FVE vs ar_gold_test overall FVE (reconstructor ceiling).

Run on the box that has the results:
  python -m multilayer_nla.analyze_sweep --eval-dir $DATA/sweep_eval
Self-test (no data; fabricates outputs shaped like the real ones and checks the math):
  python -m multilayer_nla.analyze_sweep --selftest
"""

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path

import numpy as np

CONDS = ("local", "duplicate", "wide", "single", "s2_19_21_23", "s2_20_22_24")
AV_INPUT_LAYERS = {"local": "23,24,25", "duplicate": "24,24,24", "wide": "20,24,28", "single": "24",
                   "s2_19_21_23": "19,21,23", "s2_20_22_24": "20,22,24"}
TAPS = ("prev", "centre", "next")


# ----------------------------------------------------------------- loading

def _load_json(path):
    return json.loads(Path(path).read_text())


def _load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_results(eval_dir):
    """Return {cond: {"summary": dict, "rows": [per-example dicts]}} for whatever exists."""
    eval_dir = Path(eval_dir)
    test_dir = eval_dir / "test"
    out = {}
    for c in CONDS:
        sj, jl = test_dir / f"test_{c}.json", test_dir / f"test_{c}.jsonl"
        if sj.exists():
            out[c] = {"summary": _load_json(sj),
                      "rows": _load_jsonl(jl) if jl.exists() else None}
    return out


# ----------------------------------------------------------------- 1. table

def _pct(x):
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{x * 100:+.1f}"


def _ci(c):
    if not c or any(v is None or (isinstance(v, float) and v != v) for v in c):
        return "—"
    return f"[{c[0] * 100:+.1f}, {c[1] * 100:+.1f}]"


def table(results):
    hdr = ("| cond | AV in | ext% | mean tok | FVE prev | FVE centre | FVE next | "
           "FVE overall | overall CI95 | pen FVE | pen CI95 | shuffled pen |")
    sep = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines = [hdr, sep]
    for c in CONDS:
        if c not in results:
            lines.append(f"| {c} | {AV_INPUT_LAYERS[c]} | " + " | ".join(["—"] * 10) + " |")
            continue
        s = results[c]["summary"]
        ext = s.get("successful_extraction_rate")
        lines.append(
            f"| {c} | {AV_INPUT_LAYERS[c]} "
            f"| {'' if ext is None else f'{ext*100:.1f}'} "
            f"| {s.get('mean_generated_tokens', float('nan')):.0f} "
            f"| {_pct(s.get('fve_prev'))} | {_pct(s.get('fve_centre'))} | {_pct(s.get('fve_next'))} "
            f"| {_pct(s.get('fve_overall'))} | {_ci(s.get('fve_overall_ci95'))} "
            f"| {_pct(s.get('pen_fve_overall'))} | {_ci(s.get('pen_fve_overall_ci95'))} "
            f"| {_pct(s.get('shuffled_pen_fve_overall'))} |")
    return "\n".join(lines)


# ----------------------------------------------------------------- 2. headline

def _row_key(r):
    return (r["doc_id"], r.get("src_row_id"))


def _join(rows_a, rows_b):
    """Join two conditions' per-example rows on (doc_id, src_row_id). Returns
    (matched_pairs, n_only_a, n_only_b). Pairs are (doc_id, a_row, b_row)."""
    bx = {_row_key(r): r for r in rows_b}
    pairs, only_a = [], 0
    seen_b = set()
    for r in rows_a:
        k = _row_key(r)
        if k in bx:
            pairs.append((r["doc_id"], r, bx[k]))
            seen_b.add(k)
        else:
            only_a += 1
    only_b = sum(1 for r in rows_b if _row_key(r) not in seen_b)
    return pairs, only_a, only_b


def _pen_fve(r, tap=None):
    """Per-row penalized FVE: the row's fve (overall or a tap) if parsed, else 0."""
    if not r.get("parse_success"):
        return 0.0
    key = "fve_overall" if tap is None else f"fve_{tap}"
    v = r.get(key)
    return 0.0 if v is None else float(v)


def paired_bootstrap_diff(pairs, tap=None, penalized=True, n_boot=2000, seed=0):
    """Bootstrap (over DOCUMENTS) the mean per-row difference (A - B) in FVE.

    pairs: list of (doc_id, a_row, b_row), A=local, B=duplicate, joined per row.
    penalized: failure->0 (uses every matched row). penalized=False: success-only,
    restricted to rows where BOTH conditions parsed (clean within-row contrast).
    Returns dict(mean_diff, ci_lo, ci_hi, n_docs, n_rows, frac_boot_gt0)."""
    by_doc = {}
    for did, a, b in pairs:
        if penalized:
            d = _pen_fve(a, tap) - _pen_fve(b, tap)
        else:
            if not (a.get("parse_success") and b.get("parse_success")):
                continue
            ka = "fve_overall" if tap is None else f"fve_{tap}"
            va, vb = a.get(ka), b.get(ka)
            if va is None or vb is None:
                continue
            d = float(va) - float(vb)
        by_doc.setdefault(did, []).append(d)
    docs = list(by_doc.keys())
    all_diffs = [d for ds in by_doc.values() for d in ds]
    n_rows = len(all_diffs)
    if not docs or not all_diffs:
        return {"mean_diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n_docs": len(docs), "n_rows": n_rows, "frac_boot_gt0": float("nan")}
    mean_diff = float(np.mean(all_diffs))
    rng = np.random.default_rng(seed)
    vals = []
    nd = len(docs)
    for _ in range(n_boot):
        pick = rng.choice(nd, size=nd, replace=True)
        rows = [d for idx in pick for d in by_doc[docs[idx]]]
        vals.append(float(np.mean(rows)))
    vals = np.array(vals)
    return {"mean_diff": mean_diff, "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5)), "n_docs": nd, "n_rows": n_rows,
            "frac_boot_gt0": float((vals > 0).mean())}


def headline(results, n_boot=2000, seed=0):
    out = ["## HEADLINE: local vs duplicate\n"]
    have_both = "local" in results and "duplicate" in results
    if not have_both:
        return "## HEADLINE: local vs duplicate\n\n(missing local and/or duplicate test summary)\n"

    sl, sd = results["local"]["summary"], results["duplicate"]["summary"]
    # (a) pre-registered: marginal CI comparison
    out.append("### (a) Pre-registered: marginal bootstrap CIs (penalized FVE)")
    out.append(f"- local     pen FVE = {_pct(sl.get('pen_fve_overall'))}%  CI95 {_ci(sl.get('pen_fve_overall_ci95'))}")
    out.append(f"- duplicate pen FVE = {_pct(sd.get('pen_fve_overall'))}%  CI95 {_ci(sd.get('pen_fve_overall_ci95'))}")
    cl, cd = sl.get("pen_fve_overall_ci95"), sd.get("pen_fve_overall_ci95")
    if cl and cd and all(v is not None for v in [*cl, *cd]):
        separated = cl[0] > cd[1] or cd[0] > cl[1]
        out.append(f"- marginal CIs {'SEPARATED (local>duplicate)' if separated else 'OVERLAP (no separation at this N)'}")
    out.append("")

    # (b) paired (needs both jsonls)
    rl, rd = results["local"].get("rows"), results["duplicate"].get("rows")
    out.append("### (b) Paired difference, bootstrapped over shared test documents")
    if not rl or not rd:
        out.append("- (per-example jsonl missing for local and/or duplicate — cannot run the paired test)")
        return "\n".join(out)
    pairs, oa, ob = _join(rl, rd)
    out.append(f"- joined {len(pairs)} rows on (doc_id, src_row_id); unmatched local={oa} duplicate={ob}")
    if oa or ob:
        out.append("  ⚠ unmatched rows: local/duplicate did NOT evaluate the identical row set — investigate before trusting the paired test.")
    for label, pen in (("penalized (failure→0, all matched rows)", True),
                       ("success-only (both parsed)", False)):
        d = paired_bootstrap_diff(pairs, tap=None, penalized=pen, n_boot=n_boot, seed=seed)
        verdict = ("local > duplicate (CI excludes 0)" if d["ci_lo"] > 0
                   else "duplicate > local (CI excludes 0)" if d["ci_hi"] < 0
                   else "NO detectable difference (CI includes 0)")
        out.append(f"- overall {label}: Δ(local−duplicate) = {d['mean_diff']*100:+.2f}%  "
                   f"CI95 [{d['ci_lo']*100:+.2f}, {d['ci_hi']*100:+.2f}]  "
                   f"(n_docs={d['n_docs']}, n_rows={d['n_rows']}, P(Δ>0)={d['frac_boot_gt0']:.3f}) → {verdict}")
    out.append("- per-tap paired Δ (penalized):")
    for tap in TAPS:
        d = paired_bootstrap_diff(pairs, tap=tap, penalized=True, n_boot=n_boot, seed=seed)
        out.append(f"    {tap:6s} Δ = {d['mean_diff']*100:+.2f}%  CI95 [{d['ci_lo']*100:+.2f}, {d['ci_hi']*100:+.2f}]"
                   f"  P(Δ>0)={d['frac_boot_gt0']:.3f}")
    return "\n".join(out)


# ----------------------------------------------------------------- 3. distributions

def _quantiles(xs):
    if not xs:
        return "n=0"
    a = np.array(xs, dtype=float)
    qs = np.percentile(a, [0, 5, 25, 50, 75, 95, 100])
    return (f"n={len(a)} mean={a.mean()*100:+.1f} | "
            f"min/p5/p25/med/p75/p95/max = "
            + "/".join(f"{q*100:+.0f}" for q in qs))


def distributions(results):
    out = ["## Per-example distribution inspection (FVE overall, %)\n"]
    for c in CONDS:
        if c not in results or not results[c].get("rows"):
            continue
        rows = results[c]["rows"]
        ok = [r for r in rows if r.get("parse_success")]
        fail = [r for r in rows if not r.get("parse_success")]
        succ_fve = [r["fve_overall"] for r in ok if r.get("fve_overall") is not None]
        out.append(f"### {c}  (n={len(rows)}, parsed={len(ok)} [{len(ok)/max(len(rows),1)*100:.1f}%], failed={len(fail)})")
        out.append(f"- success-only FVE: {_quantiles(succ_fve)}")
        # length subgroups (generation tokens) on parsed rows
        if ok and all(r.get("generation_length") is not None for r in ok):
            lens = np.array([r["generation_length"] for r in ok], dtype=float)
            med = np.median(lens)
            short = [r["fve_overall"] for r in ok if r["generation_length"] <= med and r.get("fve_overall") is not None]
            long = [r["fve_overall"] for r in ok if r["generation_length"] > med and r.get("fve_overall") is not None]
            out.append(f"- by gen-length (parsed): ≤median({med:.0f}tok) mean FVE={np.mean(short)*100:+.1f} (n={len(short)}); "
                       f">median mean FVE={np.mean(long)*100:+.1f} (n={len(long)})")
        # fraction of parsed rows with negative FVE (worse than mean despite parsing)
        if succ_fve:
            neg = np.mean(np.array(succ_fve) < 0)
            out.append(f"- parsed rows with FVE<0 (worse than predict-the-mean): {neg*100:.1f}%")
        out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------- 3b. cross-condition compare

_EXPL_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)


def _expl_text(generated, maxchars=400):
    """The explanation payload (or raw text), whitespace-collapsed and truncated."""
    if not generated:
        return "(empty)"
    m = _EXPL_RE.search(generated)
    t = " ".join((m.group(1) if m else generated).split())
    return (t[:maxchars] + "…") if len(t) > maxchars else t


def _load_source_texts(bank_dir, needed_ids):
    """Map src_row_id -> {text, doc_id} by streaming the rl bank in the SAME sorted-glob
    order build_sweep.build_rl_eval used, so the Nth row has src_row_id == N. doc_id is
    carried so the caller can verify the join (bank doc_id must == the eval row's doc_id).
    Only `needed_ids` are kept."""
    import glob as _glob
    import pyarrow.parquet as _pq
    paths = sorted(_glob.glob(str(Path(bank_dir) / "rl.shard*of*.parquet")))
    if not paths:
        return {}
    needed = set(needed_ids)
    out, gidx = {}, 0
    for p in paths:
        pf = _pq.ParquetFile(p)
        names = pf.schema_arrow.names
        cols = [c for c in ("detokenized_text_truncated", "doc_id") if c in names]
        for b in pf.iter_batches(batch_size=8192, columns=cols):
            txt = (b.column("detokenized_text_truncated").to_pylist()
                   if "detokenized_text_truncated" in cols else [None] * b.num_rows)
            dids = b.column("doc_id").to_pylist() if "doc_id" in cols else [None] * b.num_rows
            for i in range(b.num_rows):
                if gidx + i in needed:
                    out[gidx + i] = {"text": txt[i], "doc_id": dids[i]}
            gidx += b.num_rows
    return out


def verify_source_join(joined, src_texts):
    """Cross-check the src_row_id->bank join: the bank row's doc_id MUST equal the eval
    row's doc_id (a misordered shard read would land in a different doc). Returns
    (n_ok, n_total, n_missing)."""
    n_ok = n_missing = 0
    for x in joined:
        rec = src_texts.get(x["src_row_id"])
        if rec is None:
            n_missing += 1
        elif rec.get("doc_id") == x["doc_id"]:
            n_ok += 1
    return n_ok, len(joined), n_missing


def _src_tail(text, chars=260):
    """The END of the source prefix (the activation sits at its final token), collapsed."""
    if not text:
        return "(source text unavailable)"
    t = " ".join(text.split())
    return ("…" + t[-chars:]) if len(t) > chars else t


def _join_conditions(results):
    """Inner-join per-example rows across all available conditions on (doc_id, src_row_id).
    Returns (conds_present, [{doc_id, src_row_id, by_cond:{cond:row}}]) — only rows that exist
    in EVERY available condition (same input position, fair side-by-side)."""
    conds = [c for c in CONDS if results.get(c, {}).get("rows")]
    if len(conds) < 2:
        return conds, []
    idx = {c: {(_r["doc_id"], _r.get("src_row_id")): _r for _r in results[c]["rows"]} for c in conds}
    keys = set.intersection(*[set(idx[c]) for c in conds])
    joined = [{"doc_id": k[0], "src_row_id": k[1], "by_cond": {c: idx[c][k] for c in conds}}
              for k in keys]
    return conds, joined


def _row_fve(r):
    """Per-row penalized FVE: the row's overall FVE if parsed, else 0 (failure)."""
    v = r.get("fve_overall")
    return float(v) if (r.get("parse_success") and v is not None) else 0.0


def _cmp_block(row, conds):
    lines = []
    for c in conds:
        r = row["by_cond"][c]
        tag = (f"FVE {r['fve_overall']*100:+.1f}%" if (r.get("parse_success") and r.get("fve_overall") is not None)
               else "PARSE FAIL")
        lines.append(f"- **{c:9s}** [{tag}] {_expl_text(r.get('generated_text'))}")
    return "\n".join(lines)


def _compare_buckets(joined, conds, k):
    """The sampled row buckets for the cross-condition view: local-vs-all extremes, the
    honest counterweight, the typical (median) case, and widest disagreement. Returned as
    [(title, [rows])] so the compare table AND the next-token probe show the SAME rows."""
    buckets = []
    if "local" in conds and len(conds) > 1:
        # Rows where local beats ALL others by the most (and loses to all by the most).
        # Extremes selected to favour/disfavour local — a SELECTION EFFECT, not evidence.
        others = [c for c in conds if c != "local"]
        margin = lambda x: _row_fve(x["by_cond"]["local"]) - max(_row_fve(x["by_cond"][o]) for o in others)
        by_margin = sorted(joined, key=margin)
        buckets.append(("local >> ALL others — cherry-picked FOR local (selection effect; illustrative, NOT evidence)",
                        by_margin[-k:][::-1]))
        buckets.append(("local << ALL others — cherry-picked AGAINST local (the honest counterweight)",
                        by_margin[:k]))
    if "local" in conds and "duplicate" in conds:
        joined.sort(key=lambda x: _row_fve(x["by_cond"]["local"]) - _row_fve(x["by_cond"]["duplicate"]))
        m = len(joined) // 2
        buckets += [("near-median local-duplicate (the typical case, not an extreme)",
                     joined[max(0, m - k // 2): m + (k - k // 2)])]
    if not buckets:
        buckets = [("sample rows", joined[:k])]
    spread = lambda x: (lambda vs: max(vs) - min(vs))([_row_fve(x["by_cond"][c]) for c in conds])
    buckets.append(("widest spread across conditions", sorted(joined, key=spread, reverse=True)[:k]))
    return buckets


def compare(results, k=4, bank_dir=None):
    """Side-by-side generated explanations across conditions for the SAME doc/row, sampled at
    the local-vs-duplicate extremes, the median, and the widest cross-condition disagreement.
    If bank_dir is given, the source prefix (ending at the verbalized final token) is shown."""
    conds, joined = _join_conditions(results)
    if not joined:
        return "## Cross-condition comparison\n\n(need >=2 conditions' jsonl with matching rows)\n"
    src_texts = _load_source_texts(bank_dir, {x["src_row_id"] for x in joined}) if bank_dir else {}
    out = [f"## Cross-condition comparison — same doc/row ({len(joined)} joined; conds: {', '.join(conds)})",
           "_Read whether the AV input config actually changes the explanation's CONTENT, or just",
           "produces near-identical boilerplate that scores by distributional luck._\n"]
    if src_texts:
        ok, tot, miss = verify_source_join(joined, src_texts)
        flag = "✓ aligned" if (ok == tot and miss == 0) else f"⚠ {tot - ok} MISMATCH / {miss} missing — SOURCE TEXT UNRELIABLE"
        out.append(f"_source-join check: bank doc_id matches eval doc_id for {ok}/{tot} rows — {flag}_\n")
    buckets = _compare_buckets(joined, conds, k)
    for title, rows in buckets:
        out.append(f"### {title}")
        for x in rows:
            summ = " / ".join(f"{c} {_row_fve(x['by_cond'][c])*100:+.0f}" for c in conds)
            out.append(f"\n**doc {x['doc_id']} · row {x['src_row_id']}**  ({summ})")
            if src_texts:
                out.append(f"- _SOURCE (ends at the verbalized final token)_: {_src_tail((src_texts.get(x['src_row_id']) or {}).get('text'))}")
            out.append(_cmp_block(x, conds))
        out.append("")
    return "\n".join(out)


# ----------------------------------------------------------------- 4. leakage / sanity

def doc_bucket(doc_id, fracs, seed):
    """EXACT copy of multilayer_nla.datasets.doc_bucket — re-derive the split to verify."""
    h = hashlib.sha256(f"{seed}|{doc_id}".encode()).digest()
    u = int.from_bytes(h[:8], "big") / float(1 << 64)
    cum = 0.0
    for i, f in enumerate(fracs):
        cum += f
        if u < cum:
            return i
    return len(fracs) - 1


def leakage_checks(results, eval_dir, split_seed=None, fracs=(0.8, 0.1, 0.1)):
    out = ["## Leakage / sanity checks\n"]
    # shuffled control must collapse strongly negative
    out.append("### Shuffled-generation control (penalized FVE; must be ≈0 or strongly negative)")
    for c in CONDS:
        if c in results:
            s = results[c]["summary"].get("shuffled_pen_fve_overall")
            flag = "" if (s is not None and s < -0.05) else "  ⚠ NOT collapsed — investigate"
            out.append(f"- {c}: {_pct(s)}%{flag}")
    out.append("")

    # test/dev doc disjointness (uses dev jsonls if present)
    eval_dir = Path(eval_dir)
    dev_dir = eval_dir / "dev"
    out.append("### Test vs dev document disjointness")
    for c in CONDS:
        if c not in results or not results[c].get("rows"):
            continue
        test_docs = {r["doc_id"] for r in results[c]["rows"]}
        dev_jl = sorted(glob.glob(str(dev_dir / f"dev_{c}*.jsonl")))
        if dev_jl:
            dev_docs = {r["doc_id"] for p in dev_jl for r in _load_jsonl(p)}
            overlap = test_docs & dev_docs
            out.append(f"- {c}: test docs={len(test_docs)}, dev docs={len(dev_docs)}, "
                       f"overlap={len(overlap)}{'  ⚠ LEAK' if overlap else '  ✓'}")
        else:
            out.append(f"- {c}: test docs={len(test_docs)} (no dev jsonl found to cross-check)")
    out.append("")

    # re-derive the split: every test doc must hash to the 'test' bucket
    if split_seed is not None:
        out.append(f"### Re-derived split (seed={split_seed}, fracs={fracs}): every test doc must land in bucket 2 (test)")
        for c in CONDS:
            if c not in results or not results[c].get("rows"):
                continue
            test_docs = {r["doc_id"] for r in results[c]["rows"]}
            wrong = [d for d in test_docs if doc_bucket(d, fracs, split_seed) != 2]
            out.append(f"- {c}: {len(test_docs)} test docs, {len(wrong)} NOT in test bucket"
                       f"{'  ⚠ ' + str(wrong[:3]) if wrong else '  ✓'}")
    return "\n".join(out)


# ----------------------------------------------------------------- 5. bottleneck

def bottleneck(results, eval_dir):
    out = ["## Bottleneck: end-to-end vs AR-gold ceiling\n"]
    test_dir = Path(eval_dir) / "test"
    ar_gold = test_dir / "ar_gold_test.json"
    if not ar_gold.exists():
        return "## Bottleneck\n\n(ar_gold_test.json missing)\n"
    g = _load_json(ar_gold)
    out.append(f"- AR-gold TEST FVE (gold expl → AR → [L23,L24,L25]) "
               f"prev/centre/next/overall = {_pct(g.get('fve_prev'))}/{_pct(g.get('fve_centre'))}/"
               f"{_pct(g.get('fve_next'))}/{_pct(g.get('fve_overall'))}%")
    for c in CONDS:
        if c in results:
            e2e = results[c]["summary"].get("fve_overall")
            gap = None if (e2e is None or g.get("fve_overall") is None) else g["fve_overall"] - e2e
            out.append(f"- {c}: e2e overall {_pct(e2e)}% vs ceiling {_pct(g.get('fve_overall'))}% "
                       f"→ gap {_pct(gap)}pp "
                       f"({'verbalizer/extraction-limited' if (gap or 0) > 0.05 else 'reconstructor-limited / at ceiling'})")
    return "\n".join(out)


# ----------------------------------------------------------------- driver / selftest

def run(eval_dir, split_seed=None, n_boot=2000, seed=0, compare_k=4, bank_dir=None):
    results = load_results(eval_dir)
    if not results:
        return (f"No test_<cond>.json found under {Path(eval_dir)/'test'}. "
                f"This box has no sweep results — copy $DATA/sweep_eval here or run on the H200.")
    parts = [table(results), "", headline(results, n_boot, seed), "",
             distributions(results), "", compare(results, compare_k, bank_dir), "",
             leakage_checks(results, eval_dir, split_seed), "",
             bottleneck(results, eval_dir)]
    return "\n".join(parts)


def _selftest():
    """Fabricate outputs shaped like the real ones (with a KNOWN local>duplicate effect
    on the SAME docs) and confirm the paired test recovers it and leakage checks fire."""
    import tempfile
    rng = np.random.default_rng(0)
    n_docs, ppd = 200, 2
    base = {}  # (doc,pos) -> base difficulty
    rows_by_cond = {c: [] for c in CONDS}
    # ground-truth: local adds +0.05 FVE per row over duplicate on the SAME rows
    for di in range(n_docs):
        did = f"doc{di}"
        for pos in range(ppd):
            mu = rng.uniform(0.1, 0.6)
            parsed = rng.random() > 0.03
            for c in CONDS:
                bump = {"local": 0.05, "wide": 0.04, "duplicate": 0.0, "single": -0.03}.get(c, 0.01)
                fve = mu + bump + rng.normal(0, 0.02) if parsed else None
                taps = None
                rows_by_cond[c].append({
                    "doc_id": did, "src_row_id": di * ppd + pos,
                    "generated_text": f"<explanation>{c}: doc{di} pos{pos} on topic {di % 5}</explanation>",
                    "parse_success": parsed,
                    "fve_prev": (fve - 0.01) if parsed else None,
                    "fve_centre": fve if parsed else None,
                    "fve_next": (fve + 0.01) if parsed else None,
                    "fve_overall": fve if parsed else None,
                    "generation_length": int(rng.integers(20, 120)),
                })
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp) / "sweep_eval" / "test"
        td.mkdir(parents=True)
        for c in CONDS:
            ok = [r for r in rows_by_cond[c] if r["parse_success"]]
            fves = [r["fve_overall"] for r in ok]
            summ = {
                "condition": c, "n_total": len(rows_by_cond[c]), "n_success": len(ok),
                "successful_extraction_rate": len(ok) / len(rows_by_cond[c]),
                "fve_prev": float(np.mean([r["fve_prev"] for r in ok])),
                "fve_centre": float(np.mean([r["fve_centre"] for r in ok])),
                "fve_next": float(np.mean([r["fve_next"] for r in ok])),
                "fve_overall": float(np.mean(fves)),
                "pen_fve_overall": float(np.mean([r["fve_overall"] if r["parse_success"] else 0.0
                                                  for r in rows_by_cond[c]])),
                "fve_overall_ci95": [float(np.mean(fves)) - 0.03, float(np.mean(fves)) + 0.03],
                "pen_fve_overall_ci95": [float(np.mean(fves)) - 0.04, float(np.mean(fves)) + 0.02],
                "shuffled_pen_fve_overall": -0.76,
                "mean_generated_tokens": 70.0, "median_generated_tokens": 68.0,
            }
            (td / f"test_{c}.json").write_text(json.dumps(summ))
            with open(td / f"test_{c}.jsonl", "w") as f:
                for r in rows_by_cond[c]:
                    f.write(json.dumps(r) + "\n")
        (td / "ar_gold_test.json").write_text(json.dumps(
            {"fve_prev": 0.6, "fve_centre": 0.62, "fve_next": 0.6, "fve_overall": 0.61}))
        report = run(td.parent, split_seed=None, n_boot=1000, seed=0)
        print(report)
        # assertions on the math
        results = load_results(td.parent)
        pairs, oa, ob = _join(results["local"]["rows"], results["duplicate"]["rows"])
        assert oa == 0 and ob == 0, "selftest rows should match 1:1"
        d = paired_bootstrap_diff(pairs, penalized=False, n_boot=2000, seed=0)
        assert d["ci_lo"] > 0, f"paired test failed to recover +0.05 effect: {d}"
        assert abs(d["mean_diff"] - 0.05) < 0.01, f"effect size off: {d['mean_diff']}"
        print(f"\n[selftest] paired Δ recovered = {d['mean_diff']*100:+.2f}% "
              f"CI95 [{d['ci_lo']*100:+.2f},{d['ci_hi']*100:+.2f}]  ✓ excludes 0")
        cj, joined = _join_conditions(results)
        assert cj == list(CONDS) and len(joined) == n_docs * ppd, (cj, len(joined))
        print(f"[selftest] compare join: {len(cj)} conds x {len(joined)} shared rows  ✓")
        print("[selftest] PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", help="sweep_eval dir (expects test/ subdir with test_<cond>.json[l])")
    p.add_argument("--split-seed", type=int, default=None, help="if given, re-derive the doc split to verify test bucket")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the report markdown here too")
    p.add_argument("--examples", type=int, default=4, help="rows per bucket in the cross-condition compare")
    p.add_argument("--bank", help="rl bank dir (REGEN); if set, shows the source prefix per cherry-picked row")
    p.add_argument("--selftest", action="store_true", help="run on fabricated data; no real results needed")
    args = p.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.eval_dir:
        raise SystemExit("--eval-dir required (or --selftest)")
    report = run(args.eval_dir, args.split_seed, args.n_boot, args.seed, args.examples, args.bank)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")
        print(f"\n[analyze] -> {args.out}")


if __name__ == "__main__":
    main()
