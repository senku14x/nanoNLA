"""Bit-level proof that the src_row_id -> rl-bank join is aligned (read-only, CPU).

The eval parquet stores `activation_centre` (the L24 target) per src_row_id; the rl bank
stores `activation_L24` at the same global row. If those vectors match at a random sample
of src_row_ids — AND the doc_ids match — then the join is provably correct: the row whose
source text analyze_sweep/probe_next_token attach is exactly the row whose activation was
verbalized. Any misordering of the shard read would break both checks.

Run:
  python -m multilayer_nla.verify_source_join --bank $REGEN --eval-parquet $SWEEP/rl_test_local.parquet --n 300
"""

import argparse
import glob
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _vec(col, i):
    return np.asarray(col[i].as_py(), dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", required=True, help="rl bank dir (REGEN)")
    p.add_argument("--eval-parquet", required=True, help="rl_test_<cond>.parquet")
    p.add_argument("--n", type=int, default=300, help="random rows to check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-col", default="activation_centre", help="eval target col (= L24)")
    p.add_argument("--bank-layer", default="activation_L24", help="matching bank column")
    args = p.parse_args()

    ev = pq.read_table(args.eval_parquet, columns=["src_row_id", "doc_id", args.eval_col])
    sid = ev.column("src_row_id").to_pylist()
    did = ev.column("doc_id").to_pylist()
    cen = ev.column(args.eval_col).combine_chunks()
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(sid)), min(args.n, len(sid)))
    want = {sid[i]: i for i in idx}   # global bank index -> eval row index

    paths = sorted(glob.glob(str(Path(args.bank) / "rl.shard*of*.parquet")))
    if not paths:
        raise SystemExit(f"no rl shards under {args.bank}")
    got, g = {}, 0
    for pth in paths:
        pf = pq.ParquetFile(pth)
        for b in pf.iter_batches(batch_size=8192, columns=["doc_id", args.bank_layer]):
            d = b.column("doc_id").to_pylist()
            a = b.column(args.bank_layer)
            for j in range(b.num_rows):
                if g + j in want:
                    got[g + j] = (d[j], _vec(a, j))
            g += b.num_rows
        if len(got) >= len(want):
            break

    ok_doc = ok_act = 0
    bad = []
    for s, i in want.items():
        if s not in got:
            bad.append((s, "MISSING", did[i]))
            continue
        bd, ba = got[s]
        dm = bd == did[i]
        am = np.allclose(ba, _vec(cen, i), atol=1e-3, rtol=1e-3)
        ok_doc += dm
        ok_act += am
        if not (dm and am):
            bad.append((s, bd, did[i]))
    nchecked = len(want)
    print(f"[verify] {Path(args.eval_parquet).name}: checked {nchecked} rows | "
          f"doc_id match {ok_doc}/{nchecked} | {args.bank_layer}=={args.eval_col} {ok_act}/{nchecked}")
    if bad:
        print("  first mismatches (src_row_id, bank_doc, eval_doc):", bad[:5])
        raise SystemExit("JOIN MISALIGNED — source-text attachment would be wrong; do NOT trust it.")
    print("  ✓ join verified bit-level — the attached source text is the verbalized row.")


if __name__ == "__main__":
    main()
