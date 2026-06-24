"""Phase 3 gate: compare a bucketed vs unbucketed regen of the same slice.

Verifies row IDENTITY + metadata order are unchanged and the activations are
numerically equivalent. Prints a concise report and exits:
  0  -> safe to proceed (identical rows/metadata AND numerically close)
  2  -> STOP: row identity / metadata order / retained-row set differ
  3  -> STOP: large numerical discrepancy

    python -m multilayer_nla.ops.bench_compare --baseline a.parquet --bucketed b.parquet \\
        --layers 19-29 --max-abs 1e-3 --min-cos 0.99999
"""

import argparse
import sys

import numpy as np
import pyarrow.parquet as pq


def _layers(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return sorted(set(out))


def _col_matrix(t, name):
    c = t.column(name).combine_chunks()
    return c.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(c), -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--bucketed", required=True)
    ap.add_argument("--layers", default="19-29")
    ap.add_argument("--max-abs", type=float, default=1e-3, help="max per-layer abs diff to accept")
    ap.add_argument("--min-cos", type=float, default=0.99999, help="min per-row cosine to accept")
    ap.add_argument("--id-cols", default="doc_id,n_raw_tokens", help="non-activation cols that must match exactly & in order")
    args = ap.parse_args()
    layers = _layers(args.layers)

    a = pq.read_table(args.baseline)
    b = pq.read_table(args.bucketed)
    print(f"rows: baseline={a.num_rows}  bucketed={b.num_rows}")
    if a.num_rows != b.num_rows:
        print("STOP: retained-row count differs (rounding/drop divergence)"); sys.exit(2)

    # 1) non-activation columns identical AND in the same order
    for col in [c.strip() for c in args.id_cols.split(",") if c.strip()]:
        if col not in a.schema.names or col not in b.schema.names:
            print(f"STOP: id column {col!r} missing in one side"); sys.exit(2)
        if a.column(col).to_pylist() != b.column(col).to_pylist():
            print(f"STOP: id column {col!r} differs (order or content)"); sys.exit(2)
    print(f"id columns match exactly & in order: {args.id_cols}")

    # 2) all expected activation columns present
    for k in layers:
        for name, t in (("baseline", a), ("bucketed", b)):
            if f"activation_L{k}" not in t.schema.names:
                print(f"STOP: {name} missing activation_L{k}"); sys.exit(2)

    # 3) per-layer abs diff + per-row cosine
    worst_abs, worst_mean, min_cos = 0.0, 0.0, 1.0
    for k in layers:
        x = _col_matrix(a, f"activation_L{k}")
        y = _col_matrix(b, f"activation_L{k}")
        d = np.abs(x - y)
        mx, mn = float(d.max()), float(d.mean())
        num = (x * y).sum(1)
        den = (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)) + 1e-12
        cos = (num / den)
        cmin = float(cos.min())
        worst_abs, worst_mean, min_cos = max(worst_abs, mx), max(worst_mean, mn), min(min_cos, cmin)
        print(f"  L{k}: max|Δ|={mx:.3e}  mean|Δ|={mn:.3e}  min cos={cmin:.8f}")

    print(f"OVERALL: max|Δ|={worst_abs:.3e}  mean|Δ|={worst_mean:.3e}  min cos={min_cos:.8f}")
    if worst_abs > args.max_abs or min_cos < args.min_cos:
        print(f"STOP: numerical discrepancy exceeds thresholds (max-abs {args.max_abs}, min-cos {args.min_cos})")
        sys.exit(3)
    print("PASS: rows/metadata identical and activations numerically equivalent — safe to proceed.")


if __name__ == "__main__":
    main()
