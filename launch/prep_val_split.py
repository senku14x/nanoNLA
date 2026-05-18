"""Split av_sft_shuf.parquet 90/10 → av_train.parquet + av_val.parquet (sidecars copied)."""

import argparse
import shutil
from pathlib import Path

import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--train-out", required=True)
    p.add_argument("--val-out", required=True)
    p.add_argument("--val-frac", type=float, default=0.1)
    args = p.parse_args()

    in_path = Path(args.input)
    sidecar = in_path.with_suffix(in_path.suffix + ".nla_meta.yaml")
    assert sidecar.exists(), f"missing sidecar {sidecar}"

    table = pq.read_table(args.input)
    n = table.num_rows
    n_val = int(n * args.val_frac)
    n_train = n - n_val
    print(f"input rows: {n} → train {n_train} / val {n_val}")

    # Take FIRST n_train as train, LAST n_val as val. Source is shuffled already.
    train_tbl = table.slice(0, n_train)
    val_tbl = table.slice(n_train, n_val)

    pq.write_table(train_tbl, args.train_out)
    pq.write_table(val_tbl, args.val_out)
    shutil.copy2(sidecar, args.train_out + ".nla_meta.yaml")
    shutil.copy2(sidecar, args.val_out + ".nla_meta.yaml")
    print(f"wrote {args.train_out} + sidecar")
    print(f"wrote {args.val_out} + sidecar")


if __name__ == "__main__":
    main()
