"""Merge sharded datagen outputs into the canonical full cache (in shard order)."""
import argparse, json, os
import numpy as np

OUT = "/workspace/nla-ckpts"
NSHARDS = 3
BASE = f"{OUT}/qwen3.5-397b-nla247k"

ap = argparse.ArgumentParser()
ap.add_argument("--expect-total", type=int,
                default=int(os.environ.get("EXPECT_TOTAL", 0)) or None,
                help="assert merged row count equals this (post-filter global N)")
args = ap.parse_args()

golds, expls, doc_ids, global_ns = [], [], [], []
for i in range(NSHARDS):
    g = np.load(f"{OUT}/golds_shard{i}.npy")
    e = json.load(open(f"{OUT}/expls_shard{i}.json"))
    assert len(g) == len(e), f"shard {i} mismatch: {len(g)} vs {len(e)}"
    assert len(g) > 0, f"shard {i} is EMPTY"
    meta_p = f"{OUT}/shardmeta_{i}.json"
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        expect = meta["hi"] - meta["lo"]
        assert len(g) == expect, f"shard {i} INCOMPLETE: {len(g)} rows, expected {expect}"
        global_ns.append(meta["global_N"])
    docs_p = f"{OUT}/doc_ids_shard{i}.json"
    if os.path.exists(docs_p):
        d = json.load(open(docs_p))
        assert len(d) == len(e), f"shard {i} doc_ids mismatch: {len(d)} vs {len(e)}"
        doc_ids += d
    golds.append(g)
    expls += e
    print(f"shard {i}: {len(g)}")
golds = np.concatenate(golds, 0)
if global_ns:
    assert len(set(global_ns)) == 1, f"shards disagree on global N: {global_ns}"
    assert len(golds) == global_ns[0], f"merged {len(golds)} != global N {global_ns[0]}"
if args.expect_total is not None:
    assert len(golds) == args.expect_total, f"merged {len(golds)} != --expect-total {args.expect_total}"
np.save(f"{BASE}_golds_full.npy", golds)
json.dump(expls, open(f"{BASE}_expls_full.json", "w"))
if len(doc_ids) == len(expls):
    json.dump(doc_ids, open(f"{BASE}_doc_ids_full.json", "w"))
else:
    print(f"[WARN] doc_ids incomplete ({len(doc_ids)}/{len(expls)}) — not saving; eval splits will be doc-leaky")
# compute + save the centering mean now (full 247k estimate) for AV + AR + RL consistency
mean = golds.mean(0, keepdims=True)
np.save(f"{BASE}_mean.npy", mean)
print(f"MERGED {golds.shape}, {len(expls)} expls; mean saved -> {BASE}_mean.npy")
