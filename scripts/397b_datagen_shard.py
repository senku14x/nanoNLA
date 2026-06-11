"""Sharded 397B datagen for data-parallel extraction.

Run N instances, each pinned to a GPU-pair via CUDA_VISIBLE_DEVICES; each does a
contiguous shard of the (context, explanation) pairs and extracts hidden[K] at
the context's last token. A merge step concatenates shards in order.

Datagen needs NO marker / NO injection — just forward the context, grab hidden[K].
Raw (uncentered) vectors; centering happens at train time.
"""
import os, json, time, argparse
import numpy as np
import torch

import gptqmodel.nn_modules.qlinear.gemm_awq as _gawq
if not hasattr(_gawq, "AwqGEMMQuantLinear"):
    _gawq.AwqGEMMQuantLinear = type("AwqGEMMQuantLinear", (), {})
import pyarrow.parquet as pq
from gptqmodel import GPTQModel, BACKEND
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--nshards", type=int, default=3)
ap.add_argument("--n-data", type=int, default=250000)
ap.add_argument("--batch", type=int, default=64)
args = ap.parse_args()

M = "/workspace/models/qwen3.5-397b-gptq-int4"
DATA = "/workspace/nla-data/qwen3-8b-nla-ds/av_sft_shuf.parquet"
OUT = "/workspace/nla-ckpts"
os.makedirs(OUT, exist_ok=True)
K_LAYER = 40
MAXLEN = 256   # context token cap; LONGER pairs are DROPPED (truncating would extract
               # the activation mid-context while the explanation describes the full context)
GOLDS = f"{OUT}/golds_shard{args.shard}.npy"
EXPLS = f"{OUT}/expls_shard{args.shard}.json"
DOCS = f"{OUT}/doc_ids_shard{args.shard}.json"
META = f"{OUT}/shardmeta_{args.shard}.json"
t0 = time.time()


def log(*a):
    print(f"[shard{args.shard} +{int(time.time()-t0):5d}s]", *a, flush=True)


log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}  n_gpu={torch.cuda.device_count()}")
gm = GPTQModel.load(M, backend=BACKEND.TORCH, device_map="auto")
hf = gm.model
log("model loaded")
tk = AutoTokenizer.from_pretrained(M)
tk.padding_side = "right"
if tk.pad_token is None:
    tk.pad_token = tk.eos_token

# build global pairs (deterministic: same filter+order in every proc), take my contiguous shard
pf = pq.ParquetFile(DATA)
pairs = []
for rg in range(pf.num_row_groups):
    if len(pairs) >= args.n_data:
        break
    tb = pf.read_row_group(rg, columns=["response", "detokenized_text_truncated", "doc_id"])
    rs = tb.column("response").to_pylist()
    cs = tb.column("detokenized_text_truncated").to_pylist()
    ds = tb.column("doc_id").to_pylist()
    for c, r, d in zip(cs, rs, ds):
        if c and r and len(c) > 20:
            pairs.append((c, r, d))
        if len(pairs) >= args.n_data:
            break

# drop over-long contexts BEFORE sharding so every proc shards the same list
n_total = len(pairs)
kept = []
for s in range(0, n_total, 1024):
    batch = pairs[s:s + 1024]
    lens = tk([c for c, _, _ in batch], truncation=True, max_length=MAXLEN + 1)["input_ids"]
    kept.extend(p for p, ids in zip(batch, lens) if len(ids) <= MAXLEN)
pairs = kept
log(f"dropped {n_total - len(pairs)}/{n_total} pairs with context > {MAXLEN} tokens (would mislabel activation)")

N = len(pairs)
lo = args.shard * N // args.nshards
hi = (args.shard + 1) * N // args.nshards
mine = pairs[lo:hi]
log(f"global N={N}; shard [{lo}:{hi}] = {len(mine)} pairs")
meta_new = {"lo": lo, "hi": hi, "global_N": N}
if os.path.exists(META):
    meta_old = json.load(open(META))
    assert meta_old == meta_new, (
        f"shard meta drift: cache built with {meta_old}, now {meta_new} — the "
        f"pair list changed (filter/corpus/nshards); resuming would misalign "
        f"golds/expls. Delete {OUT}/*shard{args.shard}* to rebuild."
    )
else:
    json.dump(meta_new, open(META, "w"))

if os.path.exists(GOLDS) and os.path.exists(EXPLS):
    golds = list(np.load(GOLDS))
    expls = json.load(open(EXPLS))
    done = len(expls)
    golds = golds[:done]   # crash between np.save and json.dump can leave golds longer
    assert len(golds) == done
    docs = [d for _, _, d in mine[:done]]   # derived from `mine` -> rebuild prefix
    log(f"resume from {done}/{len(mine)}")
else:
    golds, expls, docs, done = [], [], [], 0

hf.eval()
for s in range(done, len(mine), args.batch):
    chunk = mine[s:s + args.batch]
    enc = tk([c for c, _, _ in chunk], return_tensors="pt", truncation=True,
             max_length=MAXLEN, padding=True).to("cuda:0")
    with torch.no_grad():
        out = hf(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                 output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[K_LAYER]
    last = (enc.attention_mask.sum(1) - 1).to(hs.device)
    h = hs[torch.arange(len(chunk), device=hs.device), last].float().cpu().numpy()
    golds.extend(list(h))
    expls.extend([r for _, r, _ in chunk])
    docs.extend([d for _, _, d in chunk])
    if (s - done) % (args.batch * 20) == 0:
        rate = (time.time() - t0) / max(1, s - done + len(chunk))
        eta = rate * (len(mine) - s) / 3600
        log(f"{len(expls)}/{len(mine)}  ({rate:.3f}s/doc, eta {eta:.1f}h)")
    if len(expls) % 4000 < args.batch:
        np.save(GOLDS, np.stack(golds))
        json.dump(expls, open(EXPLS, "w"))
        json.dump(docs, open(DOCS, "w"))
np.save(GOLDS, np.stack(golds))
json.dump(expls, open(EXPLS, "w"))
json.dump(docs, open(DOCS, "w"))
log(f"SHARD DONE: {len(expls)} pairs in {int(time.time()-t0)}s")
