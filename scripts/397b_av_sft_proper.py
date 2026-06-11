"""PROPER AV-SFT for 397B NLA.

The 150-step demo trained the verbalizer on ~1,200 examples (≈0.2 epochs on a
6k subset) -> word-salad explanations (NLL ~13). The paper trained the AV on
~250k (activation, explanation) pairs for ~1 epoch (batch 256, cosine 2e-5).

This re-runs AV-SFT at scale, faithful to that recipe but as a LoRA:
  - datagen N_DATA fresh (gold-activation, 8B-explanation) pairs, re-extracting
    397B hidden[K] (incremental cache -> resumable)
  - r=128 rsLoRA alpha16, effective batch 32, cosine LR, multi-epoch with
    held-out NLL early-stopping + best-checkpointing.

Reuses the 6k-run's centering MEAN (_mean.npy) so the new AV stays in the SAME
centered activation space as the already-trained AR (RL needs them consistent).
The AR is NOT retrained (already 33% held-out FVE).

Run resident (one ~24-min load): load -> datagen -> AV-SFT.
"""
import os, json, time, math
import numpy as np
import torch
import torch.nn.functional as F

# peft<->gptqmodel 7.0 import shim (peft 0.19 imports a class gptqmodel renamed)
import gptqmodel.nn_modules.qlinear.gemm_awq as _gawq
if not hasattr(_gawq, "AwqGEMMQuantLinear"):
    _gawq.AwqGEMMQuantLinear = type("AwqGEMMQuantLinear", (), {})

import pyarrow.parquet as pq
from gptqmodel import GPTQModel, BACKEND
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

M = "/workspace/models/qwen3.5-397b-gptq-int4"
DATA = "/workspace/nla-data/qwen3-8b-L24/av_sft_shuf.parquet"
CK = "/workspace/nla-ckpts/qwen3.5-397b-nla6k"     # existing run: _mean.npy lives here
GOLDS = CK + "_golds_full.npy"   # canonical full activation cache (was _golds60k, renamed on box)
EXPLS = CK + "_expls_full.json"
DOCS = CK + "_doc_ids_full.json"
SAVE_AV = CK + "_av_lora_proper"
MARKER = "☴"          # single-token marker
K_LAYER = 40              # hidden_states[40] of 60
MSE_SCALE = 64.0
TGT = ["q_proj", "k_proj", "v_proj", "o_proj",
       "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
N_DATA = 250000          # ALL usable pairs (~247k) — single pass, NEVER epoch on activations
N_EVAL = 2000            # held-out tail for NLL early-stop
MAXLEN = 256             # context token cap; LONGER pairs are DROPPED (truncating would extract
                         # the activation mid-context while the explanation describes the full context)
DATAGEN_BATCH = 64       # forward-only + weight-read-bound -> bigger batch = ~2x faster datagen
CACHE_EVERY = 4000       # datagen checkpoint cadence (docs)
R, ALPHA = 128, 16
EFF_BATCH = 256          # paper's global batch (free here: cost is fwd/bwd passes, not opt-steps)
LR_PEAK, LR_END, WARMUP = 3e-5, 3e-6, 100   # 3e-5 = our standard for activation-oracle LoRAs
EPOCHS = 1               # single pass over all ~247k unique pairs
EVAL_EVERY = 100         # opt-steps between held-out evals (eff256 -> ~960 opt-steps/epoch)
PATIENCE = 4             # stop after this many evals with no held-out improvement
DEV0 = "cuda:0"
t_start = time.time()


def log(*a):
    print(f"[+{int(time.time()-t_start):6d}s]", *a, flush=True)


def normalize(v, scale=MSE_SCALE):
    n = v.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return v.float() / n * scale


# ============================== load ==============================
t = time.time()
gm = GPTQModel.load(M, backend=BACKEND.TORCH, device_map="auto")
hf = gm.model
log(f"loaded in {int(time.time()-t)}s")
dec = hf
for a in ("model", "language_model"):
    if hasattr(dec, a):
        dec = getattr(dec, a)
log("decoder", type(dec).__name__, "n_layers", len(dec.layers))
tk = AutoTokenizer.from_pretrained(M)
tk.padding_side = "right"
if tk.pad_token is None:
    tk.pad_token = tk.eos_token
mid = tk.encode(MARKER, add_special_tokens=False)
assert len(mid) == 1, f"marker {MARKER!r} not single-token: {mid}"
MARKER_ID = mid[0]
log("marker", repr(MARKER), "id", MARKER_ID)

# ---- injection hook (norm-matched ADD at marker on dec.layers[1]) ----
INJ = {"v": None, "ids": None}


def emb_hook(module, args, kwargs, output):
    ids = kwargs.get("input") if kwargs else None
    if ids is None and args:
        ids = args[0]
    INJ["ids"] = ids
    return output


def layer_hook(module, args, output):
    resid = output[0] if isinstance(output, tuple) else output
    v, ids = INJ["v"], INJ["ids"]
    if v is None or ids is None or resid.shape[1] < 2:
        return output
    ids = ids.to(resid.device)
    out = resid.clone()
    for b in range(ids.shape[0]):
        pos = (ids[b] == MARKER_ID).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            continue
        p = int(pos[0])
        hp = out[b, p].float()
        hn = hp.norm().clamp(min=1e-8)
        vv = v[b].to(resid.device).float()
        out[b, p] = (hp + hn * vv / vv.norm().clamp(min=1e-8)).to(out.dtype)
    return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out


hf.get_input_embeddings().register_forward_hook(emb_hook, with_kwargs=True)
dec.layers[1].register_forward_hook(layer_hook)

# ============================== datagen (resumable) ==============================
log(f"=== datagen up to {N_DATA} pairs (397B hidden[{K_LAYER}]) ===")
pf = pq.ParquetFile(DATA)
pairs = []
for rg in range(pf.num_row_groups):
    if len(pairs) >= N_DATA:
        break
    tb = pf.read_row_group(rg, columns=["response", "detokenized_text_truncated", "doc_id"])
    rs = tb.column("response").to_pylist()
    cs = tb.column("detokenized_text_truncated").to_pylist()
    ds = tb.column("doc_id").to_pylist()
    for c, r, d in zip(cs, rs, ds):
        if c and r and len(c) > 20:
            pairs.append((c, r, d))
        if len(pairs) >= N_DATA:
            break

# drop over-long contexts (truncated extraction would mislabel the pair)
n_total = len(pairs)
kept = []
for s in range(0, n_total, 1024):
    batch = pairs[s:s + 1024]
    lens = tk([c for c, _, _ in batch], truncation=True, max_length=MAXLEN + 1)["input_ids"]
    kept.extend(p for p, ids in zip(batch, lens) if len(ids) <= MAXLEN)
pairs = kept
log(f"dropped {n_total - len(pairs)}/{n_total} pairs with context > {MAXLEN} tokens (would mislabel activation)")

N = len(pairs)
log(f"{N} usable (context, explanation) pairs available")

if os.path.exists(GOLDS) and os.path.exists(EXPLS):
    golds = list(np.load(GOLDS))
    expls = json.load(open(EXPLS))
    done = len(expls)
    golds = golds[:done]   # crash between np.save and json.dump can leave golds longer
    assert len(golds) == done
    docs = [d for _, _, d in pairs[:done]]   # derived from `pairs` -> rebuild prefix
    log(f"resuming datagen from cache: {done}/{N} done")
else:
    golds, expls, docs, done = [], [], [], 0

if done < N:
    hf.eval()
    t0 = time.time()
    for s in range(done, N, DATAGEN_BATCH):
        chunk = pairs[s:s + DATAGEN_BATCH]
        enc = tk([c for c, _, _ in chunk], return_tensors="pt", truncation=True,
                 max_length=MAXLEN, padding=True).to(DEV0)
        INJ["v"] = None
        with torch.no_grad():
            out = hf(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                     output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[K_LAYER]
        last = (enc.attention_mask.sum(1) - 1).to(hs.device)
        h = hs[torch.arange(len(chunk), device=hs.device), last].float().cpu().numpy()
        golds.extend(list(h))
        expls.extend([r for _, r, _ in chunk])
        docs.extend([d for _, _, d in chunk])
        if (s - done) % (DATAGEN_BATCH * 20) == 0:
            rate = (time.time() - t0) / max(1, s - done + len(chunk))
            eta = rate * (N - s) / 3600
            log(f"  datagen {len(expls)}/{N}  ({rate:.3f}s/doc, eta {eta:.1f}h)")
        if len(expls) % CACHE_EVERY < DATAGEN_BATCH:
            np.save(GOLDS, np.stack(golds))
            json.dump(expls, open(EXPLS, "w"))
            json.dump(docs, open(DOCS, "w"))
    np.save(GOLDS, np.stack(golds))
    json.dump(expls, open(EXPLS, "w"))
    json.dump(docs, open(DOCS, "w"))
    log(f"datagen done: {len(expls)} pairs in {int(time.time()-t0)}s")

golds = np.stack(golds) if isinstance(golds, list) else golds
assert len(golds) == len(expls), f"golds/expls misaligned: {len(golds)} vs {len(expls)}"
log(f"golds {golds.shape}")

# ============================== center (reuse 6k mean) ==============================
assert os.path.exists(CK + "_mean.npy"), "need the 6k _mean.npy for consistent AV/AR centering"
MEAN = np.load(CK + "_mean.npy")
new_mean = golds.mean(0, keepdims=True)
drift = float(np.linalg.norm(MEAN - new_mean) / np.linalg.norm(MEAN))
log(f"centering: reuse 6k mean; drift ||mean6k - mean60k||/||mean6k|| = {drift:.4f}")
golds = golds - MEAN
golds = torch.tensor(golds, dtype=torch.float32)

# doc-disjoint eval split: eval = last N_EVAL rows; drop train rows sharing a doc
if len(docs) == len(expls):
    ev_doc_set = set(docs[-N_EVAL:])
    keep = [i for i in range(len(expls) - N_EVAL) if docs[i] not in ev_doc_set]
    n_leak = (len(expls) - N_EVAL) - len(keep)
    log(f"doc-disjoint split: dropped {n_leak} train rows sharing docs with eval")
    tr_g, ev_g = golds[keep], golds[-N_EVAL:]
    tr_e, ev_e = [expls[i] for i in keep], expls[-N_EVAL:]
else:
    log(f"[WARN] no doc_ids ({len(docs)} vs {len(expls)}) — eval split is doc-leaky")
    tr_g, ev_g = golds[:-N_EVAL], golds[-N_EVAL:]
    tr_e, ev_e = expls[:-N_EVAL], expls[-N_EVAL:]
log(f"train {len(tr_e)}  held-out {len(ev_e)}")

# ============================== AV LoRA ==============================
pm = get_peft_model(hf, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.0,
                                   target_modules=TGT, task_type="CAUSAL_LM",
                                   use_rslora=True))
n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad) / 1e6
log(f"AV LoRA r={R} rsLoRA alpha={ALPHA}: {n_train:.1f}M trainable")

AV_TMPL = ("You are looking at a hidden activation from a transformer, passed in "
           "the marker.\n<concept>" + MARKER + "</concept>\nDescribe what it represents.")
PROMPT = tk.apply_chat_template([{"role": "user", "content": AV_TMPL}],
                                tokenize=False, add_generation_prompt=True)
PLEN = len(tk.encode(PROMPT, add_special_tokens=False))   # prompt is fixed -> constant length
log(f"prompt len {PLEN} toks (fixed)")

try:
    import bitsandbytes as bnb
    Adam = bnb.optim.AdamW8bit
except ImportError:
    Adam = torch.optim.AdamW
av_params = [p for p in pm.parameters() if p.requires_grad]
opt = Adam(av_params, lr=LR_PEAK, betas=(0.9, 0.95))


def build_batch(gs, es, idx):
    full = [PROMPT + es[i][:300] + tk.eos_token for i in idx]
    enc = tk(full, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV0)
    mask = torch.zeros_like(enc.input_ids[:, 1:], dtype=torch.float)
    mask[:, PLEN - 1:] = 1.0
    mask = mask * enc.attention_mask[:, 1:].float()
    return enc, mask, gs[idx].to(DEV0)


def loss_on(enc, mask, v):
    INJ["v"] = v
    try:
        out = pm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    finally:
        INJ["v"] = None
    logits = out.logits[:, :-1].float()
    tgt = enc.input_ids[:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                         reduction="none").reshape(tgt.shape)
    return (ce * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(micro):
    pm.eval()
    tot, nb = 0.0, 0
    for s in range(0, len(ev_e), micro):
        idx = list(range(s, min(s + micro, len(ev_e))))
        enc, mask, v = build_batch(ev_g, ev_e, idx)
        tot += loss_on(enc, mask, v).item()
        nb += 1
    pm.train()
    return tot / max(nb, 1)


def set_lr(step, total):
    if step < WARMUP:
        lr = LR_PEAK * step / WARMUP
    else:
        prog = (step - WARMUP) / max(1, total - WARMUP)
        lr = LR_END + 0.5 * (LR_PEAK - LR_END) * (1 + math.cos(math.pi * prog))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


# ---- pre-flight micro-batch probe (try 32 -> 16 -> 8, pick largest that fits) ----
pm.train()
micro = 8
for cand in (32, 16, 8):
    try:
        idx = list(range(cand))
        enc, mask, v = build_batch(tr_g, tr_e, idx)
        loss_on(enc, mask, v).backward()
        opt.zero_grad()
        torch.cuda.empty_cache()
        micro = cand
        log(f"micro-batch {cand} fits")
        break
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad()
        torch.cuda.empty_cache()
        log(f"micro {cand} OOM, trying smaller")
accum = max(1, EFF_BATCH // micro)
steps_per_epoch = len(tr_e) // (micro * accum)
total_steps = steps_per_epoch * EPOCHS
log(f"micro={micro} accum={accum} eff_batch={micro*accum} "
    f"steps/epoch={steps_per_epoch} total_steps={total_steps}")

rng = np.random.default_rng(0)
best, since_best, opt_step = 1e9, 0, 0
pm.train()
base_nll = evaluate(micro)
log(f"baseline held-out NLL (untrained AV) = {base_nll:.4f}")

stop = False
for ep in range(EPOCHS):
    if stop:
        break
    perm = rng.permutation(len(tr_e))
    ptr = 0
    while ptr + micro * accum <= len(perm):
        lr = set_lr(opt_step, total_steps)
        opt.zero_grad()
        agg = 0.0
        ts = time.time()
        for _ in range(accum):
            idx = perm[ptr:ptr + micro].tolist()
            ptr += micro
            enc, mask, v = build_batch(tr_g, tr_e, idx)
            l = loss_on(enc, mask, v) / accum
            l.backward()
            agg += l.item()
        torch.nn.utils.clip_grad_norm_(av_params, 1.0)
        opt.step()
        opt_step += 1
        if opt_step % 20 == 0:
            log(f"ep{ep} step {opt_step}/{total_steps} | nll {agg:.4f} | "
                f"lr {lr:.2e} | {time.time()-ts:.1f}s")
        if opt_step % EVAL_EVERY == 0:
            ev = evaluate(micro)
            tag = ""
            if ev < best:
                best, since_best = ev, 0
                pm.save_pretrained(SAVE_AV)
                json.dump({"marker": MARKER, "marker_id": MARKER_ID, "k_layer": K_LAYER,
                           "mse_scale": MSE_SCALE, "mean_path": CK + "_mean.npy",
                           "av_tmpl": AV_TMPL, "r": R, "alpha": ALPHA,
                           "best_nll": best, "opt_step": opt_step},
                          open(SAVE_AV + "/av_meta.json", "w"))
                tag = "  *** new best, saved"
            else:
                since_best += 1
                tag = f"  (no improve {since_best}/{PATIENCE})"
            log(f"  >>> held-out NLL {ev:.4f} (best {best:.4f}){tag}")
            if since_best >= PATIENCE:
                log(f"early stop: no held-out improvement for {PATIENCE} evals")
                stop = True
                break
    if not stop:
        ev = evaluate(micro)
        log(f"== epoch {ep} done | held-out NLL {ev:.4f} (best {best:.4f}) ==")

log(f"DONE. best held-out NLL {best:.4f} (baseline {base_nll:.4f}) -> {SAVE_AV}")
