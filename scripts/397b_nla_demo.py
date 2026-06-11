"""Scaled 397B NLA demonstration on Qwen3.5-397B-A17B (GPTQ via gptqmodel).

One resident load (~24 min), then: Stage-0 datagen -> AR-SFT (FVE) -> AV-SFT (NLL).
RL is skipped (generation ~1.5 s/tok under naive MP makes rollouts impractical).

Reuses the 8B dataset's (context, explanation) pairs and re-extracts 397B
activations at hidden_states[K] (mid-depth) -- so no Sonnet re-labelling.

Key 397B-specific bits:
  - gptqmodel TORCH backend + peft<->gptqmodel import shim (AwqGEMMQuantLinear)
  - decoder at hf.model.language_model.layers (multimodal wrapper)
  - injection marker = U+2634 (single-token in Qwen3.5's 248k vocab)
  - FLA installed so the 45 Gated-DeltaNet layers don't crawl on torch fallback
"""
import json, time
import numpy as np
import torch
import torch.nn as nn
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
SAVE = "/workspace/nla-ckpts/qwen3.5-397b-nla6k"
MARKER = "☴"          # ☴ single-token marker
K_LAYER = 40              # capture hidden_states[40] of 60 (~0.67 depth, matches 8B L24/36)
MSE_SCALE = 64.0         # sqrt(d_model=4096)
TGT = ["q_proj", "k_proj", "v_proj", "o_proj",
       "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
N_DATA = 6000
N_EVAL = 200
MAXLEN = 256             # context token cap; LONGER pairs are DROPPED (truncating would extract
                         # the activation mid-context while the explanation describes the full context)
AR_STEPS = 300
AV_STEPS = 150
BATCH = 16
AV_BATCH = 8
DATAGEN_BATCH = 16
DEV0 = "cuda:0"
t_start = time.time()


def log(*a):
    print(f"[+{int(time.time()-t_start):5d}s]", *a, flush=True)


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
tk.padding_side = "right"   # last-token capture + response masking assume right pad
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

# ============================== datagen ==============================
log("=== Stage-0 datagen: 397B activations at hidden[%d] ===" % K_LAYER)
import os
if os.path.exists(SAVE + "_golds.npy") and os.path.exists(SAVE + "_expls.json"):
    golds = np.load(SAVE + "_golds.npy")
    expls = json.load(open(SAVE + "_expls.json"))
    docs = json.load(open(SAVE + "_doc_ids.json")) if os.path.exists(SAVE + "_doc_ids.json") else None
    log(f"loaded cached datagen: {golds.shape}")
else:
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
    log(f"{len(pairs)} (context, explanation) pairs from 8B dataset")
    hf.eval()
    golds, expls, docs = [], [], []
    t0 = time.time()
    for s in range(0, len(pairs), DATAGEN_BATCH):  # BATCHED datagen (~12x faster than batch-1)
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
        if s % (DATAGEN_BATCH * 20) == 0:
            log(f"  datagen {s}/{len(pairs)}  ({(time.time()-t0)/(s+1):.3f}s/doc)")
    golds = np.stack(golds)
    np.save(SAVE + "_golds.npy", golds)
    json.dump(expls, open(SAVE + "_expls.json", "w"))
    json.dump(docs, open(SAVE + "_doc_ids.json", "w"))
    log(f"datagen done: {golds.shape} in {int(time.time()-t0)}s")

# CENTER: by hidden[K] the residual stream is ~99% a near-constant attention-sink
# component (norm ~1.1M, all parallel -> baseline FVE=0). The input-specific
# signal is the ~1% that varies; NLA operates on that. Subtract the dataset mean
# everywhere (targets, injected vectors, and the AR's own constant-dominated hidden).
MEAN = golds.mean(0, keepdims=True)
np.save(SAVE + "_mean.npy", MEAN)
MEAN_T = torch.tensor(MEAN, dtype=torch.float32).to(DEV0)
golds = golds - MEAN
gt = torch.tensor(golds)
mu = gt.mean(0, keepdim=True)
BASE = ((normalize(gt) - normalize(mu)) ** 2).mean(-1).mean().item()
log(f"centered; predict-mean baseline mse_nrm = {BASE:.4f}")
# held-out split for an honest FVE (train on the rest), doc-disjoint:
# eval = last N_EVAL rows; drop train rows sharing a doc with eval
eval_golds = golds[-N_EVAL:].copy()
eval_expls = expls[-N_EVAL:]
if docs is not None and len(docs) == len(expls):
    ev_doc_set = set(docs[-N_EVAL:])
    keep = [i for i in range(len(expls) - N_EVAL) if docs[i] not in ev_doc_set]
    n_leak = (len(expls) - N_EVAL) - len(keep)
    log(f"doc-disjoint split: dropped {n_leak} train rows sharing docs with eval")
    golds = golds[keep]
    expls = [expls[i] for i in keep]
else:
    log("[WARN] no doc_ids — eval split is doc-leaky")
    golds = golds[:-N_EVAL]
    expls = expls[:-N_EVAL]
egt = torch.tensor(eval_golds)
EVAL_BASE = ((normalize(egt) - normalize(egt.mean(0, keepdim=True))) ** 2).mean(-1).mean().item()
log(f"train={len(expls)}  held-out eval={len(eval_expls)}  eval baseline mse_nrm={EVAL_BASE:.4f}")

# ============================== AR-SFT ==============================
log("=== AR-SFT: reconstruct 397B activation from explanation text ===")
AR_TMPL = ("You are looking at a hidden activation from a transformer.\n"
           "Reasoning the model represents: {expl}\n<summary>")
pm = get_peft_model(hf, LoraConfig(r=64, lora_alpha=16, lora_dropout=0.0,
                                   target_modules=TGT, task_type="CAUSAL_LM",
                                   use_rslora=True))
log(f"peft attached, LoRA trainable={sum(p.numel() for p in pm.parameters() if p.requires_grad)/1e6:.1f}M")
value_head = nn.Linear(4096, 4096, bias=False).to(DEV0)
with torch.no_grad():
    value_head.weight.copy_(torch.eye(4096))
value_head = value_head.to(torch.bfloat16)
try:
    import bitsandbytes as bnb
    Adam = bnb.optim.AdamW8bit
except ImportError:
    Adam = torch.optim.AdamW
ar_params = [p for p in pm.parameters() if p.requires_grad] + list(value_head.parameters())
opt = Adam(ar_params, lr=1e-4, betas=(0.9, 0.95))
rng = np.random.default_rng(0)
pm.train()


def ar_predict(expl_batch):
    prompts = []
    for e in expl_batch:
        msgs = [{"role": "user", "content": AR_TMPL.format(expl=e[:600])}]
        prompts.append(tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    enc = tk(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV0)
    INJ["v"] = None
    out = pm(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
             output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[K_LAYER].to(DEV0)         # device_map spreads layers; pull to GPU0
    last = (enc.attention_mask.sum(1) - 1).to(DEV0)
    h = hs[torch.arange(len(expl_batch), device=DEV0), last]
    h = h.float() - MEAN_T                            # center: AR hidden is equally constant-dominated
    h = normalize(h).to(torch.bfloat16)
    return value_head(h).float()


def ar_eval():
    pm.eval()
    preds = []
    with torch.no_grad():
        for j in range(0, len(eval_expls), 8):
            preds.append(ar_predict(eval_expls[j:j + 8]))
    pm.train()
    pred = torch.cat(preds)
    gold = torch.tensor(eval_golds).to(DEV0)
    return 100 * (1 - F.mse_loss(normalize(pred), normalize(gold)).item() / EVAL_BASE)

best_eval = -1e9
for step in range(AR_STEPS):
    ts = time.time()
    idx = rng.choice(len(expls), BATCH, replace=False)
    pred = ar_predict([expls[i] for i in idx])
    gold = torch.tensor(golds[idx]).to(DEV0)
    loss = F.mse_loss(normalize(pred), normalize(gold))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(ar_params, 1.0); opt.step()
    if (step + 1) % 50 == 0 or step == AR_STEPS - 1:
        ev = ar_eval(); best_eval = max(best_eval, ev)
        log(f"  AR step {step:03d} | train mse {loss.item():.4f} | HELD-OUT FVE {ev:5.1f}% (best {best_eval:.1f}%) | {time.time()-ts:.1f}s")
    else:
        log(f"  AR step {step:03d} | train mse {loss.item():.4f} | trainFVE {100*(1-loss.item()/BASE):5.1f}% | {time.time()-ts:.1f}s")
log(f"AR-SFT done. HELD-OUT FVE best={best_eval:.1f}%")
pm.save_pretrained(SAVE + "_ar_lora")
torch.save(value_head.state_dict(), SAVE + "_ar_valuehead.pt")

# ============================== AV-SFT ==============================
log("=== AV-SFT: verbalize the injected 397B activation ===")
pm.add_adapter("av", LoraConfig(r=64, lora_alpha=16, lora_dropout=0.0,
                                target_modules=TGT, task_type="CAUSAL_LM", use_rslora=True))
pm.set_adapter("av")
AV_TMPL = ("You are looking at a hidden activation from a transformer, passed in "
           "the marker.\n<concept>" + MARKER + "</concept>\nDescribe what it represents.")
av_params = [p for n, p in pm.named_parameters() if p.requires_grad and "av" in n]
opt2 = Adam(av_params, lr=1e-4, betas=(0.9, 0.95))
log(f"AV LoRA trainable={sum(p.numel() for p in av_params)/1e6:.1f}M")
for step in range(AV_STEPS):
    ts = time.time()
    idx = rng.choice(len(expls), AV_BATCH, replace=False)
    prompts, resps = [], []
    for i in idx:
        msgs = [{"role": "user", "content": AV_TMPL}]
        prompts.append(tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        resps.append(expls[i][:300] + tk.eos_token)
    full = [p + r for p, r in zip(prompts, resps)]
    enc = tk(full, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV0)
    plens = [len(tk.encode(p, add_special_tokens=False)) for p in prompts]
    INJ["v"] = torch.tensor(golds[idx]).to(DEV0)
    try:
        out = pm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    finally:
        INJ["v"] = None
    logits = out.logits[:, :-1].float()
    tgt = enc.input_ids[:, 1:]
    mask = torch.zeros_like(tgt, dtype=torch.float)
    for b, pl in enumerate(plens):
        mask[b, pl - 1:] = 1.0
    mask = mask * enc.attention_mask[:, 1:].float()
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
    loss = (ce * mask).sum() / mask.sum().clamp(min=1)
    opt2.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(av_params, 1.0); opt2.step()
    log(f"  AV step {step:03d} | nll {loss.item():.4f} | {time.time()-ts:.1f}s")
pm.save_pretrained(SAVE + "_av_lora", selected_adapters=["av"])
log("AV-SFT saved. ALL DONE.")
