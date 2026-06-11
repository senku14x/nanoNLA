"""Diagnose the AV: is the injection actually conditioning the output?
Generate on maximally-diverse real activations + ZERO + RANDOM. If all outputs
are identical, the injection isn't working / AV ignores it."""
import json
import numpy as np
import torch

import gptqmodel.nn_modules.qlinear.gemm_awq as _gawq
if not hasattr(_gawq, "AwqGEMMQuantLinear"):
    _gawq.AwqGEMMQuantLinear = type("AwqGEMMQuantLinear", (), {})
from gptqmodel import GPTQModel, BACKEND
from transformers import AutoTokenizer
from peft import PeftModel


def log(*a):
    print(*a, flush=True)


M = "/workspace/models/qwen3.5-397b-gptq-int4"
CK = "/workspace/nla-ckpts/qwen3.5-397b-nla247k"
MARKER = "☴"

gm = GPTQModel.load(M, backend=BACKEND.TORCH, device_map="auto")
hf = gm.model
log("model loaded")
dec = hf
for a in ("model", "language_model"):
    if hasattr(dec, a):
        dec = getattr(dec, a)
tk = AutoTokenizer.from_pretrained(M)
tk.padding_side = "left"
if tk.pad_token is None:
    tk.pad_token = tk.eos_token
MARKER_ID = tk.encode(MARKER, add_special_tokens=False)[0]

INJ = {"v": None, "ids": None}
HOOK_FIRED = {"n": 0}


def emb_hook(m, a, k, o):
    ids = k.get("input") if k else None
    if ids is None and a:
        ids = a[0]
    INJ["ids"] = ids
    return o


def layer_hook(m, a, o):
    resid = o[0] if isinstance(o, tuple) else o
    v, ids = INJ["v"], INJ["ids"]
    if v is None or ids is None or resid.shape[1] < 2:
        return o
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
        if vv.norm() < 1e-6:
            HOOK_FIRED["n"] += 1
            continue  # ZERO vector -> no injection
        out[b, p] = (hp + hn * vv / vv.norm()).to(out.dtype)
        HOOK_FIRED["n"] += 1
    return (out,) + tuple(o[1:]) if isinstance(o, tuple) else out


hf.get_input_embeddings().register_forward_hook(emb_hook, with_kwargs=True)
dec.layers[1].register_forward_hook(layer_hook)

pm = PeftModel.from_pretrained(hf, CK + "_av_lora")
pm.eval()
log("AV adapter loaded")

golds = np.load(CK + "_golds_full.npy")
mean = np.load(CK + "_mean.npy")
expls = json.load(open(CK + "_expls_full.json"))
gc = (golds - mean).astype(np.float32)

# greedily pick 6 maximally-diverse training activations
S = gc[:5000]
Sn = S / np.clip(np.linalg.norm(S, axis=1, keepdims=True), 1e-9, None)
chosen = [0]
for _ in range(5):
    mx = np.abs(Sn @ Sn[chosen].T).max(axis=1)
    mx[chosen] = 9
    chosen.append(int(mx.argmin()))
log("chosen diverse idx:", chosen, "| their mutual |cos|:")
log(np.round(np.abs(Sn[chosen] @ Sn[chosen].T), 2))

norm_typ = float(np.linalg.norm(gc[:1000], axis=1).mean())
vecs, labels = [], []
for c in chosen:
    vecs.append(gc[c]); labels.append(f"REAL#{c}: {expls[c][:70].strip()}")
vecs.append(np.zeros(4096, np.float32)); labels.append("ZERO (no injection)")
rr = np.random.RandomState(0).randn(4096).astype(np.float32); rr = rr / np.linalg.norm(rr) * norm_typ
vecs.append(rr); labels.append("RANDOM direction")
vecs.append(gc[chosen[0]]); labels.append(f"REAL#{chosen[0]} REPEAT (sanity=match first)")
V = torch.tensor(np.stack(vecs), dtype=torch.float32)

AV_TMPL = ("You are looking at a hidden activation from a transformer, passed in "
           "the marker.\n<concept>" + MARKER + "</concept>\nDescribe what it represents.")
PROMPT = tk.apply_chat_template([{"role": "user", "content": AV_TMPL}],
                                tokenize=False, add_generation_prompt=True)
enc = tk([PROMPT] * len(V), return_tensors="pt", add_special_tokens=False, padding=True).to("cuda:0")
INJ["v"] = V.to("cuda:0")
with torch.no_grad():
    out = pm.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                      max_new_tokens=64, do_sample=False, repetition_penalty=1.3,
                      pad_token_id=tk.pad_token_id)
INJ["v"] = None
log(f"\n[hook fired {HOOK_FIRED['n']} times during generation]")
log("\n================ INJECTION DIAGNOSTIC ================")
for j, lab in enumerate(labels):
    g = tk.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    log(f"\n[{lab}]\n  -> {g}")
log("\n================ DONE ================")
