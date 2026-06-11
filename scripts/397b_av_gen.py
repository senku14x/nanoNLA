"""Eyeball the trained AV: generate explanations from HELD-OUT 397B activations.
Loads the AV LoRA, injects centered held-out golds at the marker, greedy-decodes."""
import json
import numpy as np
import torch

import gptqmodel.nn_modules.qlinear.gemm_awq as _gawq
if not hasattr(_gawq, "AwqGEMMQuantLinear"):
    _gawq.AwqGEMMQuantLinear = type("AwqGEMMQuantLinear", (), {})
from gptqmodel import GPTQModel, BACKEND
from transformers import AutoTokenizer
from peft import PeftModel

M = "/workspace/models/qwen3.5-397b-gptq-int4"
CK = "/workspace/nla-ckpts/qwen3.5-397b-nla247k"
AV_ADAPTER = CK + "_av_lora"
MARKER = "☴"
K_LAYER = 40
N_EVAL = 2000
N_SHOW = 10


def log(*a):
    print(*a, flush=True)


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
        out[b, p] = (hp + hn * vv / vv.norm().clamp(min=1e-8)).to(out.dtype)
    return (out,) + tuple(o[1:]) if isinstance(o, tuple) else out


hf.get_input_embeddings().register_forward_hook(emb_hook, with_kwargs=True)
dec.layers[1].register_forward_hook(layer_hook)

pm = PeftModel.from_pretrained(hf, AV_ADAPTER)
pm.eval()
log("AV adapter loaded")

golds = np.load(CK + "_golds_full.npy")
mean = np.load(CK + "_mean.npy")
golds = golds - mean
expls = json.load(open(CK + "_expls_full.json"))
ev_g = torch.tensor(golds[-N_EVAL:], dtype=torch.float32)   # held-out, never trained on
ev_e = expls[-N_EVAL:]

AV_TMPL = ("You are looking at a hidden activation from a transformer, passed in "
           "the marker.\n<concept>" + MARKER + "</concept>\nDescribe what it represents.")
PROMPT = tk.apply_chat_template([{"role": "user", "content": AV_TMPL}],
                                tokenize=False, add_generation_prompt=True)

idxs = list(range(N_SHOW))
enc = tk([PROMPT] * N_SHOW, return_tensors="pt", add_special_tokens=False, padding=True).to("cuda:0")
INJ["v"] = ev_g[idxs].to("cuda:0")
with torch.no_grad():
    out = pm.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                      max_new_tokens=64, do_sample=False, pad_token_id=tk.pad_token_id)
INJ["v"] = None
log("\n================ AV GENERATIONS (held-out activations) ================")
for j, i in enumerate(idxs):
    gen = tk.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    log(f"\n----- held-out #{i} -----")
    log(f"  GROUND-TRUTH (8B expl): {ev_e[i][:220]}")
    log(f"  AV SAYS:                {gen}")
log("\n================ DONE ================")
