"""Data-parallel AV-SFT for 397B (3 ranks x 2-GPU model copies).

Each rank loads its own model copy across a GPU pair (naive-MP), trains on a
stride-shard of the data, and the ranks all-reduce ONLY the 182M LoRA grads each
opt-step (gloo/CPU — robust vs NCCL device-pinning under restricted
CUDA_VISIBLE_DEVICES; the 220GB frozen base is never synced). Weights stay
identical via (a) broadcast of rank-0 LoRA init and (b) averaged grads + identical
optimizer steps. Lockstep loop (fixed #opt-steps, barriers around eval) => no hangs.

Launch: run_av_sft_dp.sh sets CUDA_VISIBLE_DEVICES/RANK/WORLD/MASTER_* per proc.
Reads the merged 247k cache from datagen-DP (_golds_full / _expls_full / _mean).
"""
import os, json, time, math
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

import gptqmodel.nn_modules.qlinear.gemm_awq as _gawq
if not hasattr(_gawq, "AwqGEMMQuantLinear"):
    _gawq.AwqGEMMQuantLinear = type("AwqGEMMQuantLinear", (), {})
from gptqmodel import GPTQModel, BACKEND
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD"])
IS_MAIN = RANK == 0

M = "/workspace/models/qwen3.5-397b-gptq-int4"
CK = "/workspace/nla-ckpts/qwen3.5-397b-nla247k"
GOLDS = CK + "_golds_full.npy"
EXPLS = CK + "_expls_full.json"
DOCS = CK + "_doc_ids_full.json"
MEAN = CK + "_mean.npy"
SAVE_AV = CK + "_av_lora"
MARKER = "☴"
K_LAYER = 40
MSE_SCALE = 64.0
TGT = ["q_proj", "k_proj", "v_proj", "o_proj",
       "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]
N_EVAL = 2000
R, ALPHA = 128, 16
GLOBAL_BATCH = 256              # paper; per-rank ~ GLOBAL_BATCH/WORLD
LR_PEAK, LR_END, WARMUP = 3e-5, 3e-6, 100
EPOCHS = 1
EVAL_EVERY = 100
PATIENCE = 4
SEED = 0
t0 = time.time()


def log(*a):
    if IS_MAIN:
        print(f"[+{int(time.time()-t0):6d}s]", *a, flush=True)


def logr(*a):
    print(f"[rank{RANK} +{int(time.time()-t0):6d}s]", *a, flush=True)


# ---- load model FIRST, then dist-init. The launcher loads ranks SEQUENTIALLY
# (waits for each "model loaded") because 3 concurrent gptqmodel loads die on this
# box at the kernel-select->weight-load transition. dist.init after load => ranks
# sync only once all have loaded. ----
logr(f"loading model; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} n_gpu={torch.cuda.device_count()}")
gm = GPTQModel.load(M, backend=BACKEND.TORCH, device_map="auto")
hf = gm.model
logr("model loaded")
dist.init_process_group("gloo", rank=RANK, world_size=WORLD,
                        timeout=__import__("datetime").timedelta(minutes=90))
logr("dist up")
dec = hf
for a in ("model", "language_model"):
    if hasattr(dec, a):
        dec = getattr(dec, a)
tk = AutoTokenizer.from_pretrained(M)
tk.padding_side = "right"
if tk.pad_token is None:
    tk.pad_token = tk.eos_token
mid = tk.encode(MARKER, add_special_tokens=False)
assert len(mid) == 1
MARKER_ID = mid[0]

# ---- injection hook (per-process) ----
INJ = {"v": None, "ids": None}


def emb_hook(module, a, k, output):
    ids = k.get("input") if k else None
    if ids is None and a:
        ids = a[0]
    INJ["ids"] = ids
    return output


def layer_hook(module, a, output):
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

# ---- data (merged 247k cache) ----
golds = np.load(GOLDS)
expls = json.load(open(EXPLS))
mean = np.load(MEAN)
golds = golds - mean
golds = torch.tensor(golds, dtype=torch.float32)
# doc-disjoint eval split: eval = last N_EVAL rows; drop train rows sharing a doc
# (deterministic + identical on all ranks: same files, same filter)
docs = json.load(open(DOCS)) if os.path.exists(DOCS) else None
if docs is not None and len(docs) == len(expls):
    ev_doc_set = set(docs[-N_EVAL:])
    keep = [i for i in range(len(expls) - N_EVAL) if docs[i] not in ev_doc_set]
    n_leak = (len(expls) - N_EVAL) - len(keep)
    log(f"doc-disjoint split: dropped {n_leak} train rows sharing docs with eval")
    tr_g, ev_g = golds[keep], golds[-N_EVAL:]
    tr_e, ev_e = [expls[i] for i in keep], expls[-N_EVAL:]
else:
    log("[WARN] no doc_ids — eval split is doc-leaky")
    tr_g, ev_g = golds[:-N_EVAL], golds[-N_EVAL:]
    tr_e, ev_e = expls[:-N_EVAL], expls[-N_EVAL:]
n_train = len(tr_e)
log(f"train {n_train}  held-out {len(ev_e)}")

# ---- AV LoRA (identical init across ranks via broadcast) ----
torch.manual_seed(SEED)
pm = get_peft_model(hf, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.0,
                                   target_modules=TGT, task_type="CAUSAL_LM",
                                   use_rslora=True))
av_params = [p for p in pm.parameters() if p.requires_grad]
log(f"AV LoRA {sum(p.numel() for p in av_params)/1e6:.1f}M trainable")

# broadcast rank-0 LoRA weights to all ranks (robust identical start)
for p in av_params:
    c = p.detach().cpu()
    dist.broadcast(c, src=0)
    p.data.copy_(c.to(p.device))
dist.barrier()
logr("LoRA weights synced from rank0")

AV_TMPL = ("You are looking at a hidden activation from a transformer, passed in "
           "the marker.\n<concept>" + MARKER + "</concept>\nDescribe what it represents.")
PROMPT = tk.apply_chat_template([{"role": "user", "content": AV_TMPL}],
                                tokenize=False, add_generation_prompt=True)
PLEN = len(tk.encode(PROMPT, add_special_tokens=False))

try:
    import bitsandbytes as bnb
    Adam = bnb.optim.AdamW8bit
except ImportError:
    Adam = torch.optim.AdamW
opt = Adam(av_params, lr=LR_PEAK, betas=(0.9, 0.95))


def build_batch(gs, es, idx):
    full = [PROMPT + es[i][:300] + tk.eos_token for i in idx]
    enc = tk(full, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
    mask = torch.zeros_like(enc.input_ids[:, 1:], dtype=torch.float)
    mask[:, PLEN - 1:] = 1.0
    mask = mask * enc.attention_mask[:, 1:].float()
    return enc, mask, gs[idx].to("cuda:0")


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


def allreduce_lora_grads():
    # flatten grads -> one CPU buffer -> gloo all-reduce (avg) -> scatter back
    gs = [p.grad for p in av_params if p.grad is not None]
    flat = torch.cat([g.detach().flatten().float().cpu() for g in gs])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= WORLD
    off = 0
    for p in av_params:
        if p.grad is not None:
            n = p.grad.numel()
            p.grad.copy_(flat[off:off + n].view_as(p.grad).to(p.grad.device))
            off += n


def set_lr(step, total):
    if step < WARMUP:
        lr = LR_PEAK * step / WARMUP
    else:
        prog = (step - WARMUP) / max(1, total - WARMUP)
        lr = LR_END + 0.5 * (LR_PEAK - LR_END) * (1 + math.cos(math.pi * prog))
    for g in opt.param_groups:
        g["lr"] = lr
    return lr


# ---- micro-batch probe (all ranks; take the MIN that fits across ranks) ----
# catch BOTH torch.cuda.OutOfMemoryError AND Triton's RuntimeError("...out of memory")
# (FLA Gated-DeltaNet OOMs via Triton, not the torch OOM type).
pm.train()
micro = 4
for cand in (16, 8, 4):
    try:
        idx = list(range(cand))
        enc, mask, v = build_batch(tr_g, tr_e, idx)
        loss_on(enc, mask, v).backward()
        opt.zero_grad(); torch.cuda.empty_cache()
        micro = cand
        logr(f"micro-batch {cand} fits")
        break
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        opt.zero_grad(); torch.cuda.empty_cache()
        logr(f"micro {cand} OOM, trying smaller")
mt = torch.tensor([micro])
dist.all_reduce(mt, op=dist.ReduceOp.MIN)   # all ranks use the same micro
micro = int(mt.item())
per_rank = max(micro, GLOBAL_BATCH // WORLD)
accum = max(1, round(per_rank / micro))
steps_per_epoch = (n_train // WORLD) // (micro * accum)
total_steps = steps_per_epoch * EPOCHS
log(f"micro={micro} accum={accum} per_rank_batch={micro*accum} global={micro*accum*WORLD} "
    f"steps/epoch={steps_per_epoch} total={total_steps}")

best, since_best = 1e9, 0
pm.train()
if IS_MAIN:
    log(f"baseline held-out NLL {evaluate(micro):.4f}")
dist.barrier()

opt_step = 0
for ep in range(EPOCHS):
    g = torch.Generator().manual_seed(1000 + ep)           # SAME perm on all ranks
    perm = torch.randperm(n_train, generator=g).tolist()
    mine = perm[RANK::WORLD]                                 # this rank's stride-shard
    ptr = 0
    for _ in range(steps_per_epoch):
        lr = set_lr(opt_step, total_steps)
        opt.zero_grad()
        agg = 0.0
        ts = time.time()
        for _a in range(accum):
            idx = mine[ptr:ptr + micro]; ptr += micro
            enc, mask, v = build_batch(tr_g, tr_e, idx)
            l = loss_on(enc, mask, v) / accum
            l.backward()
            agg += l.item()
        allreduce_lora_grads()
        torch.nn.utils.clip_grad_norm_(av_params, 1.0)
        opt.step()
        opt_step += 1
        if IS_MAIN and opt_step % 20 == 0:
            log(f"ep{ep} step {opt_step}/{total_steps} | nll {agg:.4f} | lr {lr:.2e} | {time.time()-ts:.1f}s")
        if opt_step % EVAL_EVERY == 0:
            dist.barrier()
            if IS_MAIN:
                ev = evaluate(micro)
                tag = ""
                if ev < best:
                    best, since_best = ev, 0
                    pm.save_pretrained(SAVE_AV)
                    json.dump({"marker": MARKER, "marker_id": MARKER_ID, "k_layer": K_LAYER,
                               "mse_scale": MSE_SCALE, "mean_path": MEAN, "av_tmpl": AV_TMPL,
                               "r": R, "alpha": ALPHA, "best_nll": best, "opt_step": opt_step},
                              open(SAVE_AV + "/av_meta.json", "w"))
                    tag = "  *** new best, saved"
                else:
                    since_best += 1
                    tag = f"  (no improve {since_best}/{PATIENCE})"
                log(f"  >>> held-out NLL {ev:.4f} (best {best:.4f}){tag}")
            # broadcast stop flag so all ranks exit together
            stopf = torch.tensor([1 if (IS_MAIN and since_best >= PATIENCE) else 0])
            dist.broadcast(stopf, src=0)
            dist.barrier()
            if int(stopf.item()):
                log("early stop")
                break
    else:
        continue
    break

if IS_MAIN:
    log(f"DONE. best held-out NLL {best:.4f} -> {SAVE_AV}")
dist.barrier()
dist.destroy_process_group()
