"""397B NLA RL (GRPO) — warmstart from the 6k SFT adapters.

Actor  = 397B + AV-LoRA ("policy", trainable) + frozen "ref" adapter (KL anchor).
Critic = AR-LoRA ("critic", co-trained) + value_head. Reward = -reconstruction MSE.
All adapters share one resident 397B base. Rollouts are BATCHED (B*G in one
generate) so decode latency amortizes. Paper-faithful: many prompts / few
rollouts, kl 0.01, reward = -mse_nrm. Necessarily scaled (generation-bound).
"""
import json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gptqmodel.nn_modules.qlinear.gemm_awq as _gawq
if not hasattr(_gawq, "AwqGEMMQuantLinear"):
    _gawq.AwqGEMMQuantLinear = type("AwqGEMMQuantLinear", (), {})

from gptqmodel import GPTQModel, BACKEND
from transformers import AutoTokenizer
from peft import PeftModel
from safetensors.torch import load_file

M = "/workspace/models/qwen3.5-397b-gptq-int4"
CK = "/workspace/nla-ckpts/qwen3.5-397b-nla6k"
AV_ADAPTER = CK + "_av_lora/av"
AR_ADAPTER = CK + "_ar_lora"
MARKER = "☴"
K_LAYER = 40
MSE_SCALE = 64.0
DEV0 = "cuda:0"
B_PROMPTS = 4
G_ROLLOUTS = 4
MAX_NEW = 48
N_STEPS = 12
KL_BETA = 0.01
CLIP = 0.2
TEMP = 1.0
POLICY_LR = 1e-5
CRITIC_LR = 1e-4
t_start = time.time()


def log(*a):
    print(f"[+{int(time.time()-t_start):5d}s]", *a, flush=True)


def normalize(v, scale=MSE_SCALE):
    n = v.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return v.float() / n * scale


# ---- load base + tokenizer + marker ----
t = time.time()
gm = GPTQModel.load(M, backend=BACKEND.TORCH, device_map="auto")
hf = gm.model
log(f"loaded {int(time.time()-t)}s")
dec = hf
for a in ("model", "language_model"):
    if hasattr(dec, a):
        dec = getattr(dec, a)
tk = AutoTokenizer.from_pretrained(M)
tk.padding_side = "left"   # generation wants left-pad; we recompute logprobs by mask anyway
if tk.pad_token is None:
    tk.pad_token = tk.eos_token
MARKER_ID = tk.encode(MARKER, add_special_tokens=False)[0]

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
        vv = v[b].to(resid.device).float()
        out[b, p] = (hp + hp.norm().clamp(min=1e-8) * vv / vv.norm().clamp(min=1e-8)).to(out.dtype)
    return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out


hf.get_input_embeddings().register_forward_hook(emb_hook, with_kwargs=True)
dec.layers[1].register_forward_hook(layer_hook)

# ---- adapters: policy (AV trainable) + ref (AV frozen) + critic (AR) + value_head ----
actor = PeftModel.from_pretrained(hf, AV_ADAPTER, adapter_name="policy", is_trainable=True)
actor.load_adapter(AV_ADAPTER, adapter_name="ref")
actor.load_adapter(AR_ADAPTER, adapter_name="critic")
value_head = nn.Linear(4096, 4096, bias=False).to(DEV0).to(torch.bfloat16)
value_head.load_state_dict(torch.load(CK + "_ar_valuehead.pt", map_location=DEV0))
for p in value_head.parameters():
    p.requires_grad_(True)
actor.set_adapter("policy")     # policy active for the first rollout
log("adapters loaded: policy/ref/critic + value_head")

# centered activations (the RL "prompts") + the per-model mean
golds_raw = np.load(CK + "_golds.npy")
MEAN = np.load(CK + "_mean.npy")
MEAN_T = torch.tensor(MEAN, dtype=torch.float32).to(DEV0)
golds = golds_raw - MEAN                       # centered targets / injections
egt = torch.tensor(golds)
BASE = ((normalize(egt) - normalize(egt.mean(0, keepdim=True))) ** 2).mean(-1).mean().item()
log(f"{len(golds)} centered acts | baseline mse_nrm={BASE:.4f}")

AV_TMPL = ("You are looking at a hidden activation from a transformer, passed in "
           "the marker.\n<concept>" + MARKER + "</concept>\nDescribe what it represents.")
AR_TMPL = ("You are looking at a hidden activation from a transformer.\n"
           "Reasoning the model represents: {expl}\n<summary>")

try:
    import bitsandbytes as bnb
    Adam = bnb.optim.AdamW8bit
except ImportError:
    Adam = torch.optim.AdamW
# select by NAME, not requires_grad: set_adapter toggles requires_grad per phase,
# but the optimizer should hold all params (Adam skips those with grad=None).
policy_params = [p for n, p in actor.named_parameters() if "lora_" in n and "policy" in n]
critic_params = [p for n, p in actor.named_parameters() if "lora_" in n and "critic" in n] + list(value_head.parameters())
log(f"policy={sum(p.numel() for p in policy_params)/1e6:.1f}M ({len(policy_params)}t)  "
    f"critic={sum(p.numel() for p in critic_params)/1e6:.1f}M ({len(critic_params)}t)")
if not policy_params or len(critic_params) <= 1:
    log("DEBUG lora names:", [n for n, p in actor.named_parameters() if "lora_" in n][:6])
    raise SystemExit("empty param list")
popt = Adam(policy_params, lr=POLICY_LR, betas=(0.9, 0.95))
copt = Adam(critic_params, lr=CRITIC_LR, betas=(0.9, 0.95))


def ar_reconstruct(explanations, golds_batch):
    """Critic: explanation text -> predicted (centered) activation; return reward=-mse."""
    actor.set_adapter("critic")
    prompts = [tk.apply_chat_template([{"role": "user", "content": AR_TMPL.format(expl=e[:500])}],
                                      tokenize=False, add_generation_prompt=True) for e in explanations]
    enc = tk(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV0)
    INJ["v"] = None
    out = actor(input_ids=enc.input_ids, attention_mask=enc.attention_mask, output_hidden_states=True, use_cache=False)
    # left-pad => last real token is the final column
    h = out.hidden_states[K_LAYER][:, -1, :].to(DEV0).float() - MEAN_T
    pred = value_head(normalize(h).to(torch.bfloat16)).float()
    gold = torch.tensor(np.stack(golds_batch)).to(DEV0)
    mse = ((normalize(pred) - normalize(gold)) ** 2).mean(-1)
    return (-mse).detach(), enc, pred, gold


rng = np.random.default_rng(0)
fve_hist = []
for step in range(N_STEPS):
    ts = time.time()
    idx = rng.choice(len(golds), B_PROMPTS, replace=False)
    # ---- batched rollout: B*G sequences, each prompt's activation injected ----
    actor.set_adapter("policy")
    av_prompt = tk.apply_chat_template([{"role": "user", "content": AV_TMPL}], tokenize=False, add_generation_prompt=True)
    flat_idx = np.repeat(idx, G_ROLLOUTS)
    enc = tk([av_prompt] * (B_PROMPTS * G_ROLLOUTS), return_tensors="pt", padding=True, add_special_tokens=False).to(DEV0)
    INJ["v"] = torch.tensor(np.stack([golds[i] for i in flat_idx])).to(DEV0)
    try:
        with torch.no_grad():
            gen = actor.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                                 max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP, top_p=1.0, top_k=0,
                                 pad_token_id=tk.eos_token_id, return_dict_in_generate=True, output_logits=True)
    finally:
        INJ["v"] = None
    plen = enc.input_ids.shape[1]
    seqs = gen.sequences
    resp_ids = seqs[:, plen:]
    texts = tk.batch_decode(resp_ids, skip_special_tokens=True)
    # old logprobs (per response token) from generate's raw logits
    old_lp = []
    for t_, lg in enumerate(gen.logits):
        old_lp.append(F.log_softmax(lg.float() / max(TEMP, 1e-6), -1).gather(-1, resp_ids[:, t_:t_+1]).squeeze(-1))
    old_lp = torch.stack(old_lp, 1)  # [B*G, T]
    gen_t = time.time() - ts

    # ---- reward via critic ----
    rew, _, _, _ = ar_reconstruct(texts, [golds[i] for i in flat_idx])
    fve = 100 * (1 - (-rew.mean().item()) / BASE)
    fve_hist.append(fve)

    # ---- GRPO advantage (per-prompt group baseline) ----
    rew_g = rew.view(B_PROMPTS, G_ROLLOUTS)
    adv = (rew_g - rew_g.mean(1, keepdim=True)).view(-1)

    # ---- policy update: new + ref logprobs on the ACTUAL generated tokens ----
    # All rollout prompts are identical -> no prompt padding -> response = seqs[:, plen:].
    # generate() pads finished rows with eos (pad_token_id=eos_token_id), NOT
    # tk.pad_token_id, so mask from the FIRST eos (inclusive) — post-eos
    # positions were never sampled and must not enter surrogate/KL/attention.
    resp = seqs[:, plen:]                                            # [N, gen_len]
    is_eos = resp == tk.eos_token_id
    cum = is_eos.cumsum(1)
    rmask = ((cum == 0) | (is_eos & (cum == 1))).float()             # keep up to AND INCLUDING first eos
    am_full = torch.cat([torch.ones_like(seqs[:, :plen]), rmask.long()], dim=1)
    INJ["v"] = torch.tensor(np.stack([golds[i] for i in flat_idx])).to(DEV0)
    actor.set_adapter("policy")
    out_p = actor(input_ids=seqs, attention_mask=am_full, use_cache=False)
    with torch.no_grad():
        actor.set_adapter("ref")
        out_r = actor(input_ids=seqs, attention_mask=am_full, use_cache=False)
    actor.set_adapter("policy")
    INJ["v"] = None
    lp_p = F.log_softmax(out_p.logits[:, plen - 1:-1].float() / max(TEMP, 1e-6), -1)
    lp_r = F.log_softmax(out_r.logits[:, plen - 1:-1].float(), -1)
    g_new = lp_p.gather(-1, resp.unsqueeze(-1)).squeeze(-1)          # [N, gen_len]
    g_ref = lp_r.gather(-1, resp.unsqueeze(-1)).squeeze(-1)
    old = old_lp[:, :resp.shape[1]].to(DEV0)
    ratio = torch.exp((g_new - old).clamp(-20, 20))
    adv_e = adv.unsqueeze(1)
    surr = torch.minimum(ratio * adv_e, torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_e)
    kl_t = torch.exp((g_ref - g_new).clamp(-20, 20)) - (g_ref - g_new) - 1
    per_tok = -(surr - KL_BETA * kl_t)
    denom = rmask.sum().clamp(min=1)
    ploss = (per_tok * rmask).sum() / denom
    kl_mean = (kl_t * rmask).sum().item() / denom.item()
    popt.zero_grad(); ploss.backward()
    torch.nn.utils.clip_grad_norm_(policy_params, 1.0); popt.step()

    # ---- co-train critic (MSE on the rollouts) ----
    actor.set_adapter("critic")
    _, _, pred_c, gold_c = ar_reconstruct(texts, [golds[i] for i in flat_idx])
    closs = F.mse_loss(normalize(pred_c), normalize(gold_c))
    copt.zero_grad(); closs.backward()
    torch.nn.utils.clip_grad_norm_(critic_params, 1.0); copt.step()
    actor.set_adapter("policy")

    log(f"RL step {step:02d} | FVE {fve:5.1f}% | reward {rew.mean().item():.3f} | ploss {ploss.item():.4f} "
        f"| kl {kl_mean:.4f} | crit {closs.item():.4f} | gen {gen_t:.0f}s tot {time.time()-ts:.0f}s")
    log(f"   sample: {texts[0][:90]!r}")

log(f"RL DONE. FVE {fve_hist[0]:.1f}% -> {fve_hist[-1]:.1f}% (best {max(fve_hist):.1f}%)")
actor.save_pretrained(CK + "_rl_policy", selected_adapters=["policy"])
log("policy saved. ALL DONE.")
