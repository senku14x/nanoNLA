"""Reproduce the gptqmodel generation bug on a small GPTQ model (same arch family).
If gptqmodel generation garbles here too, we can fix it fast. Tests use_cache + a
manual full-forward greedy loop (matching the training forward exactly)."""
import sys
import torch

import gptqmodel.nn_modules.qlinear.gemm_awq as _gawq
if not hasattr(_gawq, "AwqGEMMQuantLinear"):
    _gawq.AwqGEMMQuantLinear = type("AwqGEMMQuantLinear", (), {})
from gptqmodel import GPTQModel, BACKEND
from transformers import AutoTokenizer

M = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/qwen3.5-35b-gptq"
gm = GPTQModel.load(M, backend=BACKEND.TORCH, device_map="auto")
hf = gm.model
print("model loaded", flush=True)
tk = AutoTokenizer.from_pretrained(M)
if tk.pad_token is None:
    tk.pad_token = tk.eos_token
pad = tk.pad_token_id

q = "What is the capital of France? Answer in one word."
p = tk.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
enc = tk(p, return_tensors="pt", add_special_tokens=False).to("cuda:0")

for uc in (True, False):
    with torch.no_grad():
        out = hf.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                          max_new_tokens=25, do_sample=False, use_cache=uc, pad_token_id=pad)
    g = tk.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n[generate use_cache={uc}] -> {g!r}", flush=True)

# manual greedy with full-forward each step (exactly the training forward path, use_cache=False)
ids = enc.input_ids.clone()
with torch.no_grad():
    for _ in range(25):
        o = hf(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        nxt = o.logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
        if nxt.item() == tk.eos_token_id:
            break
g = tk.decode(ids[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
print(f"\n[MANUAL full-forward greedy] -> {g!r}", flush=True)
print("\nDONE", flush=True)
