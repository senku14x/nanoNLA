"""Isolate: is the no-compile EAGER 397B generation itself broken?
Generate from the BASE model (no adapter, no injection) with simple prompts.
If garbage -> eager/no-compile is the bug. If coherent -> the AV-SFT is."""
import numpy as np
import torch

import gptqmodel.nn_modules.qlinear.gemm_awq as _gawq
if not hasattr(_gawq, "AwqGEMMQuantLinear"):
    _gawq.AwqGEMMQuantLinear = type("AwqGEMMQuantLinear", (), {})
from gptqmodel import GPTQModel, BACKEND
from transformers import AutoTokenizer


def log(*a):
    print(*a, flush=True)


M = "/workspace/models/qwen3.5-397b-gptq-int4"
gm = GPTQModel.load(M, backend=BACKEND.TORCH, device_map="auto")
hf = gm.model
log("model loaded")
tk = AutoTokenizer.from_pretrained(M)
tk.padding_side = "left"
if tk.pad_token is None:
    tk.pad_token = tk.eos_token

chats = [
    "What is the capital of France? Answer in one word.",
    "Write one sentence about the ocean.",
    "Briefly: what is 2+2?",
]
prompts = [tk.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                                  add_generation_prompt=True) for c in chats]
enc = tk(prompts, return_tensors="pt", add_special_tokens=False, padding=True).to("cuda:0")
with torch.no_grad():
    out = hf.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                      max_new_tokens=40, do_sample=False, pad_token_id=tk.pad_token_id)
log("\n================ BASE MODEL (no adapter, eager) ================")
for j, c in enumerate(chats):
    g = tk.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    log(f"\n[Q: {c}]\n  -> {g}")
log("\n================ DONE ================")
