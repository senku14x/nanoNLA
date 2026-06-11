"""De-risk bf16+FSDP for the Gated-DeltaNet arch: does FLA's linear-attention
backward survive FSDP wrapping? (The old FSDP-QLoRA crash was bnb-specific.)
Run: torchrun --nproc_per_node=2 fsdp_derisk.py"""
import os, functools, torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

dist.init_process_group("nccl")
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)


def log(*a):
    if lr == 0:
        print(*a, flush=True)


M = "/workspace/models/qwen3.5-2b"
tk = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16)
m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32,
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                 task_type="CAUSAL_LM"))
dec = m.base_model.model
for a in ("model", "language_model"):
    if hasattr(dec, a):
        dec = getattr(dec, a)
layer_cls = type(dec.layers[0])
log("layer class:", layer_cls.__name__, "| n layers:", len(dec.layers))
pol = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_cls})
m = FSDP(m, auto_wrap_policy=pol, device_id=lr, use_orig_params=True,
         mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
log("FSDP wrapped OK")

inp = tk("The quick brown fox jumps over the lazy dog and runs away fast.",
         return_tensors="pt").to(lr)
out = m(input_ids=inp.input_ids, labels=inp.input_ids)
out.loss.backward()
print(f"[rank{lr}] *** FLA + FSDP forward+backward SURVIVED, loss={out.loss.item():.3f} ***", flush=True)
dist.barrier()
if lr == 0:
    print("DONE", flush=True)
dist.destroy_process_group()
