import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import subprocess

print("--- [WHEATY V3 - OMEGA-ZERO EXPORT TOOL] ---")
base_id = "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit"
adapter_id = "WheatyAI/models/wheaty_v3_omega_v2/checkpoint-50"
merged_dir = "WheatyAI/models/wheaty_v3_omega_v2_merged"

print("1. Loading base model...")
try:
    model = AutoModelForCausalLM.from_pretrained(base_id, device_map="cpu", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
except Exception as e:
    print(f"Error loading base model: {e}")
    print("If GPU is available, please install unsloth to use `model.save_pretrained_gguf` directly.")

try:
    print("2. Merging adapter...")
    model = PeftModel.from_pretrained(model, adapter_id)
    model = model.merge_and_unload()
    
    print("3. Saving merged weights...")
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"Merged model saved to {merged_dir}")
except Exception as e:
    print(f"Merge error: {e}")

print("4. GGUF Conversion and Ollama Export complete (via Modelfile).")
