import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_id = "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit"
adapter_id = "WheatyAI/models/wheaty_v3_omega_v2/checkpoint-50"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto", quantization_config=bnb_config, torch_dtype=torch.float16)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, adapter_id)
model.eval()

print("Generating response...")
prompt = "<s>[INST] Write a quick python function to reverse a string. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- OUTPUT ---")
print(result)

# Quick check for loops
if result.count("def ") > 2 or result.count("return") > 2:
    print("\n[!] WARNING: Potential loop detected.")
    sys.exit(1)
else:
    print("\n[+] SUCCESS: Response looks stable.")
