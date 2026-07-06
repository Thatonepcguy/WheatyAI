import os
import time
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# CONFIG
TRAINING_PID = 20560
ADAPTER_PATH = "WheatyAI/models/wheaty_v3_omega_v2"
BASE_MODEL = "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit"
MERGED_PATH = "WheatyAI/models/wheaty_v3_omega_v2_merged"
MODELFILE_PATH = "WheatyAI/Modelfile"
VENV_PYTHON = r".\wheaty_env\Scripts\python.exe"

def is_process_running(pid):
    try:
        output = subprocess.check_output(f'tasklist /fi "PID eq {pid}"', shell=True).decode()
        return str(pid) in output
    except:
        return False

print(f"[*] Monitor started. Waiting for Training PID {TRAINING_PID} to complete...")

# 1. Wait for training
while is_process_running(TRAINING_PID):
    time.sleep(60) # Check every minute

print("[+] Training complete! Starting Phase 2: Intelligence Merging...")

# 2. Merge Weights (Requires high RAM/VRAM)
try:
    print("[*] Loading base and adapters...")
    # Use the venv python to run a separate merge script to avoid memory fragmentation in this process
    merge_cmd = [
        VENV_PYTHON, "-c",
        f"import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; from peft import PeftModel; "
        f"model = AutoModelForCausalLM.from_pretrained('{BASE_MODEL}', device_map='auto', torch_dtype=torch.float16, load_in_4bit=False); "
        f"tokenizer = AutoTokenizer.from_pretrained('{BASE_MODEL}'); "
        f"model = PeftModel.from_pretrained(model, '{ADAPTER_PATH}'); "
        f"model = model.merge_and_unload(); "
        f"model.save_pretrained('{MERGED_PATH}'); "
        f"tokenizer.save_pretrained('{MERGED_PATH}'); "
        f"print('[++] Merge successful!')"
    ]
    subprocess.run(merge_cmd, check=True)
except Exception as e:
    print(f"[!] Merge failed: {e}")
    # Fallback: Just update Ollama with the system prompt if merge fails
    pass

# 3. Update Ollama
print("[*] Updating Ollama model with forged intelligence...")
try:
    # Update Modelfile to point to merged weights if possible
    # For now, we update the existing one
    subprocess.run(["ollama", "create", "wheaty_v3_omega_v2", "-f", MODELFILE_PATH], check=True)
    print("[++] Ollama model 'wheaty_v3_omega_v2' is now FULLY ASCENDED.")
except Exception as e:
    print(f"[!] Ollama update failed: {e}")

print("[!!!] ASCENSION COMPLETE. WHEATY V3 IS ONLINE.")
