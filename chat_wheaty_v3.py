import os
import warnings
import sys
import subprocess
import re
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TextStreamer, 
    StoppingCriteria, 
    StoppingCriteriaList,
    logging as hf_logging
)
from peft import PeftModel
from colorama import init, Fore, Style
import shutil

# --- [SYSTEM SETUP] ---
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()
init(autoreset=True)

CONFIG = {
    "BASE_MODEL": "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit",
    "LORA_PATH": "WheatyAI/models/wheaty_v3_omega",
    "MAX_NEW_TOKENS": 2048,
    "TEMPERATURE": 0.3,
    "REPETITION_PENALTY": 1.15
}

class WheatyToolbox:
    @staticmethod
    def run_shell(command):
        print(f"\n{Fore.YELLOW}🔨 EXEC ❯ {Style.DIM}{command}{Style.RESET_ALL}")
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=45)
            output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            return output[:2000] if len(output) > 2000 else output
        except Exception as e: return f"ERROR: {str(e)}"

    @staticmethod
    def write_file(path, content):
        print(f"\n{Fore.YELLOW}💾 SAVE ❯ {Style.DIM}{path}{Style.RESET_ALL}")
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f: f.write(content)
            return f"SUCCESS: Written {path}"
        except Exception as e: return f"ERROR: {str(e)}"

class WheatyV3Streamer(TextStreamer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.stop_signal = False
        self.in_think = False
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.stop_signal: return
        
        # Handle <think> tags visually
        if "<think>" in text:
            self.in_think = True
            sys.stdout.write(f"\n{Fore.WHITE}{Style.DIM}█ THINKING ❯{Style.RESET_ALL} ")
            return
        if "</think>" in text:
            self.in_think = False
            sys.stdout.write(f"\n\n{Fore.GREEN}{Style.BRIGHT}█ RESPONSE ❯{Style.RESET_ALL}\n")
            return

        if any(tag in text.lower() for tag in ["[/inst]", "user ❯"]):
            self.stop_signal = True; return
            
        color = Fore.WHITE + Style.DIM if self.in_think else Fore.GREEN
        sys.stdout.write(f"{color}{text}{Style.RESET_ALL}")
        sys.stdout.flush()

# --- [BRAIN LOAD] ---
print(f"\n{Fore.MAGENTA}╔{'═' * 78}╗")
print(f"{Fore.MAGENTA}║{Style.BRIGHT}{'WHEATY V3 - OMEGA-ZERO AGENTIC LINK':^78}{Style.NORMAL}{Fore.MAGENTA}║")
print(f"{Fore.MAGENTA}║{'24B Heavyweight | Adaptive Reasoning | Titan RTX':^78}║")
print(f"{Fore.MAGENTA}╚{'═' * 78}╝\n")

tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-bnb-4bit") # Stable proxy
base_model = AutoModelForCausalLM.from_pretrained(
    CONFIG["BASE_MODEL"], device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, CONFIG["LORA_PATH"])
model.eval()

def agent_loop(user_input):
    system_prompt = (
        "You are Wheaty V3, an Elite local AI Agent. Operating under Omega-Zero Protocol.\n"
        "You use <think> blocks for formal verification and proof-checking.\n"
        "TOOLS: <run_shell>cmd</run_shell>, <write_file path='f'>content</write_file>\n"
        "MANDATE: Technical precision only."
    )
    
    # Mistral Format
    ctx = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_input} [/INST]"
    
    for turn in range(10):
        inputs = tokenizer(ctx, return_tensors="pt").to("cuda")
        streamer = WheatyV3Streamer(tokenizer)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, streamer=streamer, max_new_tokens=2048,
                temperature=0.3, do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )
        
        res = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        executed = False
        if m := re.search(r"<run_shell>(.*?)</run_shell>", res, re.DOTALL):
            r = WheatyToolbox.run_shell(m.group(1).strip()); executed = True
        elif m := re.search(r"<write_file path=['\"](.*?)['\"]>(.*?)</write_file>", res, re.DOTALL):
            r = WheatyToolbox.write_file(m.group(1), m.group(2)); executed = True

        if executed:
            ctx += f"{res} [RESULT]: {r} [/INST]"
        else:
            break

if __name__ == "__main__":
    while True:
        try:
            print(f"\n{Fore.MAGENTA}{Style.BRIGHT}User ❯{Style.RESET_ALL} ", end="")
            u_in = input("")
            if u_in.lower() in ["exit", "quit"]: break
            agent_loop(u_in)
        except KeyboardInterrupt: print("\n ● Reset."); continue
