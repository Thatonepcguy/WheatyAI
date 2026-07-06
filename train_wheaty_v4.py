"""
WHEATY V3 - OMEGA-ZERO ASCENSION (STABLE GOLDEN STACK) - FIXED V2
Architected by: Gemini CLI for the Titan RTX 24GB Environment
Version: 3.5.0-Omega (Corrected Tokenizer + Expanded Data)
"""

import os
import sys
import torch
import warnings
from datetime import datetime

# --- [SYSTEM SETUP] ---
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, concatenate_datasets
from colorama import init, Fore, Style

init(autoreset=True)

MODEL_ID = "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit"
OUTPUT_DIR = "WheatyAI/models/wheaty_v3_omega_v2"
LOG_FILE = "wheaty_v3_omega_v2.log"

def print_log(msg, color=Fore.CYAN):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{ts}] {msg}{Style.RESET_ALL}")
    with open(LOG_FILE, "a") as f: f.write(f"[{ts}] {msg}\n")

# --- [PHASE 1: DATASET] ---
print_log("--- PHASE 1: PREPARING OMEGA-ZERO ELITE DATASET ---", Fore.YELLOW)
try:
    # CORRECT TOKENIZER: Must match the model!
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print_log(f"TOKENIZER ERROR: {e}. Falling back to standard Mistral tokenizer.", Fore.RED)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Small-Instruct-2409", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

try:
    # ELITE EXPANSION
    opus_logic = load_dataset("teknium/OpenHermes-2.5", split="train[:5000]") 
    o1_coding = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train[:4000]")
    math_logic = load_dataset("microsoft/Orca-Math-Word-Problems-200K", split="train[:2000]")

    def format_omega_zero(example):
        instruction = example.get("instruction", "")
        # Handle different dataset structures
        if not instruction:
            instruction = example.get("question", "")
        
        output = example.get("output", "")
        if not output:
            output = example.get("answer", "")

        # Mistral Instruct Format: [INST] prompt [/INST]
        text = f"<s>[INST] {instruction} [/INST] THOUGHT: Analyzing... CODE: {output}</s>"
        res = tokenizer(text, truncation=True, max_length=1024) # Increased length
        return {"input_ids": res["input_ids"], "attention_mask": res["attention_mask"]}

    dataset = concatenate_datasets([opus_logic, o1_coding, math_logic]).shuffle(seed=42)
    tokenized_dataset = dataset.map(format_omega_zero, remove_columns=dataset.column_names)
    print_log(f"Elite Dataset Ready: {len(tokenized_dataset)} examples.")
except Exception as e:
    print_log(f"DATA ERROR: {e}", Fore.RED); sys.exit(1)

# --- [PHASE 2: MODEL] ---
print_log("--- PHASE 2: LOADING 24B BRAIN ---", Fore.MAGENTA)
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager" 
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=64, # Boosted Logic Capacity
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    print_log("Logic Capacity: Rank 64 - Ascension Mode.")
except Exception as e:
    print_log(f"MODEL ERROR: {e}", Fore.RED); sys.exit(1)

# --- [PHASE 3: TRAINING] ---
print_log("--- PHASE 3: TRAINING START ---", Fore.GREEN)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=128, # Restored for high-fidelity gradients
    warmup_steps=100,
    max_steps=250, # Full Ascension steps
    learning_rate=5e-6, 
    fp16=True,
    logging_steps=1,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    save_strategy="steps",
    save_steps=50,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# MANDATORY SAVE
print_log("--- SAVING FINAL ADAPTERS ---", Fore.YELLOW)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print_log("ASCENSION COMPLETE. Wheaty V3 OMEGA-ZERO V2 is forged.")
