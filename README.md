# 🌾 Wheaty AI

A locally-trained AI agent built on top of **Mistral Small 24B**, fine-tuned with LoRA adapters and served via **Ollama**.

Wheaty is a friendly and capable AI assistant that helps with coding, explanations, and everyday questions — running entirely on your own hardware.

## ✨ Features

- **Local-first** — Runs entirely on your machine via Ollama (no cloud dependency)
- **Fine-tuned intelligence** — Custom LoRA adapters trained on curated coding, logic, and creative datasets
- **Agentic tooling** — Built-in `<run_shell>` and `<write_file>` tool-use capabilities
- **Thinking mode** — Uses `<think>` blocks for step-by-step reasoning before responding
- **Streaming chat** — Real-time streamed responses with color-coded thinking/output

## 🛠️ Tech Stack

| Component | Details |
|-----------|---------|
| Base Model | `unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit` |
| Fine-tuning | LoRA (rank 64) via Hugging Face PEFT + Transformers |
| Quantization | 4-bit NF4 via BitsAndBytes |
| Serving | Ollama |
| Hardware | NVIDIA Titan RTX (24GB VRAM) |

## 📂 Project Structure

```
WheatyAI/
├── Modelfile              # Ollama model definition
├── dataset/
│   └── training_data.json # Custom training examples
├── train_wheaty_v4.py     # Latest training script (Omega-Zero v2)
├── chat_wheaty_v3.py      # Interactive chat client with tool-use
├── export_wheaty.py       # Merge LoRA adapters & export to Ollama
├── finalize_ascension.py  # Automated post-training pipeline
├── autopilot_test.py      # Quick model validation test
└── test_wheaty.py         # Dependency smoke test
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with 24GB+ VRAM
- [Ollama](https://ollama.com/) installed
- CUDA toolkit

### Run with Ollama (easiest)
```bash
ollama run wheaty:latest
```

### Train from scratch
```bash
pip install torch transformers peft datasets bitsandbytes colorama
python train_wheaty_v4.py
```

### Chat via Python client
```bash
python chat_wheaty_v3.py
```

## 📝 Training Details

Wheaty is fine-tuned using a mix of high-quality datasets:
- **[OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)** — General instruction following (5K examples)
- **[Evol-Instruct-Code-80k](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1)** — Coding tasks (4K examples)
- **[Orca Math](https://huggingface.co/datasets/microsoft/Orca-Math-Word-Problems-200K)** — Mathematical reasoning (2K examples)

Training config: 250 steps, batch size 1 with 128 gradient accumulation, cosine LR schedule (5e-6), LoRA rank 64.

## 📄 License

This project is for personal/educational use. The base model ([Mistral Small](https://huggingface.co/mistralai)) is subject to its own license terms.
