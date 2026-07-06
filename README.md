# 🌾 Wheaty AI

A local AI agent built on **Mistral**, served via **Ollama**. Wheaty is a capable AI assistant that helps with coding, explanations, and everyday questions — running entirely on your own hardware.

## 🚀 Quick Start

### 1. Install Ollama
Download and install from [ollama.com](https://ollama.com/)

### 2. Clone this repo
```bash
git clone https://github.com/Thatonepcguy/WheatyAI.git
cd WheatyAI
```

### 3. Create the model
```bash
ollama create wheaty -f Modelfile
```

### 4. Run it
```bash
ollama run wheaty
```

That's it! Ollama will download the base model automatically on first run.

## ✨ Features

- **Local-first** — Runs entirely on your machine, no cloud, no API keys
- **Agentic tooling** — Built-in `<run_shell>` and `<write_file>` tool-use capabilities
- **Thinking mode** — Uses `<think>` blocks for step-by-step reasoning before responding
- **No setup headaches** — Just Ollama + the Modelfile

## 🛠️ Tech Stack

| Component | Details |
|-----------|---------|
| Base Model | Mistral |
| Serving | [Ollama](https://ollama.com/) |
| Config | Modelfile |

## 📄 License

Apache 2.0 — See [LICENSE](LICENSE) for details.

Copyright 2026 Wheat Studios
