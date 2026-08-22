# PDF-Chatbot-V2 🚀

> **Lightweight, High-Performance Hybrid RAG Pipeline for General & Technical Documents**  
> Powered by **Qwen3:8B** (32K Context), **Nomic-Embed-v1.5** (768-dim), **ChromaDB + BM25 Fusion (RRF)**, and **Cross-Encoder Re-ranking**.
> Features a Premium **FastAPI + SSE** Web UI and **Dockerized** serving infrastructure.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture & Key Features](#-architecture--key-features)
- [Directory Structure](#-directory-structure)
- [Hardware & Software Requirements](#-hardware--software-requirements)
- [Installation Guide](#-installation-guide)
- [Quickstart (Train & Launch)](#-quickstart-train--launch)
- [Master Launchers (`run.ps1` / `run.sh`)](#-master-launchers)
- [Running in Google Colab (Free GPU)](#-running-in-google-colab-free-gpu)
- [Core Pipeline Scripts](#-core-pipeline-scripts)
- [Configuration Reference (`config.py`)](#-configuration-reference-configpy)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [License](#-license)

---

## 🌟 Overview

**PDF-Chatbot-V2** is a modular, high-accuracy Retrieval-Augmented Generation (RAG) system engineered to index and query large collections of PDF documents. It delivers grounded, factual answers with page-level citations while maintaining low system resource consumption.

With V2, you can interact with your documents via a **Premium Web Chatbot UI** (Dark Glassmorphism) or a fast **Interactive CLI**. 

### Handles All Document Types:
- 📋 **Instructional Guides & Manuals:** Step-by-step procedures, installation handbooks, and operational runbooks.
- 📄 **Informational Reports & Specifications:** Whitepapers, architectural blueprints, policy briefs, and summaries.
- 💻 **Software Engineering Documentation:** In-depth technical references across Python, Java, DevOps, and more.
- 🔍 **Scanned / Image-Heavy PDFs:** Fast ONNX-based OCR fallback via RapidOCR.

---

## 🏗️ Architecture & Key Features

```
┌─────────────────┐       ┌─────────────────┐       ┌────────────────────────┐
│  data/pdfs/     │ ----> │    PyMuPDF      │ ----> │ Structure-Aware        │
│  (Recursive)    │       │  (+ RapidOCR)   │       │ Chunker (512 / 64)     │
└─────────────────┘       └─────────────────┘       └───────────┬────────────┘
                                                                │
                                        ┌───────────────────────┴──────────────────────┐
                                        ▼                                              ▼
                             ┌──────────────────────┐                       ┌─────────────────────┐
                             │ nomic-embed (768d)   │                       │ Code-Aware BM25     │
                             │ ChromaDB Vector Store│                       │ Sparse Index        │
                             └──────────┬───────────┘                       └──────────┬──────────┘
                                        │                                              │
                                        └───────────────────────┬──────────────────────┘
                                                                │
                                                                ▼
                                                    ┌────────────────────────┐
                                                    │ Reciprocal Rank Fusion │
                                                    │ (RRF: 50% BM25 / 50% V)│
                                                    └───────────┬────────────┘
                                                                │
                                                                ▼
                                                    ┌────────────────────────┐
                                                    │ Cross-Encoder Reranker │
                                                    │ (mxbai-rerank-xsmall)  │
                                                    └───────────┬────────────┘
                                                                │
                    ┌───────────────────────────────────────────┴──────────────────────────────────────────┐
                    ▼                                                                                      ▼
 ┌─────────────────────────────────────┐                                ┌──────────────────────────────────────────────────┐
 │      DOCKER SERVING (Default)       │                                │                 LOCAL CLI MODE                   │
 │                                     │                                │                                                  │
 │  ┌──────────────┐ ┌──────────────┐  │                                │  ┌──────────────┐ ┌───────────────────────────┐  │
 │  │ Ollama       │ │ FastAPI      │  │                                │  │ Ollama       │ │ query.py                  │  │
 │  │ qwen3:8b     │ │ (web/app.py) │  │                                │  │ qwen3:8b     │ │ Interactive Terminal      │  │
 │  └──────┬───────┘ └──────┬───────┘  │                                │  └──────┬───────┘ └─────────────┬─────────────┘  │
 │         │                │          │                                │         │                       │                │
 │         └───────┬────────┘          │                                │         └───────────┬───────────┘                │
 │                 ▼                   │                                │                     ▼                            │
 │  ┌───────────────────────────────┐  │                                │  ┌────────────────────────────────────────────┐  │
 │  │   Browser (Web UI)            │  │                                │  │  Terminal Output (Streaming + Citations)   │  │
 │  │   Premium Dark Glassmorphism  │  │                                │  └────────────────────────────────────────────┘  │
 │  └───────────────────────────────┘  │                                │                                                  │
 └─────────────────────────────────────┘                                └──────────────────────────────────────────────────┘
```

- **Dual-Indexing Hybrid Retrieval:** Combines semantic vector search with sparse keyword search (BM25) using RRF.
- **Cross-Encoder Re-ranking:** Utilizes `mixedbread-ai/mxbai-rerank-xsmall-v1` to score semantic relevance before context generation.
- **Structure-Aware Chunking:** Intelligently respects code fences, markdown tables, headings, and indentation structures.
- **Dockerized Serving with BuildKit Caching:** Deploy the inference engine and web UI in a self-contained environment. Subsequent builds are extremely fast thanks to persistent `apt` and `pip` local cache mounts.
- **FastAPI Web Server:** SSE streaming endpoints that connect your local model to a beautiful, reactive front-end.
- **Hardware Acceleration:** Auto-detects NVIDIA CUDA GPU with VRAM monitoring and graceful fallback to optimized CPU execution.

---

## 📁 Directory Structure

```
PDF-ChatBot-V2/
├── .gitignore              # Ignores PDFs, indexes, logs, pycache, and docker artifacts
├── Dockerfile.serve        # Docker build configuration for inference & web app
├── docker-compose.yml      # Docker orchestration (with GPU passthrough)
├── README.md               # Complete documentation & usage guide
├── config.py               # Central configuration & tunable hyperparameters
├── data/pdfs/              # Place your PDF documents here (scanned recursively)
├── docker/                 # Docker entrypoint scripts
├── evaluate.py             # RAG evaluation metrics (Precision, Faithfulness, Latency)
├── ingest.py               # Standalone PDF ingestion & indexing pipeline
├── logs/                   # Runtime execution logs (auto-created)
├── main.py                 # Master pipeline entry point (Train & Chat)
├── query.py                # Standalone interactive terminal & single-query runner
├── requirements.txt        # Core Python dependencies
├── run.ps1                 # Master setup & pipeline launcher (PowerShell)
├── run.sh                  # Master setup & pipeline launcher (Bash)
├── test_rag.py             # 6-suite verification and regression test runner
├── utils.py                # Logging, hardware detection, tokenizer, & helpers
├── vectorstore/            # ChromaDB database & BM25 index (auto-created)
└── web/                    # FastAPI Backend and UI assets
    ├── app.py              # FastAPI server
    ├── requirements.txt    # Web dependencies
    └── static/index.html   # Single-file HTML/CSS/JS frontend
```

---

## 💻 Hardware & Software Requirements

### Recommended Hardware (GPU Accelerated):
- **GPU:** NVIDIA GPU with $\ge 6$ GB VRAM (e.g., RTX 3060, RTX 4060 or higher)
- **CPU:** 4 cores (x86_64 or ARM64) with AVX2 support
- **RAM:** 16 GB recommended

### Software Prerequisites:
- **Operating System:** Windows 10/11, Ubuntu 20.04+, macOS, or WSL2
- **Python:** `3.10`+ *(Auto-installed by `run.ps1`/`run.sh`)*
- **Ollama:** [Ollama](https://ollama.com/) *(Auto-installed by `run.ps1`/`run.sh`)*
- **Docker:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Optional, for web container serving)

---

## 📦 Installation Guide

Simply clone the repository and run the startup script — it will automatically verify/create directories, set up the virtual environment, install missing Python packages, configure Ollama, and start the system:

#### **Windows (PowerShell):**
```powershell
git clone https://github.com/Arjob-Das/PDF-ChatBot-V2.git
cd PDF-ChatBot-V2
.\run.ps1
```

#### **Linux / macOS / WSL (Bash):**
```bash
git clone https://github.com/Arjob-Das/PDF-ChatBot-V2.git
cd PDF-ChatBot-V2
./run.sh
```

---

## 🚀 Quickstart (Train & Launch)

Place your PDF documents into `data/pdfs/` (nested subfolders are fully supported), then execute the launcher. 

By default, the script trains your data locally, builds a Docker image, and launches the web interface at `http://localhost:8000`.

### Launcher Execution Modes:

```powershell
# 1. Default Mode (Train → Build Docker with BuildKit cache → Launch Web UI with GPU)
.\run.ps1

# 2. CPU-Only Mode (Forces container or local execution to run strictly on CPU)
.\run.ps1 --cpu

# 3. CLI Mode (Train → Launch Interactive Terminal, skips Docker/Web)
.\run.ps1 -cli

# 4. Local Web Mode (Train → Launch Local Web Server, skips Docker)
.\run.ps1 --no-docker

# 5. Retrain Mode (Wipe existing indexes and re-ingest all PDFs with optimized chunking)
.\run.ps1 --reset

# 6. Skip Ingestion (Skip training, just start the chatbot with existing data)
.\run.ps1 --skip-ingest

# 7. No Browser (Start container in background without auto-launching browser)
.\run.ps1 --no-browser
```

*(On Linux/macOS, use `./run.sh` with the same flags).*

---

## 🛠️ Master Launchers & Caching Strategy

The `run.ps1` and `run.sh` scripts are powerful end-to-end orchestrators. They handle:
1. Validating project directories.
2. Installing Python 3.10+ (if missing) and setting up a `venv`.
3. Installing PyTorch with CUDA support if an NVIDIA GPU is detected (or PyTorch CPU if `--cpu` is specified).
4. Setting up Ollama, verifying the daemon, and ensuring model availability (`qwen3:8b`).
5. Running the local document ingestion pipeline via `ingest.py` (structure-aware 1000-char chunks).
6. Building the `Dockerfile.serve` Docker image using Docker BuildKit cache mounts (`/var/cache/apt` and `/root/.cache/pip`), making rebuilds complete in under 2 seconds.
7. Starting the container with GPU passthrough (or CPU mode when `--cpu` is specified) with persistent host cache volume mounts:
   - `-v "${VectorstoreFull}:/app/vectorstore"` (ChromaDB + BM25 indexes)
   - `-v "${USERPROFILE}/.ollama:/root/.ollama"` (Cached LLM weights shared with host)
   - `-v "${USERPROFILE}/.cache/huggingface:/root/.cache/huggingface"` (Cached embedding & re-ranker weights shared with host)
8. Launching the web UI at `http://localhost:8000`.

---

## ☁️ Running in Google Colab (Free GPU)

If your local computer lacks an NVIDIA GPU, you can execute document ingestion and interactive querying on **Google Colab's free GPU instances (T4 GPU)**.

1. Open a new [Google Colab](https://colab.research.google.com/) notebook and enable the T4 GPU.
2. Run the following to set up the environment:
```python
!git clone https://github.com/Arjob-Das/PDF-ChatBot-V2.git
%cd PDF-ChatBot-V2
!pip install -r requirements.txt
!curl -fsSL https://ollama.com/install.sh | sh
```
3. Start Ollama and pull the model:
```python
import subprocess, time
subprocess.Popen(["ollama", "serve"])
time.sleep(5)
!ollama pull qwen3:8b
```
4. Upload your PDFs to `data/pdfs/` and run training:
```python
!python ingest.py --reset
```
5. Export your trained `vectorstore/` index back to your local machine for fast querying.

---

## 🧩 Core Pipeline Scripts

If you prefer granular control, you can execute the core Python scripts directly:

### 1. Ingestion (`ingest.py`)
Processes PDFs recursively, generates structure-aware chunks, creates embeddings, and builds the dual index.
```bash
python ingest.py --reset           # Wipe index and re-ingest
python ingest.py --no-ocr          # Ingest without OCR
```

### 2. Interactive Terminal (`query.py`)
The classic interactive terminal with streaming, memory, and multi-PDF synthesis.
```bash
python query.py
python query.py --query "How do I configure network settings?"
```

### 3. Self-Validation Tests (`test_rag.py`)
Executes 6 validation suites verifying index integrity, BM25 sync, retrieval quality, and synthetic chunking speed.
```bash
python test_rag.py --skip-llm      # Deterministic tests only
```

### 4. Performance Benchmark (`benchmark.py`)
Evaluates precision and latency across 30 curated questions in 6 domains.
```bash
python benchmark.py --domain Python
```

---

## ⚙️ Configuration Reference (`config.py`)

All pipeline hyperparameters can be customized in [`config.py`](config.py):

| Parameter | Default Value | Description |
| :--- | :---: | :--- |
| **`CHUNK_SIZE`** | `1000` | Target character length per chunk (preserves complete procedures) |
| **`CHUNK_OVERLAP`** | `150` | Overlap character count between consecutive chunks |
| **`EMBEDDING_MODEL`** | `"nomic-ai/nomic-embed-text-v1.5"` | HuggingFace embedding model (768-dim) |
| **`OLLAMA_MODEL`** | `"qwen3:8b"` | Primary Ollama model name |
| **`OLLAMA_CONTEXT_WINDOW`** | `8192` | 8K context window (fits 100% in VRAM, zero CPU offload) |
| **`TOP_K`** | `10` | Number of final chunks presented to the LLM |
| **`RETRIEVAL_CANDIDATES`** | `80` | Raw candidate pool depth per retriever before RRF & re-ranking |
| **`VECTOR_WEIGHT`** | `0.5` | Dense vector contribution in Reciprocal Rank Fusion |
| **`USE_RERANKER`** | `True` | Enables Cross-Encoder re-ranking (`mxbai-rerank-xsmall-v1`) |
| **`ENABLE_OCR`** | `True` | RapidOCR ONNX fallback for scanned pages |
| **`SERVE_PORT`** | `8000` | FastAPI Web server port |

---

## ❓ Troubleshooting & FAQs

### Q: Does PDF-Chatbot-V2 support CUDA GPU acceleration?
**Yes.** If PyTorch with CUDA is installed and an NVIDIA GPU is present, both embedding generation and cross-encoder re-ranking automatically run on `[CUDA]`. The Docker container also automatically maps GPUs if available.

### Q: Ollama returns `Model 'qwen3:8b' not found`
Run `ollama pull qwen3:8b`. The master run scripts should handle this for you automatically. If you prefer using an already installed model (e.g. `llama3:8b`), PDF-Chatbot-V2 will automatically fall back to it without crashing.

### Q: Docker fails to build or start.
Make sure Docker Desktop is running and WSL2 backend is configured correctly (on Windows). If you just want to run the web UI without Docker, use `.\run.ps1 --no-docker`.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
