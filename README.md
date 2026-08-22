# PDF-Chatbot-V2 🚀

> **Lightweight, High-Performance Hybrid RAG Pipeline for General & Technical Documents**  
> Powered by **Qwen3:8B** (32K Context), **Nomic-Embed-v1.5** (768-dim), **ChromaDB + BM25 Fusion (RRF)**, and **Cross-Encoder Re-ranking**.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Architecture & Key Features](#-architecture--key-features)
- [Directory Structure](#-directory-structure)
- [Hardware & Software Requirements](#-hardware--software-requirements)
- [Installation Guide](#-installation-guide)
- [Ollama LLM Setup](#-ollama-llm-setup)
- [Quickstart (Train & Chat)](#-quickstart-train--chat)
- [Running in Google Colab (Free GPU / Cloud Training)](#-running-in-google-colab-free-gpu--cloud-training)
- [CLI Scripts & Run Guides](#-cli-scripts--run-guides)
  - [1. Master Launchers (`run.ps1` / `run.sh` / `main.py`)](#1-master-launchers-runps1--runsh--mainpy)
  - [2. Interactive Chat Terminal (`query.py`)](#2-interactive-chat-terminal-querypy)
  - [3. Document Ingestion & Training (`ingest.py`)](#3-document-ingestion--training-ingestpy)
  - [4. Self-Validation Test Suite (`test_rag.py`)](#4-self-validation-test-suite-test_ragpy)
  - [5. Performance Benchmark (`benchmark.py`)](#5-performance-benchmark-benchmarkpy)
  - [6. Evaluation Framework (`evaluate.py`)](#6-evaluation-framework-evaluatepy)
- [Configuration Reference (`config.py`)](#-configuration-reference-configpy)
- [Interactive Terminal Slash Commands](#-interactive-terminal-slash-commands)
- [Repository & Git Guidelines](#-repository--git-guidelines)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [License](#-license)

---

## 🌟 Overview

**PDF-Chatbot-V2** is a modular, high-accuracy Retrieval-Augmented Generation (RAG) system engineered to index and query large collections of PDF documents. It delivers grounded, factual answers with page-level citations while maintaining low system resource consumption.

### Handles All Document Types:
- 📋 **Instructional Guides & Manuals:** Step-by-step procedures, installation handbooks, and operational runbooks.
- 📄 **Informational Reports & Specifications:** Whitepapers, architectural blueprints, policy briefs, and summaries.
- 💻 **Software Engineering Documentation:** In-depth technical references across **Python, Java, DevOps (Kubernetes/Docker/Terraform), Shell Scripting, and Fullstack Web development**.
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
                                                                ▼
                                                    ┌────────────────────────┐
                                                    │ Qwen3:8B (32k Context) │
                                                    │ Streaming Answer + Cits│
                                                    └────────────────────────┘
```

- **Dual-Indexing Hybrid Retrieval:** Combines dense semantic vector search with sparse keyword search (BM25) using Reciprocal Rank Fusion (RRF). Eliminates zero-hits on exact technical terms, commands, and error codes.
- **Cross-Encoder Re-ranking:** Utilizes `mixedbread-ai/mxbai-rerank-xsmall-v1` to score semantic relevance against candidate pools before context generation.
- **Structure-Aware Chunking:** Intelligently respects code fences, markdown tables, headings, and indentation structures to prevent severed syntax blocks.
- **Extended Context Window:** Configured for **32,768 tokens** with `qwen3:8b` (with automatic fallback to available Ollama models like `llama3:8b`).
- **Contiguous Chunk Merging:** Automatically stitches adjacent chunks from the same page and document to preserve reading coherence.
- **Hardware Acceleration:** Auto-detects NVIDIA CUDA GPU with VRAM monitoring and graceful fallback to optimized CPU execution.

---

## 📁 Directory Structure

```
PDF-ChatBot-V2/
├── .gitattributes          # Line ending normalization & binary attributes
├── .gitignore              # Ignores PDFs, indexes, logs, pycache, and agent files
├── LICENSE                 # MIT License
├── README.md               # Complete documentation & usage guide
├── benchmark.py            # 30-query multi-domain performance benchmark
├── comparison_report.md    # Generated evaluation report
├── config.py               # Central configuration & tunable hyperparameters
├── data/
│   └── pdfs/               # Place your PDF documents here (scanned recursively)
│       └── .gitkeep        # Preserves directory in git while ignoring PDF files
├── evaluate.py             # RAG evaluation metrics (Precision, Faithfulness, Latency)
├── ingest.py               # Standalone PDF ingestion & indexing pipeline
├── logs/                   # Runtime execution logs (auto-created)
│   └── .gitkeep        # Preserves logs directory in git
├── main.py                 # Master pipeline entry point (Train & Chat)
├── query.py                # Standalone interactive terminal & single-query runner
├── requirements.txt        # Python dependency manifest
├── run.ps1                 # Automated setup & pipeline launcher (PowerShell)
├── run.sh                  # Automated setup & pipeline launcher (Bash)
├── test_rag.py             # 6-suite verification and regression test runner
├── utils.py                # Logging, hardware detection, tokenizer, & helpers
└── vectorstore/            # ChromaDB database & BM25 index (auto-created)
    └── .gitkeep            # Preserves vectorstore directory in git
```

---

## 💻 Hardware & Software Requirements

### Minimum Hardware:
- **CPU:** 4 cores (x86_64 or ARM64) with AVX2 support
- **RAM:** 8 GB RAM (16 GB recommended)
- **Disk:** 5 GB free disk space (for models and vector index)

### Recommended Hardware (GPU Accelerated):
- **GPU:** NVIDIA GPU with $\ge 6$ GB VRAM (e.g., RTX 3060, RTX 4060 or higher)
- **CUDA:** Version 12.x or 13.x with PyTorch CUDA support

### Software Prerequisites:
- **Operating System:** Windows 10/11, Ubuntu 20.04+, Debian 11+, macOS (Apple Silicon or Intel), or WSL2
- **Python:** `3.10`, `3.11`, or `3.12` *(Automatically verified & set up by `run.ps1`/`run.sh`)*
- **Ollama:** [Ollama](https://ollama.com/) *(Automatically verified, launched, and models pulled by `run.ps1`/`run.sh`)*

---

## 📦 Installation Guide

### Option A: Fully Automated Launch (Recommended)
Simply clone the repository and run the startup script — it will automatically verify/create directories, set up the virtual environment, install missing Python packages and CUDA PyTorch, check Ollama, verify installed models, and start the chatbot:

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
chmod +x run.sh
./run.sh
```

---

### Option B: Manual Setup

1. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS / WSL
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   # If using NVIDIA GPU:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # Install requirements:
   pip install -r requirements.txt
   ```

3. **Install & Start Ollama:**
   ```bash
   ollama serve
   ollama pull qwen3:8b
   ```

4. **Launch Pipeline:**
   ```bash
   python main.py
   ```

---

## 🦙 Ollama LLM Setup

1. **Start the Ollama daemon:**
   ```bash
   ollama serve
   ```

2. **Pull the Qwen3 model (Recommended):**
   ```bash
   ollama pull qwen3:8b
   ```
   *(Note: If `qwen3:8b` is not installed, the chatbot automatically falls back to any existing installed model, such as `llama3:8b`)*.

---

## 🚀 Quickstart (Train & Chat)

Place your PDF documents into `data/pdfs/` (nested subfolders are fully supported), then execute:

#### **PowerShell (Windows — Automated):**
```powershell
.\run.ps1
```

#### **Bash (Linux / macOS / WSL — Automated):**
```bash
./run.sh
```

#### **Direct Python Entrypoint:**
```bash
python main.py
```

---

## ☁️ Running in Google Colab (Free GPU / Cloud Training)

If your local computer lacks sufficient RAM or an NVIDIA GPU, you can execute document ingestion and interactive querying on **Google Colab's free GPU instances (T4 GPU)**.

### Step 1: Open Colab and Enable GPU
1. Go to [Google Colab](https://colab.research.google.com/).
2. In the top menu, navigate to **Runtime** $\rightarrow$ **Change runtime type**.
3. Select **T4 GPU** (or any available GPU) under *Hardware accelerator* and click **Save**.

---

### Step 2: Clone the Repository & Change Directory
Run the following cell to clone the repository directly into your Colab session and navigate to the project directory:

```python
# Clone the repository
!git clone https://github.com/Arjob-Das/PDF-ChatBot-V2.git

# Change working directory to the repository folder
%cd PDF-ChatBot-V2
```

---

### Step 3: Mount Google Drive & Upload PDFs (Optional)
If your PDF documents are stored in Google Drive, mount your drive and copy them into `data/pdfs/`:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copy your PDFs into the data/pdfs directory:
# !cp -r "/content/drive/MyDrive/My_PDFs/"* data/pdfs/
```

*(Alternatively, upload PDFs directly using Colab's file browser into `/content/PDF-ChatBot-V2/data/pdfs/`)*.

---

### Step 4: Install Python Dependencies
Install all required Python packages with PyTorch CUDA acceleration:

```python
!pip install -r requirements.txt
```

---

### Step 5: Install & Start Ollama in Background
Colab does not include Ollama by default. Run the following cell to install Ollama, start the daemon in the background, and pull the LLM model:

```python
# 1. Install Ollama binary
!curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama daemon in the background
import subprocess, time
subprocess.Popen(["ollama", "serve"])
time.sleep(5)  # Allow background service to initialize

# 3. Pull the recommended LLM model (or llama3:8b)
!ollama pull qwen3:8b
```

---

### Step 6: Ingest & Train on PDFs

#### Option 6A: Command-Line Training Execution
```python
# Run document ingestion and index creation
!python ingest.py --batch-size 20
```

#### Option 6B: Interactive Notebook Magic Method (`%run -i`)
Using `%run -i` executes the pipeline while loading all functions, classes, and variables directly into Colab's interactive session memory. This allows you to inspect objects and query directly from notebook cells:

```python
# Interactive execution: loads ChatEngine, Retriever, and variables into notebook memory
%run -i main.py --skip-ingest
```

---

### Step 7: Run Programmatic Queries in Colab Cells
Once indexed or loaded, you can query directly inside Colab code cells:

```python
from query import ChatEngine

# Initialize the chat engine
chat = ChatEngine(top_k=8)

# Ask questions programmatically
response = chat.ask("What are the key instructions and configurations described in the document?")
```

---

### Step 8: Download Generated Indexes for Local Use (Optional)
Once Colab finishes embedding and building the index, download the generated `vectorstore/` folder to your local machine so you can query locally with zero training overhead:

```python
# Archive the vector store and BM25 index
!zip -r vectorstore_export.zip vectorstore/

# Download directly in browser:
from google.colab import files
files.download('vectorstore_export.zip')
```
*(Extract `vectorstore_export.zip` into your local `PDF-ChatBot-V2/` directory and run `python main.py --skip-ingest` to query immediately).*

---

## 🛠️ CLI Scripts & Run Guides

### 1. Master Launchers (`run.ps1` / `run.sh` / `main.py`)
Verifies hardware and software dependencies, checks/runs ingestion on `data/pdfs/`, and launches the interactive terminal.

```bash
# Automated scripts (forward all flags to main.py)
.\run.ps1 --reset
./run.sh --no-ocr

# Or run directly via Python
python main.py [FLAGS]
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--reset` | boolean | `False` | Force wipe existing vector store and re-train/re-ingest all PDFs from scratch |
| `--skip-ingest` | boolean | `False` | Skip training/ingestion check and launch interactive chat immediately |
| `--pdf-dir` | path | `data/pdfs/` | Custom directory containing PDF files to ingest |
| `--batch-size` | int | `10` | Number of PDFs processed per batch (OOM guard) |
| `--top-k`, `-k` | int | `8` | Number of context chunks retrieved for question answering |
| `--ocr` / `--no-ocr` | boolean | `True` | Enable or disable RapidOCR fallback for scanned pages |

---

### 2. Interactive Chat Terminal (`query.py`)
Run the interactive terminal or execute single-shot queries with citations and conversation memory.

```bash
# Launch interactive conversation terminal
python query.py

# Single-shot query from terminal
python query.py --query "How do I configure network settings?"

# Query with custom top-K retrieved context chunks
python query.py --query "What is Appliance Manager?" --top-k 10
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--query`, `-q` | string | `None` | Run a single query directly and exit |
| `--top-k`, `-k` | int | `8` | Number of retrieved chunks to inject into context |
| `--device` | string | auto | Compute device override (`cuda` or `cpu`) |
| `--model` | string | `qwen3:8b` | Ollama model name override |

---

### 3. Document Ingestion & Training (`ingest.py`)
Processes PDFs recursively, generates structure-aware chunks, creates 768-dim embeddings, and builds the dual ChromaDB + BM25 index.

```bash
# Standard ingestion of data/pdfs/
python ingest.py

# Wipe existing index and re-ingest
python ingest.py --reset

# Ingest without OCR (faster for purely digital text PDFs)
python ingest.py --no-ocr --batch-size 20
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--pdf-dir` | path | `data/pdfs/` | Path to directory containing PDF documents (scans subdirectories recursively) |
| `--reset` | boolean | `False` | Delete existing ChromaDB collection & BM25 index before ingesting |
| `--batch-size` | int | `10` | Number of PDF files processed per batch |
| `--ocr` / `--no-ocr` | boolean | `True` | Enable or disable RapidOCR fallback |
| `--device` | string | auto | Compute device override (`cuda` or `cpu`) |

---

### 4. Self-Validation Test Suite (`test_rag.py`)
Executes 6 validation suites verifying index integrity, BM25 sync, embedding throughput, retrieval quality, QA faithfulness, and synthetic chunking speed.

```bash
# Run full validation suite
python test_rag.py

# Run validation without requiring Ollama (deterministic tests only)
python test_rag.py --skip-llm

# Detailed logging output
python test_rag.py --verbose
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--skip-llm` | boolean | `False` | Skip LLM-dependent tests (runs vector, BM25, and retrieval tests offline) |
| `--verbose`, `-v` | boolean | `False` | Display detailed chunk previews and debugging logs |

---

### 5. Performance Benchmark (`benchmark.py`)
Evaluates retrieval precision, latency, and keyword grounding across 30 curated questions in 6 domains (General, Python, Java, DevOps, Shell, Fullstack).

```bash
# Run full 30-query benchmark
python benchmark.py

# Filter benchmark to a single domain
python benchmark.py --domain DevOps
python benchmark.py --domain Python
python benchmark.py --domain Shell
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--domain` | string | `None` | Filter queries to a specific domain (`General`, `Instructional`, `Python`, `Java`, `DevOps`, `Shell`, `Fullstack`) |
| `--skip-llm` | boolean | `False` | Run retrieval-only benchmark without LLM generation |

---

### 6. Evaluation Framework (`evaluate.py`)
Computes reference-free Context Precision, Faithfulness, and Retrieval Latency metrics and exports a formatted markdown report.

```bash
# Run evaluation and export comparison_report.md
python evaluate.py --output comparison_report.md

# Fast evaluation without Ollama
python evaluate.py --skip-llm
```

| Flag | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--output`, `-o` | path | `comparison_report.md` | Path to save the markdown evaluation report |
| `--skip-llm` | boolean | `False` | Run retrieval evaluation only |

---

## ⚙️ Configuration Reference (`config.py`)

All pipeline hyperparameters can be customized in [`config.py`](config.py):

| Parameter | Default Value | Description |
| :--- | :---: | :--- |
| **`PDF_DIR`** | `PROJECT_ROOT / "data" / "pdfs"` | Directory scanned recursively for PDF files |
| **`VECTORSTORE_DIR`** | `PROJECT_ROOT / "vectorstore"` | Persistent directory for ChromaDB vector storage |
| **`BM25_INDEX_PATH`** | `VECTORSTORE_DIR / "bm25_index.pkl"` | Path to serialized BM25 index file |
| **`LOG_DIR`** | `PROJECT_ROOT / "logs"` | Directory for timestamped execution logs |
| **`CHUNK_SIZE`** | `512` | Character length target per chunk |
| **`CHUNK_OVERLAP`** | `64` | Overlap character count between consecutive chunks |
| **`BATCH_SIZE`** | `10` | Number of PDFs processed per batch during ingestion |
| **`CODE_BLOCK_AWARE`** | `True` | Enables structure-aware boundary detection for code and markdown |
| **`EMBEDDING_MODEL`** | `"nomic-ai/nomic-embed-text-v1.5"` | HuggingFace embedding model |
| **`EMBEDDING_DIMENSION`** | `768` | Vector embedding dimension |
| **`OLLAMA_MODEL`** | `"qwen3:8b"` | Primary Ollama model name |
| **`OLLAMA_BASE_URL`** | `"http://localhost:11434"` | URL of local Ollama service |
| **`OLLAMA_CONTEXT_WINDOW`** | `32768` | Context window size passed in Ollama options |
| **`TOP_K`** | `8` | Number of final chunks presented to the LLM |
| **`RETRIEVAL_CANDIDATES`** | `60` | Raw candidates retrieved per retriever before fusion & re-ranking |
| **`VECTOR_WEIGHT`** | `0.5` | Dense vector contribution in Reciprocal Rank Fusion |
| **`BM25_WEIGHT`** | `0.5` | Sparse BM25 keyword contribution in Reciprocal Rank Fusion |
| **`RRF_K`** | `60` | Reciprocal Rank Fusion smoothing constant |
| **`USE_RERANKER`** | `True` | Enables Cross-Encoder re-ranking |
| **`RERANKER_MODEL`** | `"mixedbread-ai/mxbai-rerank-xsmall-v1"` | Lightweight cross-encoder model |
| **`ENABLE_OCR`** | `True` | RapidOCR ONNX fallback for scanned pages |
| **`OCR_MIN_CHARACTERS`** | `50` | Minimum character threshold below which a page is sent to OCR |
| **`OCR_DPI`** | `200` | Pixmap resolution for OCR rendering |
| **`MAX_MEMORY_TURNS`** | `5` | Maximum number of past conversation turns stored in memory |

---

## ⌨️ Interactive Terminal Slash Commands

Inside the interactive chat terminal (`python query.py`, `python main.py`, `.\run.ps1`, or `./run.sh`), the following commands are available:

| Command | Action |
| :--- | :--- |
| `/clear` | Resets conversation memory buffer |
| `/sources` | Displays a detailed breakdown table of the chunks retrieved for the last question |
| `/count` | Shows the total number of chunks currently indexed in ChromaDB |
| `/help` | Displays the help menu with all available commands |
| `/quit` or `/exit` | Exits the interactive chat session |

---

## 🛡️ Repository & Git Guidelines

- **Data Safety:** The directory `data/pdfs/` is tracked via `.gitkeep`, but all PDF files, subfolders, and documents are automatically ignored by `.gitignore`.
- **Logs:** The directory `logs/` is preserved via `.gitkeep`, while timestamped `*.log` files are ignored.
- **Vector Store:** The directory `vectorstore/` is tracked via `.gitkeep`, while generated ChromaDB sqlite files and `bm25_index.pkl` are ignored.
- **Cache & Agents:** `__pycache__/`, Python bytecode, and `.agents/` directories are completely ignored from Git tracking.
- **Line Endings:** Handled cleanly across Windows, Linux, and macOS through `.gitattributes`.

---

## ❓ Troubleshooting & FAQs

### Q: Ollama returns `Model 'qwen3:8b' not found`
**Solution:** Run `ollama pull qwen3:8b`. If you prefer using an already installed model (e.g. `llama3:8b`), PDF-Chatbot-V2 will automatically fall back to it without crashing.

### Q: How do I force re-indexing after adding new PDFs?
**Solution:** Run `.\run.ps1 --reset` or `python ingest.py --reset`. This wipes the old index and processes all PDFs recursively from `data/pdfs/`.

### Q: Does PDF-Chatbot-V2 support CUDA GPU acceleration?
**Solution:** Yes. If PyTorch with CUDA is installed and an NVIDIA GPU is present, both embedding generation and cross-encoder re-ranking automatically run on `[CUDA]`.

### Q: How does the system handle multi-turn questions?
**Solution:** The conversation memory retains past turns while isolating the current question to prevent historical topic bleed and hallucinations. Use `/clear` at any time to start a fresh topic.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
