#!/usr/bin/env bash
# =============================================================================
# PDF-Chatbot-V2 — Master Pipeline Launcher & Auto-Setup (Linux / macOS / WSL)
# =============================================================================
# 1. Pre-creates all necessary project directories (data/pdfs, logs, vectorstore)
# 2. Verifies / installs Python (>=3.10)
# 3. Creates & activates Python virtual environment (venv)
# 4. Verifies & installs Python dependencies (requirements.txt + CUDA PyTorch if GPU available)
# 5. Verifies / installs Ollama, starts Ollama daemon, and verifies models
# 6. Executes main.py with all forwarded CLI arguments.
# =============================================================================

set -e

# Change directory to the script's root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure UTF-8 output
export PYTHONIOENCODING=utf-8

echo -e "\033[1;36m================================================================\033[0m"
echo -e "\033[1;36m  PDF-Chatbot-V2 — Environment & Pipeline Setup\033[0m"
echo -e "\033[1;36m================================================================\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Directory Initialization
# ─────────────────────────────────────────────────────────────────────────────
mkdir -p "$SCRIPT_DIR/data/pdfs" "$SCRIPT_DIR/logs" "$SCRIPT_DIR/vectorstore"
echo -e "\033[0;32m[OK] Directories verified (data/pdfs, logs, vectorstore).\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Python Check & Installation
# ─────────────────────────────────────────────────────────────────────────────
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo -e "\033[1;33m[!] Python not found. Attempting installation...\033[0m"
    if command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
        PYTHON_CMD="python3"
    elif command -v brew &>/dev/null; then
        brew install python
        PYTHON_CMD="python3"
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y python3 python3-pip
        PYTHON_CMD="python3"
    else
        echo -e "\033[0;31m[ERROR] Python is not installed. Please install Python 3.10+.\033[0m"
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Virtual Environment Setup
# ─────────────────────────────────────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "\033[1;33m[*] Creating virtual environment at '$VENV_DIR'...\033[0m"
    "$PYTHON_CMD" -m venv "$VENV_DIR" || {
        echo -e "\033[1;33m[*] Attempting venv package install...\033[0m"
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y python3-venv
            "$PYTHON_CMD" -m venv "$VENV_DIR"
        fi
    }
else
    echo -e "\033[0;32m[OK] Virtual environment detected at '$VENV_DIR'.\033[0m"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Python Package Dependencies Installation
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\033[0;90m[*] Verifying Python dependencies...\033[0m"
MISSING_PACKAGES=0
"$VENV_PYTHON" -c "
import sys
pkgs = ['pymupdf', 'chromadb', 'sentence_transformers', 'rich', 'rank_bm25', 'requests', 'rapidocr_onnxruntime', 'torch', 'einops']
for p in pkgs:
    __import__(p)
" 2>/dev/null || MISSING_PACKAGES=1

if [ "$MISSING_PACKAGES" -eq 1 ]; then
    echo -e "\033[1;33m[*] Installing required packages from requirements.txt...\033[0m"
    "$VENV_PIP" install --upgrade pip >/dev/null 2>&1 || true

    # Install CUDA PyTorch if NVIDIA GPU is present
    if command -v nvidia-smi &>/dev/null; then
        echo -e "\033[1;33m[*] NVIDIA GPU detected. Ensuring PyTorch with CUDA support...\033[0m"
        "$VENV_PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || "$VENV_PIP" install torch torchvision
    fi

    "$VENV_PIP" install -r "$SCRIPT_DIR/requirements.txt"
    echo -e "\033[0;32m[OK] Python dependencies installed successfully.\033[0m"
else
    echo -e "\033[0;32m[OK] Python dependencies already installed (skipped reinstall).\033[0m"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. Ollama Setup & Model Verification
# ─────────────────────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    echo -e "\033[1;33m[!] Ollama not found. Attempting automated installation...\033[0m"
    if [ "$(uname)" = "Darwin" ] && command -v brew &>/dev/null; then
        brew install ollama
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
else
    echo -e "\033[0;32m[OK] Ollama installation detected.\033[0m"
fi

# Check if Ollama daemon is running
OLLAMA_RUNNING=0
if curl -s -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    OLLAMA_RUNNING=1
fi

if [ "$OLLAMA_RUNNING" -eq 0 ] && command -v ollama &>/dev/null; then
    echo -e "\033[1;33m[*] Starting Ollama background service...\033[0m"
    ollama serve >/dev/null 2>&1 &
    RETRIES=10
    while [ $RETRIES -gt 0 ] && [ "$OLLAMA_RUNNING" -eq 0 ]; do
        sleep 1
        if curl -s -f http://localhost:11434/api/tags >/dev/null 2>&1; then
            OLLAMA_RUNNING=1
            break
        fi
        RETRIES=$((RETRIES - 1))
    done
fi

if [ "$OLLAMA_RUNNING" -eq 1 ]; then
    echo -e "\033[0;32m[OK] Ollama service is active.\033[0m"
    
    # Check if ANY models are installed
    INSTALLED_COUNT=$(ollama list 2>/dev/null | sed 1d | grep -v '^$' | wc -l || echo "0")
    if [ "$INSTALLED_COUNT" -gt 0 ]; then
        FIRST_MODEL=$(ollama list 2>/dev/null | sed 1d | awk 'NR==1{print $1}')
        echo -e "\033[0;32m[OK] Detected installed Ollama model '$FIRST_MODEL' (skipping model download).\033[0m"
    else
        TARGET_MODEL="qwen3:8b"
        echo -e "\033[1;33m[*] No Ollama models found. Pulling '$TARGET_MODEL' (this may take a few minutes)...\033[0m"
        ollama pull "$TARGET_MODEL"
        echo -e "\033[0;32m[OK] Model '$TARGET_MODEL' ready.\033[0m"
    fi
else
    echo -e "\033[1;33m[!] Note: Ollama daemon not reachable on http://localhost:11434 (offline fallback mode).\033[0m"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Execute Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "\033[1;36m================================================================\033[0m"
echo -e "\033[1;36m  Launching PDF-Chatbot-V2 Pipeline...\033[0m"
echo -e "\033[1;36m================================================================\033[0m"
echo ""

exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" "$@"
