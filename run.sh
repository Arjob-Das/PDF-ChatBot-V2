#!/usr/bin/env bash
# =============================================================================
# PDF-Chatbot-V2 — Master Pipeline: Setup → Ollama → Train → Docker → Launch
# =============================================================================
# End-to-end orchestrator that handles the complete lifecycle:
#   Step 1: Directory initialization
#   Step 2: Python check & installation
#   Step 3: Virtual environment & dependencies
#   Step 4: Ollama installation, daemon start & model pull
#   Step 5: Document ingestion / training (local)
#   Step 6: Docker image build
#   Step 7: Start Docker container & open browser
#
# Usage:
#   ./run.sh                  # Full pipeline (default: train → Docker → browser)
#   ./run.sh -cli              # Train → launch interactive CLI chatbot (no Docker)
#   ./run.sh --skip-ingest    # Skip training, use existing index
#   ./run.sh --reset          # Wipe index and re-train from scratch
#   ./run.sh --no-docker      # Skip Docker, run local web server instead
#   ./run.sh --local          # Same as -cli
#   ./run.sh --no-browser     # Don't auto-open browser at the end
# =============================================================================

set -e

# Parse arguments
SKIP_INGEST=0
RESET=0
NO_DOCKER=0
LOCAL_MODE=0
NO_BROWSER=0
FORCE_CPU=0
PDF_DIR=""
BATCH_SIZE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-ingest) SKIP_INGEST=1; shift ;;
        --reset)       RESET=1; shift ;;
        --no-docker)   NO_DOCKER=1; shift ;;
        --local|-cli|--cli) LOCAL_MODE=1; shift ;;
        --no-browser)  NO_BROWSER=1; shift ;;
        --cpu|-cpu)    FORCE_CPU=1; shift ;;
        --pdf-dir)     PDF_DIR="$2"; shift 2 ;;
        --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONIOENCODING=utf-8

# Determine total steps
TOTAL_STEPS=7
if [ "$LOCAL_MODE" -eq 1 ] || [ "$NO_DOCKER" -eq 1 ]; then
    TOTAL_STEPS=6
fi

echo ""
echo -e "\033[1;36m================================================================\033[0m"
echo -e "\033[1;36m  PDF-Chatbot-V2 — Full Pipeline Launcher\033[0m"
echo -e "\033[1;36m  Setup → Ollama → Train → Docker → Launch\033[0m"
echo -e "\033[1;36m================================================================\033[0m"
echo ""

# =============================================================================
# Step 1: Directory Initialization
# =============================================================================
echo -e "[Step 1/$TOTAL_STEPS] Initializing directories..."
mkdir -p "$SCRIPT_DIR/data/pdfs" "$SCRIPT_DIR/logs" "$SCRIPT_DIR/vectorstore"
echo -e "  \033[0;32m[OK] Directories verified.\033[0m"

# =============================================================================
# Step 2: Python Check & Installation
# =============================================================================
echo -e "[Step 2/$TOTAL_STEPS] Checking Python..."
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo -e "  \033[1;33m[!] Python not found. Attempting install...\033[0m"
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
        echo -e "  \033[0;31m[ERROR] Cannot install Python. Install Python 3.10+ manually.\033[0m"
        exit 1
    fi
fi
echo -e "  \033[0;32m[OK] Python found: $PYTHON_CMD\033[0m"

# =============================================================================
# Step 3: Virtual Environment & Dependencies
# =============================================================================
echo -e "[Step 3/$TOTAL_STEPS] Setting up virtual environment & dependencies..."
VENV_DIR="$SCRIPT_DIR/venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "  \033[1;33m[*] Creating virtual environment...\033[0m"
    "$PYTHON_CMD" -m venv "$VENV_DIR" || {
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y python3-venv
            "$PYTHON_CMD" -m venv "$VENV_DIR"
        fi
    }
    echo -e "  \033[0;32m[OK] Virtual environment created.\033[0m"
else
    echo -e "  \033[0;32m[OK] Virtual environment exists.\033[0m"
fi

# Check core dependencies
MISSING_CORE=0
"$VENV_PYTHON" -c "
pkgs = ['pymupdf', 'chromadb', 'sentence_transformers', 'rich', 'rank_bm25', 'requests', 'rapidocr_onnxruntime', 'torch', 'einops']
for p in pkgs:
    __import__(p)
" 2>/dev/null || MISSING_CORE=1

if [ "$MISSING_CORE" -eq 1 ]; then
    echo -e "  \033[1;33m[*] Installing core dependencies...\033[0m"
    "$VENV_PIP" install --upgrade pip >/dev/null 2>&1 || true

    if command -v nvidia-smi &>/dev/null; then
        echo -e "  \033[1;33m[*] NVIDIA GPU detected — installing CUDA PyTorch...\033[0m"
        "$VENV_PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || "$VENV_PIP" install torch torchvision
    fi

    "$VENV_PIP" install -r "$SCRIPT_DIR/requirements.txt"
    echo -e "  \033[0;32m[OK] Core dependencies installed.\033[0m"
else
    echo -e "  \033[0;32m[OK] Core dependencies present.\033[0m"
fi

# Check web dependencies
MISSING_WEB=0
"$VENV_PYTHON" -c "import fastapi; import uvicorn; import sse_starlette" 2>/dev/null || MISSING_WEB=1

if [ "$MISSING_WEB" -eq 1 ]; then
    echo -e "  \033[1;33m[*] Installing web dependencies...\033[0m"
    "$VENV_PIP" install -r "$SCRIPT_DIR/web/requirements.txt" --quiet
    echo -e "  \033[0;32m[OK] Web dependencies installed.\033[0m"
fi

# =============================================================================
# Step 4: Ollama Installation, Daemon Start & Model Pull
# =============================================================================
echo -e "[Step 4/$TOTAL_STEPS] Setting up Ollama..."

# 4a. Find or install Ollama
OLLAMA_INSTALLED=0
if command -v ollama &>/dev/null; then
    OLLAMA_INSTALLED=1
    echo -e "  \033[0;32m[OK] Ollama found.\033[0m"
else
    echo -e "  \033[1;33m[!] Ollama not found. Installing...\033[0m"
    if [ "$(uname)" = "Darwin" ] && command -v brew &>/dev/null; then
        brew install ollama
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
    OLLAMA_INSTALLED=1
fi

# 4b. Start Ollama daemon if not running
OLLAMA_RUNNING=0
if curl -s -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    OLLAMA_RUNNING=1
fi

if [ "$OLLAMA_RUNNING" -eq 0 ] && [ "$OLLAMA_INSTALLED" -eq 1 ]; then
    echo -e "  \033[1;33m[*] Starting Ollama daemon...\033[0m"
    ollama serve >/dev/null 2>&1 &
    RETRIES=15
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
    echo -e "  \033[0;32m[OK] Ollama daemon is running.\033[0m"
else
    echo -e "  \033[1;33m[WARNING] Ollama daemon not reachable. Docker will handle Ollama internally.\033[0m"
fi

# 4c. Pull model if needed
if [ "$OLLAMA_RUNNING" -eq 1 ]; then
    INSTALLED_COUNT=$(ollama list 2>/dev/null | sed 1d | grep -v '^$' | wc -l | tr -d ' ' || echo "0")
    if [ "$INSTALLED_COUNT" -gt 0 ]; then
        FIRST_MODEL=$(ollama list 2>/dev/null | sed 1d | awk 'NR==1{print $1}')
        echo -e "  \033[0;32m[OK] Model available: '$FIRST_MODEL'\033[0m"
    else
        TARGET_MODEL="qwen3:8b"
        echo -e "  \033[1;33m[*] No models found. Pulling '$TARGET_MODEL' (may take several minutes)...\033[0m"
        ollama pull "$TARGET_MODEL"
        echo -e "  \033[0;32m[OK] Model '$TARGET_MODEL' pulled successfully.\033[0m"
    fi
fi

# =============================================================================
# Step 5: Document Ingestion / Training (Always Local)
# =============================================================================
echo -e "[Step 5/$TOTAL_STEPS] Document ingestion & training..."

RUN_INGESTION=1
if [ "$SKIP_INGEST" -eq 1 ]; then
    RUN_INGESTION=0
    echo -e "  \033[0;90m[*] Skipping ingestion (--skip-ingest).\033[0m"
fi

# Check existing vectorstore
if [ "$RUN_INGESTION" -eq 1 ] && [ "$RESET" -eq 0 ]; then
    VECTORSTORE_PATH="$SCRIPT_DIR/vectorstore"
    BM25_PATH="$VECTORSTORE_PATH/bm25_index.pkl"

    if [ -d "$VECTORSTORE_PATH" ] && [ -f "$BM25_PATH" ]; then
        INDEX_STATUS=$("$VENV_PYTHON" -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import chromadb
client = chromadb.PersistentClient(path='$VECTORSTORE_PATH')
try:
    col = client.get_collection('pdf_chatbot_v2'); count = col.count()
    print(f'EXISTS:{count}') if count > 0 else print('EMPTY')
except: print('MISSING')
" 2>/dev/null || echo "MISSING")

        if echo "$INDEX_STATUS" | grep -q "^EXISTS:"; then
            CHUNK_COUNT=$(echo "$INDEX_STATUS" | cut -d: -f2)
            echo -e "  \033[0;32m[OK] Existing index found ($CHUNK_COUNT chunks). Skipping training.\033[0m"
            echo -e "       Use --reset to force full re-ingestion."
            RUN_INGESTION=0
        fi
    fi
fi

if [ "$RUN_INGESTION" -eq 1 ]; then
    PDF_DIRECTORY="${PDF_DIR:-$SCRIPT_DIR/data/pdfs}"
    PDF_COUNT=$(find "$PDF_DIRECTORY" -name "*.pdf" 2>/dev/null | wc -l | tr -d ' ')

    if [ "$PDF_COUNT" -eq 0 ]; then
        echo -e "  \033[1;33m[!] No PDF files found in '$PDF_DIRECTORY'.\033[0m"
        if [ ! -f "$SCRIPT_DIR/vectorstore/bm25_index.pkl" ]; then
            echo -e "  \033[0;31m[ERROR] No PDFs and no existing index. Add PDFs to 'data/pdfs/' and re-run.\033[0m"
            exit 1
        fi
        echo -e "  \033[0;90m[*] Using existing vectorstore index.\033[0m"
    else
        echo -e "  \033[1;36m[*] Found $PDF_COUNT PDF(s). Starting ingestion pipeline...\033[0m"

        INGEST_ARGS="--ocr"
        [ "$RESET" -eq 1 ] && INGEST_ARGS="$INGEST_ARGS --reset"
        [ -n "$PDF_DIR" ] && INGEST_ARGS="$INGEST_ARGS --pdf-dir $PDF_DIR"
        [ "$BATCH_SIZE" -gt 0 ] && INGEST_ARGS="$INGEST_ARGS --batch-size $BATCH_SIZE"

        "$VENV_PYTHON" "$SCRIPT_DIR/ingest.py" $INGEST_ARGS

        echo -e "  \033[0;32m[OK] Training complete!\033[0m"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# MODE: --local → Original interactive CLI chat
# ─────────────────────────────────────────────────────────────────────────────
if [ "$LOCAL_MODE" -eq 1 ]; then
    echo -e "[Step 6/$TOTAL_STEPS] Launching interactive CLI chat..."
    echo ""
    exec "$VENV_PYTHON" "$SCRIPT_DIR/main.py" --skip-ingest
fi

# ─────────────────────────────────────────────────────────────────────────────
# MODE: --no-docker → Local FastAPI web server
# ─────────────────────────────────────────────────────────────────────────────
if [ "$NO_DOCKER" -eq 1 ]; then
    echo -e "[Step 6/$TOTAL_STEPS] Launching local web server (no Docker)..."

    if [ "$OLLAMA_RUNNING" -eq 0 ]; then
        echo -e "  \033[0;31m[ERROR] Ollama not running. Start Ollama first or use Docker mode.\033[0m"
        exit 1
    fi

    echo ""
    echo -e "  \033[0;32m┌─────────────────────────────────────────────┐\033[0m"
    echo -e "  \033[0;32m│  Web UI:  http://localhost:8000              │\033[0m"
    echo -e "  \033[0;32m│  Press Ctrl+C to stop                       │\033[0m"
    echo -e "  \033[0;32m└─────────────────────────────────────────────┘\033[0m"
    echo ""

    if [ "$NO_BROWSER" -eq 0 ]; then
        if command -v xdg-open &>/dev/null; then xdg-open "http://localhost:8000" &
        elif command -v open &>/dev/null; then open "http://localhost:8000" &
        fi
    fi

    exec "$VENV_PYTHON" -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
fi

# =============================================================================
# Step 6: Docker Build
# =============================================================================
echo -e "[Step 6/$TOTAL_STEPS] Building Docker serving image..."

# 6a. Check Docker
if ! command -v docker &>/dev/null; then
    echo -e "  \033[1;33m[!] Docker not found. Install Docker: https://docs.docker.com/get-docker/\033[0m"
    echo -e "  \033[1;33m[*] Falling back to local web server mode...\033[0m"

    if [ "$OLLAMA_RUNNING" -eq 0 ]; then
        echo -e "  \033[0;31m[ERROR] Neither Docker nor Ollama available.\033[0m"
        exit 1
    fi

    echo ""
    echo -e "  \033[0;32m┌─────────────────────────────────────────────┐\033[0m"
    echo -e "  \033[0;32m│  Web UI:  http://localhost:8000              │\033[0m"
    echo -e "  \033[0;32m│  Press Ctrl+C to stop                       │\033[0m"
    echo -e "  \033[0;32m└─────────────────────────────────────────────┘\033[0m"
    echo ""
    if [ "$NO_BROWSER" -eq 0 ]; then
        if command -v xdg-open &>/dev/null; then xdg-open "http://localhost:8000" &
        elif command -v open &>/dev/null; then open "http://localhost:8000" &
        fi
    fi
    exec "$VENV_PYTHON" -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
fi

# 6b. Check Docker daemon
if ! docker info >/dev/null 2>&1; then
    echo -e "  \033[1;33m[!] Docker daemon not running. Start Docker first.\033[0m"
    echo -e "  \033[1;33m[*] Falling back to local web server...\033[0m"

    if [ "$OLLAMA_RUNNING" -eq 0 ]; then
        echo -e "  \033[0;31m[ERROR] Neither Docker daemon nor Ollama running.\033[0m"
        exit 1
    fi

    echo ""
    if [ "$NO_BROWSER" -eq 0 ]; then
        if command -v xdg-open &>/dev/null; then xdg-open "http://localhost:8000" &
        elif command -v open &>/dev/null; then open "http://localhost:8000" &
        fi
    fi
    exec "$VENV_PYTHON" -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
fi

echo -e "  \033[0;32m[OK] Docker is available.\033[0m"

# 6c. Verify vectorstore
if [ ! -f "$SCRIPT_DIR/vectorstore/bm25_index.pkl" ]; then
    echo -e "  \033[0;31m[ERROR] No trained index found. Run without --skip-ingest first.\033[0m"
    exit 1
fi

# 6d. Build image
echo -e "  \033[1;33m[*] Building 'pdf-chatbot-v2:latest'...\033[0m"
DOCKER_BUILDKIT=1 docker build -f Dockerfile.serve -t pdf-chatbot-v2:latest .
echo -e "  \033[0;32m[OK] Docker image built.\033[0m"

# =============================================================================
# Step 7: Start Docker Container & Open Browser
# =============================================================================
echo -e "[Step 7/$TOTAL_STEPS] Starting Docker container..."

# 7a. Remove existing container
if docker ps -aq --filter "name=pdf-chatbot-v2" | grep -q .; then
    echo -e "  \033[1;33m[*] Removing existing container...\033[0m"
    docker stop pdf-chatbot-v2 >/dev/null 2>&1 || true
    docker rm pdf-chatbot-v2 >/dev/null 2>&1 || true
fi

# 7b. Detect GPU
USE_GPU=false
if [ "$FORCE_CPU" -eq 1 ]; then
    echo -e "  \033[1;33m[*] CPU-only mode requested via --cpu flag.\033[0m"
elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    USE_GPU=true
fi

# 7c. Stop local Ollama (Docker runs its own — avoid port conflict)
if [ "$OLLAMA_RUNNING" -eq 1 ]; then
    echo -e "  \033[1;33m[*] Stopping local Ollama (Docker container runs its own)...\033[0m"
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 2
fi

# 7d. Run container
VECTORSTORE_FULL="$(cd "$SCRIPT_DIR/vectorstore" && pwd)"
OLLAMA_HOST_DIR="${HOME}/.ollama"
mkdir -p "$OLLAMA_HOST_DIR"
HF_HOST_DIR="${HOME}/.cache/huggingface"
mkdir -p "$HF_HOST_DIR"

if [ "$USE_GPU" = true ]; then
    echo -e "  \033[0;32m[*] Starting with GPU acceleration...\033[0m"
    docker run -d \
        --name pdf-chatbot-v2 \
        --gpus all \
        -p 8000:8000 \
        -p 11434:11434 \
        -v "$VECTORSTORE_FULL:/app/vectorstore" \
        -v "$OLLAMA_HOST_DIR:/root/.ollama" \
        -v "$HF_HOST_DIR:/root/.cache/huggingface" \
        --restart unless-stopped \
        pdf-chatbot-v2:latest
else
    echo -e "  \033[1;33m[*] Starting in CPU-only mode...\033[0m"
    docker run -d \
        --name pdf-chatbot-v2 \
        -p 8000:8000 \
        -p 11434:11434 \
        -e CUDA_VISIBLE_DEVICES="" \
        -e OLLAMA_NUM_GPU=0 \
        -v "$VECTORSTORE_FULL:/app/vectorstore" \
        -v "$OLLAMA_HOST_DIR:/root/.ollama" \
        -v "$HF_HOST_DIR:/root/.cache/huggingface" \
        --restart unless-stopped \
        pdf-chatbot-v2:latest
fi

# 7e. Wait for services
echo -e "  \033[1;33m[*] Waiting for services (Ollama model pull + FastAPI startup)...\033[0m"
WEB_READY=0
MAX_WAIT=600
WAITED=0

while [ $WAITED -lt $MAX_WAIT ] && [ "$WEB_READY" -eq 0 ]; do
    sleep 5
    WAITED=$((WAITED + 5))

    # Check container
    STATUS=$(docker inspect --format='{{.State.Status}}' pdf-chatbot-v2 2>/dev/null || echo "stopped")
    if [ "$STATUS" != "running" ]; then
        echo -e "  \033[0;31m[ERROR] Container stopped unexpectedly. Logs:\033[0m"
        docker logs --tail 30 pdf-chatbot-v2
        exit 1
    fi

    # Check web server
    if curl -s -f http://localhost:8000/api/health >/dev/null 2>&1; then
        WEB_READY=1
    else
        echo -e "  ... starting (${WAITED}s elapsed — model download & index loading)..."
    fi
done

# 7f. Final status
echo ""
echo -e "\033[0;32m================================================================\033[0m"
if [ "$WEB_READY" -eq 1 ]; then
    echo -e "\033[0;32m  ✅ PDF-Chatbot-V2 is READY!\033[0m"
else
    echo -e "\033[1;33m  ⏳ Container still starting (model pull may take a while).\033[0m"
    echo -e "\033[1;33m     Monitor: docker logs -f pdf-chatbot-v2\033[0m"
fi
echo -e "\033[0;32m================================================================\033[0m"
echo ""
echo -e "  \033[1;36m🌐 Web UI:       http://localhost:8000\033[0m"
echo -e "  \033[1;36m🤖 Ollama API:   http://localhost:11434\033[0m"
echo -e "  \033[0;90m📋 View logs:    docker logs -f pdf-chatbot-v2\033[0m"
echo -e "  \033[0;90m🛑 Stop:         docker stop pdf-chatbot-v2\033[0m"
echo -e "  \033[0;90m🔄 Restart:      docker restart pdf-chatbot-v2\033[0m"
echo ""

# 7g. Open browser
if [ "$NO_BROWSER" -eq 0 ]; then
    echo -e "  \033[1;33m[*] Opening browser...\033[0m"
    if command -v xdg-open &>/dev/null; then
        xdg-open "http://localhost:8000" &
    elif command -v open &>/dev/null; then
        open "http://localhost:8000" &
    fi
fi
