#!/bin/bash
# =============================================================================
# PDF-Chatbot-V2 — Docker Entrypoint Script
# =============================================================================
# Starts Ollama daemon, waits for readiness, ensures the configured model
# is pulled, then launches the FastAPI web server.
# =============================================================================

set -e

echo "================================================================"
echo "  PDF-Chatbot-V2 — Docker Container Starting"
echo "================================================================"

# ─────────────────────────────────────────────
# 1. Start Ollama daemon in the background
# ─────────────────────────────────────────────
echo "[*] Starting Ollama daemon..."
ollama serve &
OLLAMA_PID=$!

# ─────────────────────────────────────────────
# 2. Wait for Ollama to become ready
# ─────────────────────────────────────────────
echo "[*] Waiting for Ollama to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[OK] Ollama is ready."
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "   Waiting... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "[ERROR] Ollama failed to start within timeout."
    exit 1
fi

# ─────────────────────────────────────────────
# 3. Pull the configured model if not present
# ─────────────────────────────────────────────
MODEL_NAME="${OLLAMA_MODEL:-qwen3:8b}"
echo "[*] Checking for model '$MODEL_NAME'..."

MODEL_LIST=$(curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = [m['name'] for m in data.get('models', [])]
print(' '.join(models))
" 2>/dev/null || echo "")

if echo "$MODEL_LIST" | grep -q "$MODEL_NAME"; then
    echo "[OK] Model '$MODEL_NAME' is already available."
else
    echo "[*] Pulling model '$MODEL_NAME' (this may take several minutes)..."
    ollama pull "$MODEL_NAME"
    echo "[OK] Model '$MODEL_NAME' pulled successfully."
fi

# ─────────────────────────────────────────────
# 4. Launch FastAPI web server
# ─────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Launching FastAPI Web Server on port 8000"
echo "================================================================"
echo ""

cd /app

exec python3 -m uvicorn web.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info
