"""
PDF-Chatbot-V2 — Central Configuration
======================================
Lightweight, high-performance RAG pipeline for general-purpose document QA
with deep understanding of software development content (Python, Java,
DevOps, shell scripting, fullstack) as well as operational, instructional,
and informational PDFs.

Handles ALL document types:
  - Instructional PDFs, manuals, guides
  - Informational documents, reports, specifications
  - Technical documentation, API references, code-heavy content
  - Academic papers, course materials

All tunable constants live here. Edit this file to adjust chunking,
model selection, retrieval strategy, and paths without touching pipeline logic.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
BM25_INDEX_PATH = VECTORSTORE_DIR / "bm25_index.pkl"
LOG_DIR = PROJECT_ROOT / "logs"


def ensure_directories() -> None:
    """Ensure all required project directories exist."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# Automatically create directories on configuration import
ensure_directories()

# ──────────────────────────────────────────────
# Hardware & Compute Profile Detection
# ──────────────────────────────────────────────
def _check_cpu_mode() -> bool:
    """Determine if CPU mode is explicitly requested or forced."""
    if os.getenv("FORCE_CPU", "0").lower() in ("1", "true", "yes"):
        return True
    if os.getenv("CUDA_VISIBLE_DEVICES") == "":
        return True
    return False


FORCE_CPU = _check_cpu_mode()


def _is_cuda_available() -> bool:
    """Safe check for CUDA availability without heavyweight imports."""
    if FORCE_CPU:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


IS_CUDA_AVAILABLE = _is_cuda_available()
IS_CPU_MODE = FORCE_CPU or not IS_CUDA_AVAILABLE

# ──────────────────────────────────────────────
# Compute Profiles (GPU vs. CPU Optimized)
# ──────────────────────────────────────────────
# GPU Profile (Default when CUDA is available): Max reasoning depth & high candidate pool
GPU_PROFILE = {
    "OLLAMA_MODEL": "qwen2.5:7b",
    "OLLAMA_CONTEXT_WINDOW": 8192,
    "TOP_K": 10,
    "RETRIEVAL_CANDIDATES": 80,
    "OLLAMA_TIMEOUT": 120,
    "USE_RERANKER": True,
}

# CPU Profile (Optimized for fast CPU inference, lightweight memory bandwidth & low TTFT)
CPU_PROFILE = {
    "OLLAMA_MODEL": "qwen2.5:1.5b",
    "OLLAMA_CONTEXT_WINDOW": 4096,
    "TOP_K": 4,
    "RETRIEVAL_CANDIDATES": 15,
    "OLLAMA_TIMEOUT": 300,
    "USE_RERANKER": True,
}

_ACTIVE_PROFILE = CPU_PROFILE if IS_CPU_MODE else GPU_PROFILE

# ──────────────────────────────────────────────
# Ingestion — Text Splitting (Structure-Aware)
# ──────────────────────────────────────────────
CHUNK_SIZE = 1000           # Characters per chunk (preserves full paragraphs and multi-step procedures)
CHUNK_OVERLAP = 150         # Overlap between consecutive chunks (prevents severed context)
BATCH_SIZE = 10             # PDFs processed per batch (OOM guard)
CODE_BLOCK_AWARE = True     # Respect code fences, indentation, and structure boundaries

# ──────────────────────────────────────────────
# Embedding Model (HuggingFace)
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIMENSION = 768   # Output embedding dimension for nomic-embed-text-v1.5
EMBEDDING_PREFIX = "search_document: "   # Nomic-embed task prefix for documents
QUERY_PREFIX = "search_query: "          # Nomic-embed task prefix for queries

# ──────────────────────────────────────────────
# Vector Store (ChromaDB)
# ──────────────────────────────────────────────
COLLECTION_NAME = "pdf_chatbot_v2"

# ──────────────────────────────────────────────
# LLM — Ollama Configuration
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", _ACTIVE_PROFILE["OLLAMA_MODEL"])
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", str(_ACTIVE_PROFILE["OLLAMA_TIMEOUT"])))
OLLAMA_CONTEXT_WINDOW = int(
    os.getenv("OLLAMA_CONTEXT_WINDOW", str(_ACTIVE_PROFILE["OLLAMA_CONTEXT_WINDOW"]))
)

# ──────────────────────────────────────────────
# Retrieval — Hybrid (BM25 + Vector + RRF + Re-ranker)
# ──────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", str(_ACTIVE_PROFILE["TOP_K"])))
USE_HYBRID_SEARCH = True                 # Enable BM25 + Vector fusion
RETRIEVAL_CANDIDATES = int(
    os.getenv("RETRIEVAL_CANDIDATES", str(_ACTIVE_PROFILE["RETRIEVAL_CANDIDATES"]))
)
VECTOR_WEIGHT = 0.5                      # Semantic vector contribution in RRF
BM25_WEIGHT = 0.5                        # Keyword BM25 contribution in RRF
RRF_K = 60                               # Reciprocal Rank Fusion constant

USE_RERANKER = os.getenv("USE_RERANKER", "1").lower() in ("1", "true", "yes")
RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"

# ──────────────────────────────────────────────
# OCR Fallback Settings (RapidOCR)
# ──────────────────────────────────────────────
ENABLE_OCR = True                        # Enable RapidOCR fallback for scanned PDFs
OCR_MIN_CHARACTERS = 50                  # Run OCR if extracted text is shorter than threshold
OCR_DPI = 200                            # Pixmap DPI for OCR rendering

# ──────────────────────────────────────────────
# Conversation Memory
# ──────────────────────────────────────────────
MAX_MEMORY_TURNS = 5                     # Sliding window of past Q-A turns

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ──────────────────────────────────────────────
# Web Server (FastAPI)
# ──────────────────────────────────────────────
SERVE_HOST = os.getenv("SERVE_HOST", "0.0.0.0")
SERVE_PORT = int(os.getenv("SERVE_PORT", "8000"))


def get_active_profile() -> dict:
    """Return dictionary of the active profile parameters and hardware mode."""
    return {
        "is_cpu_mode": IS_CPU_MODE,
        "mode_name": "CPU (Optimized)" if IS_CPU_MODE else "GPU (CUDA)",
        "model": OLLAMA_MODEL,
        "context_window": OLLAMA_CONTEXT_WINDOW,
        "top_k": TOP_K,
        "candidates": RETRIEVAL_CANDIDATES,
        "timeout": OLLAMA_TIMEOUT,
        "reranker": USE_RERANKER,
    }
