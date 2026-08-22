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
# Ingestion — Text Splitting (Structure-Aware)
# ──────────────────────────────────────────────
CHUNK_SIZE = 512            # Characters per chunk (fine-grained precision)
CHUNK_OVERLAP = 64          # Overlap between consecutive chunks
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
# LLM — Ollama (Qwen3:8B)
# ──────────────────────────────────────────────
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = 120                     # Seconds per generation request
OLLAMA_CONTEXT_WINDOW = 32768            # Extended context window

# ──────────────────────────────────────────────
# Retrieval — Hybrid (BM25 + Vector + RRF)
# ──────────────────────────────────────────────
TOP_K = 8                                # Final number of context chunks passed to LLM
USE_HYBRID_SEARCH = True                 # Enable BM25 + Vector fusion
RETRIEVAL_CANDIDATES = 60                # Raw candidate pool depth per retriever
VECTOR_WEIGHT = 0.5                      # Semantic vector contribution in RRF
BM25_WEIGHT = 0.5                        # Keyword BM25 contribution in RRF
RRF_K = 60                               # Reciprocal Rank Fusion constant

USE_RERANKER = True                      # Enable Cross-Encoder re-ranking
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
