"""
PDF-Chatbot-V2 — Utilities
==========================
Hardware detection (CUDA / CPU / AVX2), logging setup, code-aware tokenizer,
content type detection, and shared helpers for the lightweight RAG pipeline.
"""

import hashlib
import logging
import platform
import re
import struct
import sys
from datetime import datetime

import config

# Force UTF-8 encoding for stdout and stderr on Windows to avoid emoji encode crashes
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def setup_logging() -> logging.Logger:
    """
    Configure dual-output logging: file + console.
    Returns the root 'pdf_chatbot_v2' logger.
    """
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_file = config.LOG_DIR / f"chatbot_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("pdf_chatbot_v2")
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

    # Prevent duplicate handlers on re-import
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def detect_device() -> str:
    """
    Detect the best available compute device for embeddings.
    Logs detailed hardware info on first call.

    Returns:
        "cuda" if an NVIDIA GPU with CUDA is available and not forced to CPU, else "cpu".
    """
    logger = logging.getLogger("pdf_chatbot_v2.hardware")

    import os
    force_cpu = (
        os.getenv("FORCE_CPU", "0").lower() in ("1", "true", "yes")
        or os.getenv("CUDA_VISIBLE_DEVICES") == ""
    )

    if not force_cpu:
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                vram_free = (
                    torch.cuda.get_device_properties(0).total_memory
                    - torch.cuda.memory_reserved(0)
                ) / (1024 ** 3)
                cuda_version = torch.version.cuda

                logger.info("=" * 60)
                logger.info("🟢 GPU DETECTED — CUDA acceleration enabled")
                logger.info(f"   GPU Name       : {gpu_name}")
                logger.info(f"   VRAM Total     : {vram_total:.1f} GiB")
                logger.info(f"   VRAM Available : {vram_free:.1f} GiB")
                logger.info(f"   CUDA Version   : {cuda_version}")
                logger.info(f"   PyTorch Version: {torch.__version__}")
                logger.info("=" * 60)
                return "cuda"
            else:
                logger.info("No CUDA GPU detected, falling back to CPU.")
        except ImportError:
            logger.warning("PyTorch not installed — cannot detect GPU. Using CPU.")
    else:
        logger.info("⚡ CPU mode explicitly forced via environment variable.")

    # CPU path — log CPU details
    cpu_info = _get_cpu_info()
    logger.info("=" * 60)
    logger.info("🔵 CPU MODE — Running on processor")
    logger.info(f"   CPU            : {cpu_info['name']}")
    logger.info(f"   Architecture   : {cpu_info['arch']}")
    logger.info(f"   AVX2 Support   : {cpu_info['avx2']}")
    logger.info(f"   Python          : {sys.version.split()[0]}")
    logger.info("=" * 60)
    return "cpu"


def _get_cpu_info() -> dict:
    """
    Gather CPU name, architecture, and AVX2 support status.
    Works on both Windows and Linux without external dependencies.
    """
    info = {
        "name": platform.processor() or "Unknown",
        "arch": platform.machine(),
        "avx2": "Unknown",
    }

    # Try to detect AVX2 support
    if platform.system() == "Windows":
        try:
            import subprocess

            result = subprocess.run(
                ["wmic", "cpu", "get", "Name", "/value"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                if line.startswith("Name="):
                    info["name"] = line.split("=", 1)[1].strip()
        except Exception:
            pass

        # AVX2 capability proxy (64-bit Python architecture)
        info["avx2"] = "Likely (64-bit Python)" if struct.calcsize("P") == 8 else "Unknown"
    else:
        # Linux — read /proc/cpuinfo
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            if "avx2" in cpuinfo.lower():
                info["avx2"] = "✅ Supported"
            else:
                info["avx2"] = "❌ Not supported"

            for line in cpuinfo.split("\n"):
                if line.startswith("model name"):
                    info["name"] = line.split(":", 1)[1].strip()
                    break
        except FileNotFoundError:
            pass

    return info


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def hash_document_id(source: str, page: int, chunk_idx: int) -> str:
    """
    Generate a deterministic document ID for deduplication.
    Same file + page + chunk index always produces the same ID.
    """
    raw = f"{source}::page_{page}::chunk_{chunk_idx}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────
# Code-Aware Tokenizer (NEW — for BM25)
# ──────────────────────────────────────────────

# Pattern to split on code-specific delimiters: underscores, dots, arrows,
# double colons, camelCase boundaries, whitespace, and common operators
_CODE_SPLIT_PATTERN = re.compile(
    r"(?<=[a-z])(?=[A-Z])"    # camelCase boundary
    r"|(?<=[A-Z])(?=[A-Z][a-z])"  # XMLParser → XML|Parser
    r"|[_.\->/:\[\](){},;=+*&|!@#$%^~`\s]+"  # delimiters + whitespace
)

# Stopwords for code contexts — common but low-signal tokens
_CODE_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "don", "now", "it", "its", "this", "that",
    "and", "but", "or", "if", "else", "while", "because", "about",
})


def code_aware_tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing with code-awareness.

    Splits on code-specific delimiters (underscores, dots, camelCase,
    arrows, double colons, etc.) and filters out common stopwords.
    Preserves meaningful code tokens like function names, class names,
    error codes, and CLI flags.

    Args:
        text: Raw text string to tokenize.

    Returns:
        List of lowercase token strings.
    """
    # Split using code-aware pattern
    tokens = _CODE_SPLIT_PATTERN.split(text)

    # Lowercase, filter empty strings and stopwords, keep tokens >= 2 chars
    result = []
    for token in tokens:
        t = token.strip().lower()
        if t and len(t) >= 2 and t not in _CODE_STOPWORDS:
            result.append(t)

    return result


# ──────────────────────────────────────────────
# Content Type Detection (NEW — for metadata)
# ──────────────────────────────────────────────

# Patterns that indicate code content
_CODE_INDICATORS = [
    re.compile(r"```"),                           # fenced code blocks
    re.compile(r"^\s*(def |class |import |from .+ import)", re.MULTILINE),  # Python
    re.compile(r"^\s*(public |private |protected |static |void |int |String )", re.MULTILINE),  # Java
    re.compile(r"^\s*(function |const |let |var |=>)", re.MULTILINE),  # JavaScript
    re.compile(r"^\s*(#!/bin/|#!/usr/bin/env )", re.MULTILINE),  # Shell
    re.compile(r"^\s*(FROM |RUN |CMD |COPY |EXPOSE |ENTRYPOINT )", re.MULTILINE),  # Dockerfile
    re.compile(r"^\s*(apiVersion:|kind:|metadata:|spec:)", re.MULTILINE),  # Kubernetes YAML
    re.compile(r"[{}\[\]();]", re.MULTILINE),     # Braces/brackets density
]

# Patterns that indicate table content
_TABLE_INDICATORS = [
    re.compile(r"\|.*\|.*\|"),                     # Markdown tables
    re.compile(r"^\s*\+[-+]+\+\s*$", re.MULTILINE),  # ASCII tables
]


def detect_content_type(text: str) -> str:
    """
    Classify a text chunk's content type for metadata enrichment.

    Returns:
        One of: "code", "prose", "mixed", "table"
    """
    code_matches = sum(1 for pattern in _CODE_INDICATORS if pattern.search(text))
    table_matches = sum(1 for pattern in _TABLE_INDICATORS if pattern.search(text))

    # Count lines that look like code (high indentation or special characters)
    lines = text.split("\n")
    code_lines = sum(1 for line in lines if line.startswith(("    ", "\t")) or "=" in line)
    code_ratio = code_lines / max(len(lines), 1)

    if table_matches >= 1:
        return "table"
    elif code_matches >= 3 or code_ratio > 0.6:
        return "code"
    elif code_matches >= 1 or code_ratio > 0.3:
        return "mixed"
    else:
        return "prose"
